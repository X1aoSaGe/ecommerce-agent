"""Group Relative Policy Optimization (GRPO) — the "RL" half of RLHF.

Contrast with DPO (train/dpo.py):
  - DPO is OFFLINE: a static set of preference pairs, one closed-form loss, no
    reward model, no sampling.
  - GRPO is ONLINE: for each prompt it samples a GROUP of G responses from the
    CURRENT policy, scores each with the reward model, and optimizes the policy
    against a group-normalized advantage. It needs a reward model but NO value /
    critic network (that is the difference from classic PPO), which is why it
    fits on a single GPU. GRPO is the algorithm behind DeepSeek-R1.

Per prompt x, sample a group {y_1..y_G} from the current policy π_θ:

    r_i  = RM(x, y_i)                          # reward-model score
    A_i  = (r_i - mean(r)) / (std(r) + ε)      # group-relative advantage
    L    = -mean_i( A_i · log π_θ(y_i|x) ) + β · mean_i KL(π_θ || π_ref)

Why the group baseline: one absolute reward is hard to calibrate; only the
answer's quality RELATIVE to its siblings matters, so we centre each group (this
replaces PPO's learned value/critic with a Monte-Carlo group estimate).
Why the KL term: keep the policy close to the SFT reference so it doesn't drift
into degenerate answers the RM never anticipated.

Note on PPO's "clipped surrogate": GRPO inherits PPO's ratio clipping, but in
this single-pass version each group is sampled and used for exactly one update,
so the importance ratio exp(log π_θ - log π_old) is 1 and the clip is a no-op.
That is why the loss above is plain REINFORCE with a group baseline + KL. If you
run multiple gradient steps per batch you must freeze the "old" log-probs and
re-introduce the clip — left as an exercise.

Three models (all LoRA on the same base):
  - policy    : being trained, starts from the SFT adapter
  - reference : frozen, supplies the KL anchor
  - reward    : frozen reward model from train/reward_model.py

Prompts come from data/processed/dpo_train.json's "prompt" field (the shared
tool-calling context ending in the assistant marker), so the RL loop optimises
the SAME "final answer given grounding" slice that DPO did — tool calling stays
fixed by SFT, and RL only reshapes answer quality.

Run from the project root:
    python -m train.grpo --sft-adapter models/sft_lora --reward models/reward_model --debug
    python -m train.grpo --sft-adapter models/sft_lora --reward models/reward_model
"""
import argparse
import json
import os
import sys

import torch

import config

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Log-prob / KL / advantage helpers
# ---------------------------------------------------------------------------

def per_token_logps(model, input_ids, attention_mask):
    """Log-prob of each target token under `model`: [B, L-1].

    logits[:, t] predicts token t+1, so we shift: drop the last logit column and
    score against input_ids[:, 1:].
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # [B, L, V]
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:]
    return torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)


def kl_penalty(logp_policy, logp_ref, answer_mask):
    """Per-response mean KL over the answer tokens, k3 unbiased estimator
    (Schulman 2020): exp(a-b) - (a-b) - 1 with a=logp_ref, b=logp_policy."""
    diff = logp_ref - logp_policy               # [B, L-1]
    kl_token = diff.exp() - diff - 1.0          # [B, L-1]
    m = answer_mask[:, 1:]                      # align with the shifted positions
    return (kl_token * m).sum(-1) / m.sum(-1).clamp(min=1.0)  # [B]


def advantage(rewards):
    """Group-relative advantage: centre each group, divide by its spread."""
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


# ---------------------------------------------------------------------------
# Sampling + batching
# ---------------------------------------------------------------------------

def sample_responses(policy, tokenizer, prompt, group_size, max_new_tokens):
    """Draw `group_size` responses from the policy for one prompt (greedy-free)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(policy.device)
    prompt_len = inputs["input_ids"].shape[1]
    policy.eval()
    with torch.no_grad():
        out = policy.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=group_size,
        )
    policy.train()
    return [
        tokenizer.decode(out[i][prompt_len:], skip_special_tokens=True)
        for i in range(group_size)
    ]


def pad_group(ids_list, mask_list, pad_id):
    m = max(len(x) for x in ids_list)
    ids = torch.tensor([x + [pad_id] * (m - len(x)) for x in ids_list], dtype=torch.long)
    mask = torch.tensor([x + [0.0] * (m - len(x)) for x in mask_list], dtype=torch.float32)
    return ids, mask


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", type=str, required=True)
    ap.add_argument("--reward", type=str, required=True, help="dir saved by train/reward_model.py")
    ap.add_argument("--output-dir", type=str, default=str(config.ROOT / "models" / "grpo_lora"))
    ap.add_argument("--group-size", type=int, default=4, help="G responses sampled per prompt")
    ap.add_argument("--beta", type=float, default=0.05, help="KL penalty weight")
    ap.add_argument("--lr", type=float, default=1e-5, help="RL uses a smaller LR than SFT")
    ap.add_argument("--steps", type=int, default=100, help="policy update steps")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--debug", action="store_true", help="4 prompts, 5 steps")
    args = ap.parse_args()

    from agent.model_loader import load_model
    from train.dpo import tokenize_pair
    from train.reward_model import load_reward_model, score

    prompts = [
        s["prompt"]
        for s in json.load(open(config.PROCESSED_DIR / "dpo_train.json", encoding="utf-8"))
    ]
    if args.debug:
        prompts = prompts[:4]

    # Policy: SFT adapter, trainable (gets RL'd).
    policy, tokenizer = load_model(config.MODEL_ID, adapter_path=args.sft_adapter, trainable=True)
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.use_cache = False
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()
    policy.train()

    # Reference: the same SFT adapter, frozen (KL anchor).
    ref_model, _ = load_model(config.MODEL_ID, adapter_path=args.sft_adapter)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()

    # Reward model: frozen.
    reward_model, _ = load_reward_model(config.MODEL_ID, args.reward)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    print(f"Running GRPO: {args.steps} steps, G={args.group_size}, beta={args.beta}")
    for step in range(args.steps):
        prompt = prompts[step % len(prompts)]
        responses = sample_responses(
            policy, tokenizer, prompt, args.group_size, args.max_new_tokens
        )

        # Tokenize the whole group (prompt + response_i) into one padded batch.
        ids_list, mask_list = [], []
        for r in responses:
            ids, m = tokenize_pair(tokenizer, prompt, r)
            ids_list.append(ids)
            mask_list.append(m)
        ids, mask = pad_group(ids_list, mask_list, tokenizer.pad_token_id)
        ids = ids.to(policy.device)
        mask = mask.to(policy.device)
        attn = (ids != tokenizer.pad_token_id).long()

        logp_policy = per_token_logps(policy, ids, attn)        # [G, L-1] with grad
        logp_policy_sum = (logp_policy * mask[:, 1:]).sum(-1)   # [G]

        with torch.no_grad():
            logp_ref = per_token_logps(ref_model, ids, attn)    # [G, L-1]
            rewards = score(reward_model, ids, mask)            # [G]

        kl = kl_penalty(logp_policy, logp_ref, mask)            # [G]
        adv = advantage(rewards).detach()                       # [G]

        loss = -(adv * logp_policy_sum).mean() + args.beta * kl.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            print(
                f"  step {step}: loss={loss.item():.4f} "
                f"mean_reward={rewards.mean().item():.3f} kl={kl.mean().item():.4f}"
            )

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved GRPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
