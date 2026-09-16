"""Direct Preference Optimization (DPO) with LoRA, loss implemented by hand.

Why DPO after SFT, and what this loss means:
  SFT made the model call tools correctly, but its answers are mediocre. DPO
  nudges the model to prefer "chosen" answers over "rejected" ones using pairs
  from data/generate_dpo_pairs.py — no reward model, no RL.

  For one pair (x=prompt, y_w=chosen, y_l=rejected) the loss is:

      L = -log σ( β · [ log π_θ(y_w|x) − log π_ref(y_w|x)
                       − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )

  - π_θ   = the policy model (SFT adapter, still training)
  - π_ref = the reference model (SFT adapter, frozen)
  - β     = how tightly we follow the reference (smaller β = bigger changes)

  Intuition: the loss is low when the policy assigns the chosen answer a HIGHER
  probability than the reference does, and the rejected answer a LOWER one.
  The `-log σ(...)` is just logistic regression on that "advantage gap".

Run from the project root:
    python -m train.dpo --sft-adapter models/sft_lora --debug
    python -m train.dpo --sft-adapter models/sft_lora --output-dir models/dpo_lora
"""
import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

import config

# Reduce CUDA memory fragmentation on the small 4GB card (same as train/sft.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data: tokenize (prompt, response) -> (input_ids, answer_mask)
# ---------------------------------------------------------------------------

def tokenize_pair(tokenizer, prompt: str, response: str):
    """Tokenize prompt+response; answer_mask is 1 only on the response tokens."""
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    input_ids = p_ids + r_ids + [eos]
    answer_mask = [0.0] * len(p_ids) + [1.0] * (len(r_ids) + 1)
    return input_ids, answer_mask


class DPODataset(Dataset):
    def __init__(self, samples, tokenizer, ref_c, ref_r):
        self.samples = samples
        self.tokenizer = tokenizer
        self.ref_c = ref_c  # reference log-probs, precomputed, same order
        self.ref_r = ref_r

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        c_ids, c_mask = tokenize_pair(self.tokenizer, s["prompt"], s["chosen"])
        r_ids, r_mask = tokenize_pair(self.tokenizer, s["prompt"], s["rejected"])
        return c_ids, c_mask, r_ids, r_mask, self.ref_c[i], self.ref_r[i]


def collate(batch, pad_id):
    def pad_ints(seqs):
        m = max(len(s) for s in seqs)
        return [s + [pad_id] * (m - len(s)) for s in seqs]

    def pad_float(seqs):
        m = max(len(s) for s in seqs)
        return [s + [0.0] * (m - len(s)) for s in seqs]

    c_ids = pad_ints([b[0] for b in batch])
    c_mask = pad_float([b[1] for b in batch])
    r_ids = pad_ints([b[2] for b in batch])
    r_mask = pad_float([b[3] for b in batch])
    ref_c = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    ref_r = torch.tensor([b[5] for b in batch], dtype=torch.float32)
    return (
        torch.tensor(c_ids, dtype=torch.long),
        torch.tensor(c_mask, dtype=torch.float32),
        torch.tensor(r_ids, dtype=torch.long),
        torch.tensor(r_mask, dtype=torch.float32),
        ref_c,
        ref_r,
    )


# ---------------------------------------------------------------------------
# Log-probability of the *answer* tokens (the model's own output only)
# ---------------------------------------------------------------------------

def answer_logps(model, input_ids, answer_mask):
    """Summed log-prob of the answer tokens for a batch [B, L]."""
    pad_id = model.config.pad_token_id
    attn = (input_ids != pad_id).long()
    logits = model(input_ids=input_ids, attention_mask=attn).logits  # [B, L, V]
    # Shift: logits[:, t] predicts token t+1.
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:]
    token_logps = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    mask = answer_mask[:, 1:]  # align with the shifted positions
    return (token_logps * mask).sum(-1)  # [B]


def dpo_loss(policy_c, policy_r, ref_c, ref_r, beta):
    chosen_adv = policy_c - ref_c
    rejected_adv = policy_r - ref_r
    return -torch.nn.functional.logsigmoid(beta * (chosen_adv - rejected_adv)).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=str(config.ROOT / "models" / "dpo_lora"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="8 pairs, 3 steps")
    args = ap.parse_args()

    from agent.model_loader import load_model

    train = json.load(open(config.PROCESSED_DIR / "dpo_train.json", encoding="utf-8"))
    val = json.load(open(config.PROCESSED_DIR / "dpo_val.json", encoding="utf-8"))
    if args.debug:
        train, val = train[:8], val[:2]

    # Reference model: SFT adapter, frozen. Its log-probs never change, so we
    # compute them once up front and then free the model (halves GPU memory).
    ref_model, tokenizer = load_model(config.MODEL_ID, adapter_path=args.sft_adapter)
    ref_model.config.pad_token_id = tokenizer.pad_token_id
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Precomputing reference log-probs...")
    ref_c, ref_r = [], []
    with torch.no_grad():
        for s in train:
            c_ids, c_mask = tokenize_pair(tokenizer, s["prompt"], s["chosen"])
            r_ids, r_mask = tokenize_pair(tokenizer, s["prompt"], s["rejected"])
            c_ids_t = torch.tensor([c_ids], dtype=torch.long).to(ref_model.device)
            r_ids_t = torch.tensor([r_ids], dtype=torch.long).to(ref_model.device)
            c_mask_t = torch.tensor([c_mask], dtype=torch.float32).to(ref_model.device)
            r_mask_t = torch.tensor([r_mask], dtype=torch.float32).to(ref_model.device)
            ref_c.append(answer_logps(ref_model, c_ids_t, c_mask_t).item())
            ref_r.append(answer_logps(ref_model, r_ids_t, r_mask_t).item())
    del ref_model
    torch.cuda.empty_cache()

    # Policy model: same SFT adapter, but trainable (its weights get DPO'd).
    policy, _ = load_model(config.MODEL_ID, adapter_path=args.sft_adapter, trainable=True)
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.use_cache = False
    # Gradient checkpointing recomputes activations in the backward pass instead
    # of storing them, so the 4GB card survives the full-vocab logits of Qwen.
    policy.enable_input_require_grads()
    policy.gradient_checkpointing_enable()
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    dataset = DPODataset(train, tokenizer, ref_c, ref_r)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    print(f"Training DPO on {len(train)} pairs for {args.epochs} epochs...")
    global_step = 0
    for epoch in range(args.epochs):
        for c_ids, c_mask, r_ids, r_mask, ref_c_t, ref_r_t in loader:
            dev = policy.device
            c_ids, c_mask = c_ids.to(dev), c_mask.to(dev)
            r_ids, r_mask = r_ids.to(dev), r_mask.to(dev)
            ref_c_t, ref_r_t = ref_c_t.to(dev), ref_r_t.to(dev)

            policy_c = answer_logps(policy, c_ids, c_mask)
            policy_r = answer_logps(policy, r_ids, r_mask)

            loss = dpo_loss(policy_c, policy_r, ref_c_t, ref_r_t, args.beta)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 5 == 0:
                print(f"  step {global_step}: loss={loss.item():.4f}")
            global_step += 1
            if args.debug and global_step >= 3:
                break
        if args.debug:
            break

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
