"""Train a reward model (RM) from the DPO preference pairs.

This is the "reward modeling" half of classic RLHF — and the piece DPO skips.
DPO folds the preference signal directly into a policy loss, so it never learns
an explicit reward. Classic RLHF (PPO / GRPO) instead trains a standalone RM
that can score ANY (prompt, answer) pair online, then uses it to guide the
policy with reinforcement learning (see train/grpo.py).

Given a (prompt, chosen, rejected) triple we learn a scalar score s(x, y) such
that s(x, chosen) > s(x, rejected):

    L = -log σ( s(x, y_w) - s(x, y_l) )        # logistic regression on the gap

Architecture:
  - base Qwen model (frozen) + the SFT LoRA adapter (trainable) + one linear
    "score head" (hidden_size -> 1) over the mean-pooled response hidden states.
  - We initialise from SFT so the RM judges from *aligned* representations; only
    the LoRA adapter and the head are updated.

Saving: the LoRA adapter is saved with save_pretrained as usual; the score head
is a plain nn.Linear that peft doesn't know about, so its state_dict is saved
separately as score_head.pt. train/grpo.py's load_reward_model() reassembles the
two.

Run from the project root:
    python -m train.reward_model --sft-adapter models/sft_lora --debug
    python -m train.reward_model --sft-adapter models/sft_lora
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from train.dpo import tokenize_pair

# Reduce CUDA memory fragmentation on the small 4GB card (same as train/sft.py).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data + scoring
# ---------------------------------------------------------------------------

class RewardDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        c_ids, c_mask = tokenize_pair(self.tokenizer, s["prompt"], s["chosen"])
        r_ids, r_mask = tokenize_pair(self.tokenizer, s["prompt"], s["rejected"])
        return c_ids, c_mask, r_ids, r_mask


def collate(batch, pad_id):
    def pad_ints(seqs):
        m = max(len(s) for s in seqs)
        return [s + [pad_id] * (m - len(s)) for s in seqs]

    def pad_float(seqs):
        m = max(len(s) for s in seqs)
        return [s + [0.0] * (m - len(s)) for s in seqs]

    c_ids = torch.tensor(pad_ints([b[0] for b in batch]), dtype=torch.long)
    c_mask = torch.tensor(pad_float([b[1] for b in batch]), dtype=torch.float32)
    r_ids = torch.tensor(pad_ints([b[2] for b in batch]), dtype=torch.long)
    r_mask = torch.tensor(pad_float([b[3] for b in batch]), dtype=torch.float32)
    return c_ids, c_mask, r_ids, r_mask


def score(model, input_ids, answer_mask):
    """Scalar reward for a batch [B, L]: mean-pool the response hidden states,
    then one linear head -> [B]."""
    pad_id = model.config.pad_token_id
    attn = (input_ids != pad_id).long()
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True)
    h = out.hidden_states[-1]                      # [B, L, H]
    mask = answer_mask.to(h.dtype).unsqueeze(-1)   # [B, L, 1], cast to model dtype
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)  # [B, H]
    return model.score(pooled).squeeze(-1)         # [B]


def reward_loss(s_w, s_l):
    return -torch.nn.functional.logsigmoid(s_w - s_l).mean()


def load_reward_model(model_id, reward_dir):
    """Reassemble a trained RM (LoRA adapter + score head), frozen for scoring."""
    from agent.model_loader import load_model

    model, tokenizer = load_model(model_id, adapter_path=str(reward_dir))
    head = nn.Linear(model.config.hidden_size, 1)
    head.load_state_dict(torch.load(Path(reward_dir) / "score_head.pt", map_location="cpu"))
    model.score = head.to(model.device, dtype=torch.float16)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", type=str, required=True)
    ap.add_argument("--output-dir", type=str, default=str(config.ROOT / "models" / "reward_model"))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--debug", action="store_true", help="8 pairs, 5 steps")
    args = ap.parse_args()

    from agent.model_loader import load_model

    train = json.load(open(config.PROCESSED_DIR / "dpo_train.json", encoding="utf-8"))
    val = json.load(open(config.PROCESSED_DIR / "dpo_val.json", encoding="utf-8"))
    if args.debug:
        train = train[:8]

    # The RM starts from the SFT adapter so it judges from aligned features.
    model, tokenizer = load_model(config.MODEL_ID, adapter_path=args.sft_adapter, trainable=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    head = nn.Linear(model.config.hidden_size, 1)
    model.score = head.to(model.device, dtype=torch.float16)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    loader = DataLoader(
        RewardDataset(train, tokenizer),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    print(f"Training reward model on {len(train)} pairs...")
    global_step = 0
    for epoch in range(args.epochs):
        for c_ids, c_mask, r_ids, r_mask in loader:
            dev = model.device
            c_ids, c_mask = c_ids.to(dev), c_mask.to(dev)
            r_ids, r_mask = r_ids.to(dev), r_mask.to(dev)

            s_w = score(model, c_ids, c_mask)
            s_l = score(model, r_ids, r_mask)
            loss = reward_loss(s_w, s_l)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 5 == 0:
                acc = (s_w > s_l).float().mean().item()  # how often chosen wins
                print(f"  step {global_step}: loss={loss.item():.4f} acc={acc:.3f}")
            global_step += 1
            if args.debug and global_step >= 5:
                break
        if args.debug:
            break

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)                                  # LoRA adapter
    torch.save(model.score.state_dict(), Path(args.output_dir) / "score_head.pt")
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved reward model (adapter + score_head.pt) to {args.output_dir}")


if __name__ == "__main__":
    main()
