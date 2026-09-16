"""Supervised fine-tuning (SFT) with LoRA and explicit completion-only masking.

What this script teaches:
  1. Why masking matters — a causal LM is trained to predict the *next token*
     of the whole conversation, but we only want it to learn the assistant's
     turns (the tool calls + final answers). So we build `labels` where every
     token that is NOT part of an assistant reply is set to -100 (= ignored by
     the cross-entropy loss). System prompt, user question, and tool results
     are all masked out — the model sees them as context but never learns to
     reproduce them.
  2. How the assistant span is found — render the conversation with the chat
     template, then for each assistant message tokenize the prefix "up to and
     including it" and "up to it", and diff the two: the delta is exactly that
     assistant turn's tokens.
  3. What LoRA modifies — only a low-rank adapter (q/k/v/o projections here) is
     trained; the base weights stay frozen. `print_trainable_parameters` shows
     the trainable ratio (~a few % of the model).

Run from the project root:
    python -m train.sft --inspect-mask   # show the masking on one sample, then exit
    python -m train.sft --debug          # smoke run: 16 samples, 5 steps
    python -m train.sft                  # full run (cloud: 7B + QLoRA)
"""
import argparse
import json
import os
import sys

import torch

# Reduce CUDA memory fragmentation on the small 4GB card (see torch docs).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import config
from tools import TOOL_SCHEMAS

# Make Windows console print UTF-8 (Chinese) correctly.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data loading + masking
# ---------------------------------------------------------------------------

def load_samples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _find_sublist(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Return the index of `needle` in `haystack`, searching from `start`."""
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    raise ValueError("assistant span not found in the tokenized conversation")


def tokenize_and_mask(messages: list[dict], tokenizer, tools) -> tuple[list[int], list[int]]:
    """Tokenize one conversation into (input_ids, labels).

    labels[i] == input_ids[i] only where the i-th token belongs to an assistant
    turn (tool call or final answer); everywhere else labels[i] == -100 so the
    loss ignores it.
    """
    full_text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )
    input_ids = _ids(tokenizer, full_text)
    labels = [-100] * len(input_ids)

    offset = 0  # where the next assistant span can be found in input_ids
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # "before" renders messages[0..i) plus a trailing <|im_start|>assistant
        # marker; "after" renders messages[0..i]. Their token diff is exactly
        # the assistant turn's content.
        before = tokenizer.apply_chat_template(
            messages[:i], tools=tools, tokenize=False, add_generation_prompt=True
        )
        after = tokenizer.apply_chat_template(
            messages[: i + 1], tools=tools, tokenize=False, add_generation_prompt=False
        )
        span = _ids(tokenizer, after)[len(_ids(tokenizer, before)) :]
        start = _find_sublist(input_ids, span, offset)
        for k, tok in enumerate(span):
            labels[start + k] = tok
        offset = start + len(span)

    return input_ids, labels


def tokenize_dataset(samples: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenize every sample; truncate (right) the rare one that exceeds max_len."""
    out = []
    for s in samples:
        ids, labels = tokenize_and_mask(s["messages"], tokenizer, TOOL_SCHEMAS)
        if len(ids) > max_len:
            ids = ids[:max_len]
            labels = labels[:max_len]
        out.append({"input_ids": ids, "labels": labels})
    return out


class DataCollatorForSFT:
    """Right-pad a batch to the longest sequence; pad labels with -100."""

    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, mask = [], [], []
        for f in features:
            n = len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad] * (max_len - n))
            labels.append(f["labels"] + [-100] * (max_len - n))
            mask.append([1] * n + [0] * (max_len - n))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Mask inspection (educational helper)
# ---------------------------------------------------------------------------

def inspect_mask(tokenizer):
    """Print the masked vs. learned parts of the first sample, then exit."""
    samples = load_samples(str(config.PROCESSED_DIR / "sft_train.json"))
    ids, labels = tokenize_and_mask(samples[0]["messages"], tokenizer, TOOL_SCHEMAS)
    print("=== TOKENS THE MODEL LEARNS (labels != -100) ===")
    learned = [t for t, l in zip(ids, labels) if l != -100]
    print(tokenizer.decode(learned, skip_special_tokens=False)[:1500])
    print("\n=== TOKENS MASKED OUT (context only) — first 300 chars ===")
    masked = [t for t, l in zip(ids, labels) if l == -100]
    print(tokenizer.decode(masked, skip_special_tokens=False)[:300])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect-mask", action="store_true", help="show masking then exit")
    ap.add_argument("--debug", action="store_true", help="tiny smoke run")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--output-dir", type=str, default=str(config.ROOT / "models" / "sft_lora"))
    ap.add_argument("--qlora", action="store_true", help="4-bit QLoRA (cloud 7B)")
    args = ap.parse_args()

    # MODEL_ID can be overridden at runtime (cloud) without editing config.py.
    model_id = os.environ.get("MODEL_ID", config.MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.inspect_mask:
        inspect_mask(tokenizer)
        return

    if args.qlora:
        # 4-bit QLoRA for the cloud 7B run (needs bitsandbytes — Linux only).
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb)
        model = prepare_model_for_kbit_training(model)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        # fp16 LoRA (local 0.5B debug). `dtype` is the transformers-5 name.
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    model.config.use_cache = False

    # Freeze everything except a low-rank adapter on the chosen projections.
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora)
    # Required for gradient checkpointing when only LoRA params need gradients.
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_samples = load_samples(str(config.PROCESSED_DIR / "sft_train.json"))
    val_samples = load_samples(str(config.PROCESSED_DIR / "sft_val.json"))
    if args.debug:
        train_samples = train_samples[:16]
        val_samples = val_samples[:4]

    print(f"Tokenizing {len(train_samples)} train / {len(val_samples)} val samples...")
    train_data = tokenize_dataset(train_samples, tokenizer, config.MAX_SEQ_LENGTH)
    val_data = tokenize_dataset(val_samples, tokenizer, config.MAX_SEQ_LENGTH)

    steps = args.max_steps if args.max_steps > 0 else -1
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        seed=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=DataCollatorForSFT(tokenizer),
    )
    trainer.train()

    # Save the LoRA adapter (NOT the full model) — tiny, easy to reload.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
