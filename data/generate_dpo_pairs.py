"""Build DPO preference pairs: (prompt, chosen, rejected).

Why this design (and why DPO comes AFTER SFT):
  - SFT teaches the model to *call tools and stay grounded*, but its final
    answers are still mediocre (too terse, miss the point, weak reasoning).
  - DPO teaches it to *prefer better answers*. To build the preference signal
    we need, for each query, a "chosen" (better) and "rejected" (worse) answer.

How the pair is made:
  - Run the teacher (DeepSeek) through the full agent loop. Its tool calls +
    tool results become the shared *context*, and its final answer is `chosen`.
  - Feed that SAME context to the SFT model and let it produce a final answer —
    that lower-quality answer is `rejected`.
  - Because both answers share the same tool results, DPO isolates exactly one
    dimension: "given this grounding context, which answer is better".

Run from the project root:
    python -m data.generate_dpo_pairs --n-per-type 6 --sft-adapter models/sft_lora
"""
import argparse
import json
import sys

import config
from agent.model_loader import load_model
from data.generate_trajectories import TeacherClient, collect_one
from data.task_templates import generate_queries
from tools import TOOL_SCHEMAS

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def generate_rejected(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Greedy-decode the SFT model's final answer given the shared context."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=6)
    ap.add_argument("--sft-adapter", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    teacher = TeacherClient()
    sft_model, sft_tok = load_model(config.MODEL_ID, adapter_path=args.sft_adapter)

    queries = generate_queries(args.n_per_type, seed=args.seed)
    pairs, skipped = [], 0
    print(f"Building DPO pairs for {len(queries)} queries...")

    for i, q in enumerate(queries):
        traj = collect_one(teacher, q)
        if not traj["ok"]:
            skipped += 1
            continue
        msgs = traj["messages"]
        chosen = msgs[-1]["content"]  # teacher's final answer
        context = msgs[:-1]  # system + user + tool calls + tool results

        prompt = sft_tok.apply_chat_template(
            context, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False
        )
        rejected = generate_rejected(sft_model, sft_tok, prompt).strip()

        # Drop useless pairs: empty rejection, or rejection identical to chosen.
        if not rejected or rejected == chosen.strip():
            skipped += 1
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} done, {len(pairs)} kept, {skipped} skipped")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split = int(len(pairs) * 0.9)
    train, val = pairs[:split], pairs[split:]
    (config.PROCESSED_DIR / "dpo_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.PROCESSED_DIR / "dpo_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(train)} train / {len(val)} val DPO pairs. {skipped} skipped.")


if __name__ == "__main__":
    main()
