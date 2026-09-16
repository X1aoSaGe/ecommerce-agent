"""Compare Base vs SFT vs DPO vs RLHF(GRPO) on the 5 task types.

Two scoring channels:
  1. Rule-based (no LLM needed, cheap and objective):
     - tool_used : did the agent call at least one tool?
     - grounded  : every product name the answer mentions must appear in the
                   agent's own tool results (i.e. no hallucination).
  2. LLM judge (DeepSeek): the answer is scored 1-5 against a ground-truth
     answer produced by the teacher model.

For each eval query the teacher first produces the ground truth once; every
model is then judged against that same reference.

Run from the project root:
    python -m evaluation.evaluate --sft-adapter models/sft_lora --dpo-adapter models/dpo_lora
"""
import argparse
import json
import re
import sys

import config
from agent.agent import Agent
from data.generate_trajectories import TeacherClient, collect_one
from data.task_templates import generate_queries
from tools.data_loader import load_products

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Eval queries use a fresh seed (not 0/1 which generated the training data), so
# the benchmark is not polluted by data the models already saw.
EVAL_SEED = 42

_ALL_NAMES = {p["name"] for p in load_products()}


def is_grounded(answer: str, steps: list[dict]) -> bool:
    """True if every product name in the answer appears in some tool result."""
    tool_text = " ".join(json.dumps(s.get("result", ""), ensure_ascii=False) for s in steps)
    mentioned = [n for n in _ALL_NAMES if n in (answer or "")]
    return all(n in tool_text for n in mentioned)


def judge_answer(client, question: str, ground_truth: str, answer: str) -> int:
    """Ask the teacher to rate an answer 1-5 against the ground truth."""
    prompt = (
        "You are evaluating a game-store assistant's answer.\n"
        f"Question: {question}\n\n"
        f"Ground-truth answer (correct):\n{ground_truth}\n\n"
        f"Model's answer to judge:\n{answer}\n\n"
        "Rate how correct and helpful the model's answer is compared to the "
        "ground truth. Consider factual correctness (names, prices), whether it "
        "answers the question, and whether any constraints are satisfied. "
        "Reply with ONLY one integer from 1 (wrong) to 5 (excellent)."
    )
    r = client.client.chat.completions.create(
        model=client.model, messages=[{"role": "user", "content": prompt}], temperature=0.0
    )
    text = r.choices[0].message.content
    m = re.search(r"\b([1-5])\b", text or "")
    return int(m.group(1)) if m else 1


def evaluate_model(agent: Agent, queries: list[str], ground_truths: dict, judge) -> dict:
    """Run one model over all queries and aggregate its metrics."""
    used_tool = grounded = success = 0
    scores = []
    for q in queries:
        out = agent.run(q, verbose=False)
        answer = out["answer"]
        steps = out["steps"]

        has_tool = len(steps) > 0
        ok_grounded = is_grounded(answer, steps)
        score = judge_answer(judge, q, ground_truths[q], answer or "(no answer)") if answer else 1

        used_tool += has_tool
        grounded += ok_grounded
        success += 1 if (ok_grounded and score >= 4) else 0
        scores.append(score)

    n = len(queries)
    return {
        "n": n,
        "tool_usage": used_tool / n,
        "grounded": grounded / n,
        "task_success": success / n,
        "avg_judge": sum(scores) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", type=str, default=str(config.ROOT / "models" / "sft_lora"))
    ap.add_argument("--dpo-adapter", type=str, default=str(config.ROOT / "models" / "dpo_lora"))
    ap.add_argument("--rlhf-adapter", type=str, default=str(config.ROOT / "models" / "grpo_lora"),
                    help="GRPO/RLHF adapter; included in the comparison only if it exists")
    ap.add_argument("--n-per-type", type=int, default=4)
    args = ap.parse_args()

    queries = generate_queries(args.n_per_type, seed=EVAL_SEED)
    print(f"Evaluating on {len(queries)} fresh queries...")

    # Ground truth: teacher runs each query once.
    teacher = TeacherClient()
    ground_truths = {}
    for q in queries:
        traj = collect_one(teacher, q)
        ground_truths[q] = traj["messages"][-1]["content"] if traj["ok"] else "(unknown)"

    from pathlib import Path

    # Base is always evaluated; each trained adapter is added only if it exists.
    models = [("Base", Agent())]
    for label, path in [("SFT", args.sft_adapter), ("DPO", args.dpo_adapter), ("RLHF", args.rlhf_adapter)]:
        if Path(path).exists():
            models.append((label, Agent(adapter_path=path)))
        else:
            print(f"(skipping {label}: no adapter at {path})")

    results = {}
    for label, agent in models:
        print(f"\n=== {label} model ===")
        results[label] = evaluate_model(agent, queries, ground_truths, teacher)
        for k, v in results[label].items():
            if k != "n":
                print(f"  {k:14s} {v:.3f}")

    labels = [label for label, _ in models]
    print("\n=== Summary ===")
    header = f"{'metric':14s} | " + " | ".join(f"{l:>6s}" for l in labels)
    print(header)
    print("-" * len(header))
    for metric in ["tool_usage", "grounded", "task_success", "avg_judge"]:
        row = f"{metric:14s} | " + " | ".join(f"{results[l][metric]:6.3f}" for l in labels)
        print(row)

    out_path = config.PROCESSED_DIR / "eval_results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
