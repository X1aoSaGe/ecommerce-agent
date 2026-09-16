"""Generate SFT trajectories using DeepSeek as the teacher model.

For each query the teacher decides which tool(s) to call; we execute them for
real against the catalog and record the whole conversation in Qwen's native
<tool_call> format. The recorded `messages` become one SFT sample.

We drive the teacher through DeepSeek's NATIVE function calling (the `tools`
parameter) rather than asking it to emit `<tool_call>` text. That avoids the
"format drift" where deepseek-chat, after receiving a tool result, switches to
its own <|DSML|> markup. We then translate the structured tool_calls back into
Qwen's `<tool_call>` text for the SFT record.

Run from the project root:
    python -m data.generate_trajectories --n-per-type 2     # small smoke run
    python -m data.generate_trajectories --n-per-type 40    # full run
"""
import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

import config
from data.task_templates import generate_queries
from tools import TOOL_SCHEMAS, execute_tool

load_dotenv()


class TeacherClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ["DEEPSEEK_BASE_URL"],
        )
        self.model = os.environ["DEEPSEEK_MODEL"]

    def chat(self, messages: list[dict]):
        """One native-function-calling turn. Returns the assistant message."""
        r = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        return r.choices[0].message


def _tool_calls_as_dicts(msg) -> list[dict]:
    """Extract the OpenAI-format tool_calls dicts from an assistant message."""
    out = []
    for tc in msg.tool_calls or []:
        out.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
        )
    return out


def _qwen_tool_call_text(tool_calls: list[dict]) -> str:
    """Render OpenAI tool_calls as Qwen's `<tool_call>` text blocks."""
    parts = []
    for tc in tool_calls:
        fn = tc["function"]
        try:
            args = json.loads(fn["arguments"]) if fn["arguments"] else {}
        except json.JSONDecodeError:
            args = fn["arguments"]  # keep raw string if somehow not valid JSON
        obj = {"name": fn["name"], "arguments": args}
        parts.append("<tool_call>\n" + json.dumps(obj, ensure_ascii=False) + "\n</tool_call>")
    return "\n".join(parts)


def to_qwen_messages(oai: list[dict]) -> list[dict]:
    """Convert an OpenAI-native conversation to Qwen message format."""
    msgs = []
    for m in oai:
        role = m["role"]
        if role == "system":
            msgs.append({"role": "system", "content": m["content"]})
        elif role == "user":
            msgs.append({"role": "user", "content": m["content"]})
        elif role == "tool":
            msgs.append(
                {"role": "tool", "content": m["content"], "tool_call_id": m["tool_call_id"]}
            )
        elif role == "assistant":
            if m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": _qwen_tool_call_text(m["tool_calls"])})
            else:
                msgs.append({"role": "assistant", "content": m.get("content") or ""})
    return msgs


def collect_one(teacher: TeacherClient, query: str, max_steps: int = 6) -> dict:
    """Drive the teacher through one query, returning the Qwen-format trajectory."""
    oai = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    answered = False
    for _ in range(max_steps):
        msg = teacher.chat(oai)
        tcs = _tool_calls_as_dicts(msg)

        if not tcs:
            # Final answer (no more tool calls).
            oai.append({"role": "assistant", "content": msg.content or ""})
            answered = True
            break

        # Record the assistant tool-call turn, then execute every tool for real.
        assistant = {"role": "assistant", "tool_calls": tcs}
        if msg.content:
            assistant["content"] = msg.content
        oai.append(assistant)

        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            oai.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    if not answered:
        return {"query": query, "messages": to_qwen_messages(oai), "ok": False, "error": "max_steps"}

    return {"query": query, "messages": to_qwen_messages(oai), "ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-type", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    teacher = TeacherClient()
    queries = generate_queries(args.n_per_type, seed=args.seed)
    print(f"Generating {len(queries)} trajectories...")

    samples, failed = [], 0
    for i, q in enumerate(queries):
        traj = collect_one(teacher, q)
        if traj["ok"]:
            samples.append({"messages": traj["messages"]})
        else:
            failed += 1
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} done, {failed} failed")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    split = int(len(samples) * 0.9)
    train, val = samples[:split], samples[split:]
    (config.PROCESSED_DIR / "sft_train.json").write_text(
        json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (config.PROCESSED_DIR / "sft_val.json").write_text(
        json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(train)} train / {len(val)} val. {failed} failed.")


if __name__ == "__main__":
    main()
