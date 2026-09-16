"""The agent loop: alternate between calling the LLM and executing tools.

Flow:
    messages = [system, user]
    loop:
        text = model(messages)
        calls = parse_tool_calls(text)
        if no calls -> text is the final answer, done
        else -> execute calls, append assistant + tool messages, repeat

Run demos from the project root:   python -m agent.agent
Run a single query:                python -m agent.agent "your question"
"""
import json
import sys

import config
from agent.model_loader import load_model
from agent.tool_executor import execute_tool_calls, parse_tool_calls
from tools import TOOL_SCHEMAS


class Agent:
    def __init__(self, model_id=None, adapter_path=None, system_prompt=None, max_new_tokens=512):
        """`adapter_path` loads a trained LoRA adapter (SFT or DPO) on top of the
        base model. None = the bare base model (used as the Phase 6 baseline)."""
        model_id = model_id or config.MODEL_ID
        self.model, self.tokenizer = load_model(model_id, adapter_path=adapter_path)
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.max_new_tokens = max_new_tokens

    def _generate(self, messages: list[dict]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def run(self, user_query: str, max_steps: int = None, verbose: bool = True) -> dict:
        """Run the agent loop and return {"answer", "steps"}.

        `steps` records every tool call + result, which is useful for
        debugging and (later) evaluation.
        """
        max_steps = max_steps or config.MAX_TOOL_STEPS
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        steps = []

        for step in range(1, max_steps + 1):
            text = self._generate(messages)
            calls = parse_tool_calls(text)

            if not calls:
                if verbose:
                    print(f"[step {step}] 最终回答（无工具调用）")
                return {"answer": text, "steps": steps}

            messages.append({"role": "assistant", "content": text})
            for i, (call, result) in enumerate(execute_tool_calls(calls)):
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result, ensure_ascii=False),
                        "tool_call_id": str(i),
                    }
                )
                steps.append(
                    {
                        "tool": call.get("name"),
                        "arguments": call.get("arguments"),
                        "result": result,
                    }
                )
                if verbose:
                    args = json.dumps(call.get("arguments"), ensure_ascii=False)
                    print(f"[step {step}] 调用 {call.get('name')}({args})")

        return {"answer": None, "steps": steps, "error": f"达到 max_steps={max_steps}"}


def main() -> None:
    # Windows console defaults to cp936; force UTF-8 so Chinese labels render.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    agent = Agent()

    demos = [
        "Recommend a beginner-friendly RPG game under $40.",
        "What is the price of Crystal Vale Chronicles?",
        "What do players like and dislike about Ashen Crown Saga?",
        "Compare Crystal Vale Chronicles and Starlight Drift.",
    ]
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else demos

    for q in queries:
        print("\n" + "=" * 72)
        print("USER:", q)
        print("=" * 72)
        out = agent.run(q)
        print("-" * 72)
        print("ANSWER:", out["answer"])


if __name__ == "__main__":
    main()
