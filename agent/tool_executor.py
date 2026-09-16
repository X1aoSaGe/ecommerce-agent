"""Turn raw model output into executed tool calls.

The model emits <tool_call>...</tool_call> blocks; this module parses them out
and runs the matching tool, with defensive handling for malformed output.
"""
import json
import re

from tools import execute_tool

# Non-greedy match over the whole turn, so multiple blocks (and any prose
# around them) are handled correctly.
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract every valid <tool_call> block. Malformed blocks are skipped."""
    calls = []
    for block in TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(block.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            calls.append(obj)
    return calls


def execute_tool_calls(calls: list[dict]) -> list[tuple[dict, dict]]:
    """Execute parsed tool calls, returning [(call, result), ...].

    Any exception inside a tool is caught and turned into an error result, so
    a single bad call can never crash the whole agent loop.
    """
    results = []
    for call in calls:
        name = call.get("name")
        args = call.get("arguments") or {}
        try:
            result = execute_tool(name, args)
        except Exception as e:  # noqa: BLE001 - the agent must never die on a tool error
            result = {"error": f"Tool '{name}' raised: {e}"}
        results.append((call, result))
    return results
