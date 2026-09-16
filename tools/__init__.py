"""Tool registry: the single place that maps tool name -> schema and -> callable.

The agent (and later the training data) import TOOL_SCHEMAS to tell the model
what tools exist, and call execute_tool() to actually run one.
"""
from . import get_product_details, get_reviews, search_products

_MODULES = [search_products, get_product_details, get_reviews]

# What the LLM sees: a list of {"type": "function", "function": {...}} entries.
TOOL_SCHEMAS = [{"type": "function", "function": m.SCHEMA} for m in _MODULES]

# name -> Python callable.
_REGISTRY = {m.SCHEMA["name"]: m.run for m in _MODULES}


def execute_tool(name: str, arguments: dict):
    fn = _REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    return fn(**arguments)
