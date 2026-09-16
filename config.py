"""Central configuration for the ecommerce_agent project.

Every path, model name, and shared constant lives here so the rest of the
codebase never hardcodes a magic string like "data/raw/products.json".
"""
from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PRODUCTS_PATH = RAW_DIR / "products.json"
REVIEWS_PATH = RAW_DIR / "reviews.json"

# --- Model ---------------------------------------------------------------
# Local:  small model for fast code-debugging (~1GB VRAM, runs on RTX 3050 4GB).
# Cloud:  switch to "Qwen/Qwen2.5-7B-Instruct" for real training / quality.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# --- Shared vocabularies --------------------------------------------------
# The data generator, the tools, and evaluation all rely on these being fixed.
CATEGORIES = ["RPG", "Action", "Puzzle", "Strategy", "Simulation", "Sports"]

# --- Agent ----------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant for a game store. You help users find and "
    "compare games and understand what players think of them.\n"
    "Rules:\n"
    "- Use the tools to answer questions about products. Never invent product "
    "names, prices, or reviews — only report what the tools return.\n"
    "- If a search returns no results, say so and suggest how to adjust the search."
)
MAX_TOOL_STEPS = 5

# --- Training (used in later phases) --------------------------------------
# 2560 covers the longest generated trajectory (max measured 2448 tokens on the
# comparison task). Raised from 2048 so those high-value multi-turn samples
# aren't truncated mid-answer.
MAX_SEQ_LENGTH = 2560
