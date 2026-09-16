"""Tool: search products by keyword / max price / category / required features."""
from .data_loader import load_products

SCHEMA = {
    "name": "search_products",
    "description": (
        "Search the catalog for products. Match a free-text query against a "
        "product's name, category, description and features, and optionally "
        "filter by max price, category, and required features."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text keyword(s) to search for, e.g. 'beginner RPG'.",
            },
            "max_price": {
                "type": "number",
                "description": "Optional maximum price in USD.",
            },
            "category": {
                "type": "string",
                "description": "Optional category: one of RPG / Action / Puzzle / Strategy / Simulation / Sports.",
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of features the product must have (all of them).",
            },
        },
        "required": ["query"],
    },
}

# Common filler words to ignore when matching so 'an RPG game under $40' still works.
_STOPWORDS = {
    "a", "an", "the", "game", "games", "for", "under", "over", "with", "and",
    "or", "of", "to", "is", "are", "i", "me", "my", "want", "please", "find",
    "cheap", "good", "best",
}


def _matches(text: str, query: str) -> bool:
    words = [w for w in query.lower().split() if w not in _STOPWORDS]
    if not words:
        return True
    haystack = text.lower()
    return all(w in haystack for w in words)


def run(query: str = "", max_price=None, category=None, features=None):
    results = []
    for p in load_products():
        text = (
            p["name"] + " " + p["category"] + " " + p.get("description", "")
            + " " + " ".join(p.get("features", []))
        )
        if not _matches(text, query):
            continue
        if max_price is not None and p["price"] > max_price:
            continue
        if category and p["category"].lower() != category.lower():
            continue
        if features:
            have = {f.lower() for f in p.get("features", [])}
            if not all(f.lower() in have for f in features):
                continue
        results.append({k: p[k] for k in ("id", "name", "category", "price", "features")})

    return {"query": query, "count": len(results), "results": results}
