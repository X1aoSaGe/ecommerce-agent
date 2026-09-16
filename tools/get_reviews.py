"""Tool: get user reviews for a product by id."""
from .data_loader import load_reviews

SCHEMA = {
    "name": "get_reviews",
    "description": "Get user reviews for a product by its id.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "The product id, e.g. 'game_001'.",
            },
            "min_rating": {
                "type": "integer",
                "description": "Optional: only return reviews with rating >= this value.",
            },
            "max_reviews": {
                "type": "integer",
                "description": "Optional: cap the number of reviews returned.",
            },
        },
        "required": ["product_id"],
    },
}


def run(product_id: str, min_rating=None, max_reviews=None):
    revs = [r for r in load_reviews() if r["product_id"] == product_id]
    if min_rating is not None:
        revs = [r for r in revs if r["rating"] >= min_rating]
    if max_reviews is not None:
        revs = revs[:max_reviews]
    return {"product_id": product_id, "count": len(revs), "reviews": revs}
