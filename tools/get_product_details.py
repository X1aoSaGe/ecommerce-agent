"""Tool: get the full details of a single product by id."""
from .data_loader import load_products

SCHEMA = {
    "name": "get_product_details",
    "description": "Get the full details of a single product by its id.",
    "parameters": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "The product id, e.g. 'game_001'.",
            }
        },
        "required": ["product_id"],
    },
}


def run(product_id: str):
    for p in load_products():
        if p["id"] == product_id:
            return {"found": True, "product": p}
    return {"found": False, "product_id": product_id, "error": "Product not found"}
