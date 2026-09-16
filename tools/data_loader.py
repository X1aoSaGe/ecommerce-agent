"""Load the synthetic catalog once and cache it (the data is tiny)."""
import functools
import json

import config


@functools.lru_cache(maxsize=1)
def load_products():
    with open(config.PRODUCTS_PATH, encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def load_reviews():
    with open(config.REVIEWS_PATH, encoding="utf-8") as f:
        return json.load(f)
