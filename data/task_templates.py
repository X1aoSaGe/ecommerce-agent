"""Query templates for the 5 task types, sampled from the synthetic catalog.

Kept in one place so the SAME templates can be reused in evaluation (Phase 6)
to compare Base vs SFT vs DPO. Each generator derives the query from real
catalog rows, so every generated query is guaranteed answerable.

Task types:
  1. factual      — "What is the price of X?"
  2. comparison   — "Compare X and Y."
  3. constraint   — "Recommend a <feature> <category> under $N."
  4. reviews      — "What do players like and dislike about X?"
  5. multi_step   — find N products + reason over their reviews.
"""
import random

from tools.data_loader import load_products


def _by_category(products):
    d = {}
    for p in products:
        d.setdefault(p["category"], []).append(p)
    return d


def factual(rng, products):
    p = rng.choice(products)
    return f"What is the price of {p['name']}?"


def comparison(rng, products):
    a, b = rng.sample(products, 2)
    return f"Compare {a['name']} and {b['name']}."


def constraint(rng, products):
    p = rng.choice(products)
    # Exclude features that duplicate the category (e.g. "sports" feature on a
    # Sports game), otherwise we'd generate "a sports Sports game".
    feats = [f for f in p["features"] if f.lower() != p["category"].lower()] or p["features"]
    feature = rng.choice(feats)
    price = int(p["price"]) + 1  # +1 so the sampled product itself qualifies
    return f"Recommend a {feature} {p['category']} game under ${price}."


def reviews(rng, products):
    p = rng.choice(products)
    return f"What do players like and dislike about {p['name']}?"


def multi_step(rng, products):
    by_cat = _by_category(products)
    cats = [c for c, ps in by_cat.items() if len(ps) >= 2]
    cat = rng.choice(cats)
    ps = sorted(by_cat[cat], key=lambda p: p["price"])
    price = int(ps[1]["price"]) + 1  # ps[0] and ps[1] are both under price
    return (
        f"Find two {cat} games under ${price} and tell me which one is more "
        "beginner-friendly based on user reviews."
    )


TEMPLATES = [factual, comparison, constraint, reviews, multi_step]


def generate_queries(n_per_type: int, seed: int = 0) -> list[str]:
    """Return n_per_type queries for each task type (shuffled interleaved order)."""
    rng = random.Random(seed)
    products = load_products()
    queries = []
    for _ in range(n_per_type):
        for t in TEMPLATES:
            queries.append(t(rng, products))
    return queries
