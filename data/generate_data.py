"""Generate the synthetic e-commerce catalog.

Products and reviews are hand-authored here (not randomly generated) so they
stay internally consistent: every review must match its product's real fields,
otherwise we'd be teaching the model to hallucinate during SFT.

Design rules followed:
  1. Names are clearly fictional (so the base model cannot answer from memory).
  2. Fields are consistent: `category` and `features` use a fixed vocabulary.
  3. Every product has BOTH a positive and a negative review.
  4. There is enough overlap (multiple RPGs under $40) for comparison tasks.

Run from the project root:  python data/generate_data.py
"""
import json
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# Each entry: id, name, category, price, description, features, reviews.
# reviews = [(rating, text), ...]  -- rating in 1..5.
PRODUCTS = [
    # ---------------- RPG ----------------
    {
        "id": "game_001",
        "name": "Crystal Vale Chronicles",
        "category": "RPG",
        "price": 39.99,
        "description": "A story-driven RPG in the crystal kingdom of Vale. Turn-based combat, forgiving difficulty and a guided tutorial make it ideal for newcomers.",
        "features": ["single-player", "beginner-friendly", "turn-based", "story-rich"],
        "reviews": [
            (5, "Perfect first RPG. The tutorial holds your hand and combat never punishes mistakes."),
            (4, "Lovely world and characters, though the plot is a bit predictable."),
            (3, "Enjoyable but the middle chapters drag on with too much walking."),
            (5, "As a total beginner I never felt lost. Highly recommend it."),
            (2, "Too easy for veterans; the difficulty never ramps up."),
        ],
    },
    {
        "id": "game_002",
        "name": "Ashen Crown Saga",
        "category": "RPG",
        "price": 54.99,
        "description": "A grim, hardcore RPG about reclaiming a fallen kingdom. Punishing turn-based combat with permadeath and deep build customization.",
        "features": ["single-player", "hardcore", "turn-based", "story-rich"],
        "reviews": [
            (5, "Brutal but fair. The build system is the deepest I have seen."),
            (4, "Fantastic if you like a challenge; the story is dark and mature."),
            (1, "Way too hard for me. I died in the first hour repeatedly."),
            (5, "Permadeath makes every decision matter. Not for the faint-hearted."),
            (3, "Great systems but the tutorial explains almost nothing."),
        ],
    },
    {
        "id": "game_003",
        "name": "Starlight Drift",
        "category": "RPG",
        "price": 24.99,
        "description": "A cozy open-world RPG about exploring a dreamlike sky world. No combat, gentle quests, and a relaxing soundtrack.",
        "features": ["single-player", "beginner-friendly", "open-world", "story-rich"],
        "reviews": [
            (5, "So relaxing. A wonderful way to unwind after work."),
            (4, "Beautiful art and music, though the quests are all fetch quests."),
            (5, "No combat, no stress, just exploration. Perfect for kids too."),
            (3, "Pretty but shallow; I wanted more substance in the story."),
            (2, "Boring if you are looking for any kind of challenge."),
        ],
    },
    {
        "id": "game_004",
        "name": "Frostbound Legacy",
        "category": "RPG",
        "price": 44.99,
        "description": "An open-world RPG set in a frozen north. Harsh survival mechanics, real-time combat and a large map to conquer.",
        "features": ["single-player", "open-world", "hardcore", "story-rich"],
        "reviews": [
            (5, "The open world is stunning and full of secrets."),
            (4, "Survival mechanics add real tension. Combat is tight."),
            (3, "The cold meter gets annoying fast; too much busywork."),
            (5, "Huge map with hundreds of hours of content."),
            (2, "Runs poorly and the difficulty spikes are unfair."),
        ],
    },
    {
        "id": "game_005",
        "name": "Moonlit Relics",
        "category": "RPG",
        "price": 19.99,
        "description": "A short, charming RPG about collecting ancient relics under moonlight. Simple turn-based battles and a sweet story.",
        "features": ["single-player", "beginner-friendly", "turn-based", "story-rich"],
        "reviews": [
            (5, "Short but sweet. A great value for the price."),
            (4, "Charming writing and lovely pixel art."),
            (3, "Very short; I finished it in four hours."),
            (5, "Perfect entry point for kids into RPGs."),
            (2, "The battles are too simple and repetitive."),
        ],
    },
    # ---------------- Action ----------------
    {
        "id": "game_006",
        "name": "Neon Rift Striker",
        "category": "Action",
        "price": 34.99,
        "description": "A fast, hardcore action game about slicing through neon enemies. Demanding real-time combat with precise timing.",
        "features": ["single-player", "real-time", "hardcore"],
        "reviews": [
            (5, "Incredible combat depth. Every fight is a dance."),
            (4, "Tough but rewarding. Great boss fights."),
            (2, "Too punishing; the timing windows are tiny."),
            (5, "The neon aesthetic is gorgeous."),
            (3, "Short campaign and no difficulty options."),
        ],
    },
    {
        "id": "game_007",
        "name": "Blazing Comet",
        "category": "Action",
        "price": 49.99,
        "description": "An open-world action adventure about a comet rider. Explore a vast world and fight in fluid real-time combat.",
        "features": ["single-player", "open-world", "real-time"],
        "reviews": [
            (5, "The traversal feels amazing, like flying."),
            (4, "Big beautiful world with lots to do."),
            (3, "Combat gets repetitive after a while."),
            (5, "One of the best action games this year."),
            (2, "Too many bugs at launch."),
        ],
    },
    {
        "id": "game_008",
        "name": "Turbo Clash Arena",
        "category": "Action",
        "price": 29.99,
        "description": "A competitive multiplayer arena brawler. Fast real-time matches with ranked play.",
        "features": ["multiplayer", "competitive", "real-time"],
        "reviews": [
            (5, "Addictive ranked mode. I can't stop playing."),
            (4, "Great with friends; matchmaking is decent."),
            (3, "The ranked grind is frustrating without friends."),
            (5, "Tight controls and fast matches."),
            (2, "Full of toxic players and smurfs."),
        ],
    },
    {
        "id": "game_009",
        "name": "Steel Serpent",
        "category": "Action",
        "price": 59.99,
        "description": "A story-rich action game about an assassin in a steampunk city. Real-time combat with stealth elements.",
        "features": ["single-player", "hardcore", "real-time", "story-rich"],
        "reviews": [
            (5, "The story had me hooked from start to finish."),
            (4, "Stealth and combat both feel great."),
            (3, "Some missions force combat when stealth fails."),
            (5, "Gorgeous steampunk world."),
            (2, "Very linear for an action game."),
        ],
    },
    {
        "id": "game_010",
        "name": "Bubble Knight",
        "category": "Action",
        "price": 9.99,
        "description": "A casual, family-friendly action platformer where a knight pops bubbles. Easy to pick up, forgiving controls.",
        "features": ["beginner-friendly", "casual", "single-player", "family-friendly"],
        "reviews": [
            (5, "My six-year-old loves it. Great family game."),
            (4, "Simple fun with charming characters."),
            (3, "A bit too easy for adults."),
            (5, "Perfect price for the amount of content."),
            (2, "Very short; we finished it in a weekend."),
        ],
    },
    # ---------------- Puzzle ----------------
    {
        "id": "game_011",
        "name": "Prism Pathway",
        "category": "Puzzle",
        "price": 14.99,
        "description": "A relaxing puzzle game about bending light through prisms. Gentle difficulty curve, no time pressure.",
        "features": ["beginner-friendly", "casual", "single-player"],
        "reviews": [
            (5, "Beautiful and soothing. The puzzles are clever."),
            (4, "Great difficulty curve for casual players."),
            (3, "Some later puzzles feel samey."),
            (5, "My favorite chill game of the year."),
            (2, "A bit repetitive after the first world."),
        ],
    },
    {
        "id": "game_012",
        "name": "Rune Logic",
        "category": "Puzzle",
        "price": 19.99,
        "description": "A hardcore logic puzzle game about deciphering ancient runes. Brain-bending puzzles with no hints.",
        "features": ["single-player", "hardcore"],
        "reviews": [
            (5, "The most satisfying puzzle game I have played."),
            (4, "Genuinely challenging, no hand-holding."),
            (2, "Too hard; I got stuck and gave up."),
            (5, "Brilliant design for puzzle veterans."),
            (3, "Wish there were optional hints."),
        ],
    },
    {
        "id": "game_013",
        "name": "Gravity Gardens",
        "category": "Puzzle",
        "price": 12.99,
        "description": "A casual puzzle game about growing gardens by flipping gravity. Bright, friendly and easy to learn.",
        "features": ["beginner-friendly", "casual", "single-player", "family-friendly"],
        "reviews": [
            (5, "Cute and clever. Kids and adults both enjoy it."),
            (4, "Simple mechanic, endless variety."),
            (3, "A little easy, but relaxing."),
            (5, "Great value for the price."),
            (2, "The gravity controls feel floaty."),
        ],
    },
    {
        "id": "game_014",
        "name": "Mystic Word Forge",
        "category": "Puzzle",
        "price": 9.99,
        "description": "A casual word puzzle game where you forge words into spells. Daily challenges and a cozy fantasy theme.",
        "features": ["casual", "single-player", "beginner-friendly"],
        "reviews": [
            (5, "Daily challenge keeps me coming back."),
            (4, "Relaxing wordplay with a cute theme."),
            (3, "Too many ads between levels."),
            (5, "Perfect coffee-break game."),
            (2, "Runs out of content quickly."),
        ],
    },
    {
        "id": "game_015",
        "name": "Chroma Conundrum",
        "category": "Puzzle",
        "price": 24.99,
        "description": "A challenging color-mixing puzzle game with abstract visuals. Hard puzzles and a unique color mechanic.",
        "features": ["single-player", "hardcore"],
        "reviews": [
            (5, "The color mechanic is unlike anything else."),
            (4, "Tough but brilliant puzzles."),
            (3, "The abstract visuals can be disorienting."),
            (5, "A must-play for puzzle fans."),
            (2, "The difficulty curve is brutal."),
        ],
    },
    # ---------------- Strategy ----------------
    {
        "id": "game_016",
        "name": "Iron Crown Tactics",
        "category": "Strategy",
        "price": 44.99,
        "description": "A hardcore turn-based tactics game about medieval warfare. Deep positioning, terrain and unit counters.",
        "features": ["single-player", "turn-based", "hardcore"],
        "reviews": [
            (5, "The best tactics game in years. So deep."),
            (4, "Every battle is a puzzle. Rewarding."),
            (2, "Very steep learning curve."),
            (5, "Tons of replay value."),
            (3, "The campaign story is forgettable."),
        ],
    },
    {
        "id": "game_017",
        "name": "Merchant's Domain",
        "category": "Strategy",
        "price": 27.99,
        "description": "A beginner-friendly strategy game about running a trading empire. Simple rules, satisfying growth loop.",
        "features": ["single-player", "beginner-friendly", "strategy"],
        "reviews": [
            (5, "Easy to learn, hard to put down."),
            (4, "Perfect introduction to strategy games."),
            (3, "Gets repetitive in the late game."),
            (5, "The trading loop is so satisfying."),
            (2, "Not enough depth for veterans."),
        ],
    },
    {
        "id": "game_018",
        "name": "Starfall Empire",
        "category": "Strategy",
        "price": 39.99,
        "description": "A space strategy game with both single-player campaigns and multiplayer. Build fleets and expand across the stars.",
        "features": ["single-player", "multiplayer", "strategy"],
        "reviews": [
            (5, "Massive scope. Space strategy at its best."),
            (4, "Multiplayer is great fun with friends."),
            (3, "The single-player campaign is a bit thin."),
            (5, "So many systems to master."),
            (2, "Steep learning curve and poor tutorial."),
        ],
    },
    {
        "id": "game_019",
        "name": "Tiny Kingdom",
        "category": "Strategy",
        "price": 16.99,
        "description": "A casual, beginner-friendly kingdom builder. Simple building placement and cheerful little citizens.",
        "features": ["beginner-friendly", "casual", "single-player", "strategy", "family-friendly"],
        "reviews": [
            (5, "Adorable and surprisingly deep for kids."),
            (4, "Great first strategy game for my son."),
            (3, "A bit limited for adults."),
            (5, "The citizens are so charming."),
            (2, "Too simple; I ran out of things to do."),
        ],
    },
    {
        "id": "game_020",
        "name": "Grand Legion",
        "category": "Strategy",
        "price": 54.99,
        "description": "A hardcore multiplayer strategy game about commanding legions in turn-based battles. Ranked ladder and tournaments.",
        "features": ["hardcore", "turn-based", "multiplayer", "strategy", "competitive"],
        "reviews": [
            (5, "The ranked ladder is brutally competitive and fun."),
            (4, "Deep strategy with a high skill ceiling."),
            (2, "New players get destroyed. Not welcoming."),
            (5, "Tournament scene is alive."),
            (3, "Needs a lot of time to learn."),
        ],
    },
    # ---------------- Simulation ----------------
    {
        "id": "game_021",
        "name": "Sunny Farm Life",
        "category": "Simulation",
        "price": 19.99,
        "description": "A cozy farming simulation. Grow crops, raise animals and befriend villagers at your own relaxed pace.",
        "features": ["beginner-friendly", "casual", "single-player", "simulation"],
        "reviews": [
            (5, "The coziest game I own. I lose hours to it."),
            (4, "Lovely villagers and relaxing farming."),
            (3, "A bit slow to get started."),
            (5, "Perfect for stress relief."),
            (2, "Repetitive chores after a while."),
        ],
    },
    {
        "id": "game_022",
        "name": "Sky Harbor",
        "category": "Simulation",
        "price": 34.99,
        "description": "An airport management simulation. Schedule flights, expand terminals and keep passengers happy.",
        "features": ["single-player", "simulation", "strategy"],
        "reviews": [
            (5, "Deep management sim with real challenge."),
            (4, "Satisfying to watch your airport grow."),
            (3, "The interface is clunky."),
            (5, "So many systems to optimize."),
            (2, "Steep learning curve at the start."),
        ],
    },
    {
        "id": "game_023",
        "name": "Deep Sea Lab",
        "category": "Simulation",
        "price": 29.99,
        "description": "An educational simulation about running an underwater research lab. Manage resources and study sea creatures.",
        "features": ["single-player", "simulation", "educational"],
        "reviews": [
            (5, "My kids learned a ton about ocean life."),
            (4, "Educational and genuinely fun."),
            (3, "The resource management is fiddly."),
            (5, "Beautiful underwater visuals."),
            (2, "Limited content once you finish the missions."),
        ],
    },
    {
        "id": "game_024",
        "name": "Pixel Diner",
        "category": "Simulation",
        "price": 14.99,
        "description": "A casual diner management simulation. Cook, serve and upgrade your restaurant in fast little sessions.",
        "features": ["casual", "beginner-friendly", "single-player", "simulation"],
        "reviews": [
            (5, "Fast, fun and addictive. Great time killer."),
            (4, "Simple but satisfying gameplay loop."),
            (3, "Gets repetitive after a few hours."),
            (5, "Perfect casual sim."),
            (2, "Too many microtransactions."),
        ],
    },
    {
        "id": "game_025",
        "name": "Orbit City",
        "category": "Simulation",
        "price": 49.99,
        "description": "A large-scale city builder in orbit. Plan districts, manage energy and watch your space city thrive.",
        "features": ["single-player", "open-world", "simulation", "strategy"],
        "reviews": [
            (5, "A masterpiece of city building."),
            (4, "Huge maps and deep systems."),
            (3, "Traffic AI can be frustrating."),
            (5, "So much freedom to build."),
            (2, "Performance tanks on large cities."),
        ],
    },
    # ---------------- Sports ----------------
    {
        "id": "game_026",
        "name": "Goal Rush 2026",
        "category": "Sports",
        "price": 49.99,
        "description": "A competitive football game with online multiplayer and real-time matches. Updated rosters and tournaments.",
        "features": ["multiplayer", "competitive", "real-time", "sports"],
        "reviews": [
            (5, "Best football game around. Online is smooth."),
            (4, "Great gameplay and updated rosters."),
            (3, "Microtransactions are annoying."),
            (5, "Multiplayer with friends is a blast."),
            (2, "Servers lag during peak hours."),
        ],
    },
    {
        "id": "game_027",
        "name": "Alpine Slalom",
        "category": "Sports",
        "price": 24.99,
        "description": "A beginner-friendly skiing game. Forgiving controls, gorgeous slopes and a gentle learning curve.",
        "features": ["single-player", "beginner-friendly", "sports"],
        "reviews": [
            (5, "Easy to pick up and very relaxing."),
            (4, "Beautiful slopes and smooth controls."),
            (3, "A bit light on content."),
            (5, "Great for casual play."),
            (2, "Too easy for serious racers."),
        ],
    },
    {
        "id": "game_028",
        "name": "Court Kings",
        "category": "Sports",
        "price": 39.99,
        "description": "A competitive basketball game with multiplayer and career modes. Real-time matches and deep controls.",
        "features": ["multiplayer", "competitive", "sports"],
        "reviews": [
            (5, "The career mode is deep and rewarding."),
            (4, "Multiplayer is competitive and fun."),
            (3, "Controls take time to learn."),
            (5, "Great basketball simulation."),
            (2, "Too many bugs in the career mode."),
        ],
    },
    {
        "id": "game_029",
        "name": "Paddle Pro",
        "category": "Sports",
        "price": 9.99,
        "description": "A casual table-tennis game with simple controls and multiplayer. Easy for anyone to pick up.",
        "features": ["casual", "multiplayer", "beginner-friendly", "sports"],
        "reviews": [
            (5, "Simple, cheap and fun with friends."),
            (4, "Great casual multiplayer."),
            (3, "Shallow; not much depth."),
            (5, "Perfect party game."),
            (2, "The single-player AI is too easy."),
        ],
    },
    {
        "id": "game_030",
        "name": "Drift Rally",
        "category": "Sports",
        "price": 44.99,
        "description": "A hardcore rally racing game with realistic physics. Real-time races, demanding handling and a full career.",
        "features": ["single-player", "real-time", "hardcore", "sports"],
        "reviews": [
            (5, "Realistic physics and thrilling races."),
            (4, "The handling model is superb."),
            (3, "Too hard without a wheel."),
            (5, "Career mode is long and rewarding."),
            (2, "Frustrating for casual players."),
        ],
    },
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    products, reviews = [], []
    for p in PRODUCTS:
        products.append({k: p[k] for k in ("id", "name", "category", "price", "description", "features")})
        for rating, text in p["reviews"]:
            reviews.append({"product_id": p["id"], "rating": rating, "text": text})

    (RAW / "products.json").write_text(
        json.dumps(products, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RAW / "reviews.json").write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(products)} products and {len(reviews)} reviews to data/raw/")


if __name__ == "__main__":
    main()
