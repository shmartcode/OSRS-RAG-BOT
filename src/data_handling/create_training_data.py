import json
import random
import os

PROCESSED_DIR = "data/processed"


def load_jsonl(filename):
    data = []
    path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def generate_drop_pairs(monsters):
    pairs = []
    for monster in monsters:
        drops = monster.get("drops", [])
        monster_text = monster.get("text", "")
        if not drops or not isinstance(drops, list):
            continue

        for drop in drops:
            drop_name = drop.get("name", "").lower()
            if not drop_name:
                continue

            queries = [
                f"What monsters drop {drop_name}?",
                f"What monsters drop {drop_name}?",
                f"What do I kill for {drop_name}?",
            ]

            # Pick a random negative monster
            negative_monster = random.choice(monsters)
            neg_drops = [d.get("name", "").lower() for d in negative_monster.get("drops", [])]
            while drop_name in neg_drops or negative_monster.get("id") == monster.get("id"):
                negative_monster = random.choice(monsters)
                neg_drops = [d.get("name", "").lower() for d in negative_monster.get("drops", [])]

            negative_text = negative_monster.get("text", "")

            for q in queries:
                pairs.append({"query": q, "positive": monster_text, "negative": negative_text})
    return pairs


def generate_item_pairs(items):
    pairs = []
    for item in items:
        item_name = item.get("name", "").lower()
        item_text = item.get("text", "")
        high_alch = item.get("high_alch")

        if not item_name or not item_text:
            continue

        # High alch query pair
        if high_alch:

            queries = [
                f"What is the high alch value of {item_name}?",
                f"What are the alch values of {item_name}?",
                f"How much does {item_name} high alch for?",
            ]
            negative_item = random.choice(items)
            while negative_item.get("id") == item.get("id"):
                negative_item = random.choice(items)

            negative_text = negative_item.get("text", "")
            for q in queries:
                pairs.append({"query": q, "positive": item_text, "negative": negative_text})
    return pairs


def generate_monster_stats_pairs(monsters):
    pairs = []
    for monster in monsters:
        monster_name = monster.get("name", "").lower()
        monster_text = monster.get("text", "")
        combat_level = monster.get("combat_level")

        if not monster_name or not monster_text:
            continue

        # Combat level / stats query pair
        if combat_level:
            queries = [
                f"What are the stats and combat level of {monster_name}?",
                f"What are the weaknesses of {monster_name}?",
                f"What weapon should I use against {monster_name}?",
                f"What combat style should I use against {monster_name}?",
            ]
            negative_monster = random.choice(monsters)
            while negative_monster.get("id") == monster.get("id"):
                negative_monster = random.choice(monsters)

            negative_text = negative_monster.get("text", "")
            for q in queries:
                pairs.append({"query": q, "positive": monster_text, "negative": negative_text})
    return pairs


def generate_prayer_pairs(prayers):
    pairs = []
    for prayer in prayers:
        prayer_name = prayer.get("name", "").lower()
        prayer_text = prayer.get("text", "")
        prayer_requierments = prayer.get("requirements")

        if not prayer_name or not prayer_text:
            continue

        # Prayer requirements query pair
        if prayer_requierments:
            queries = [
                f"What are the requirements to use {prayer}?",
                f"What level do I need to use {prayer}?",
                f"What level prayer do I need for {prayer}?",
                f"Using {prayer} requires what level?",
            ]
            negative_prayer = random.choice(prayers)
            while negative_prayer.get("id") == prayer.get("id"):
                negative_prayer = random.choice(prayers)

            negative_text = negative_prayer.get("text", "")
            for q in queries:
                pairs.append({"query": q, "positive": prayer_text, "negative": negative_text})
    return pairs


def generate_training_data():
    print("Loading corpus files...")
    monsters = load_jsonl("monsters_corpus.jsonl")
    items = load_jsonl("items_corpus.jsonl")
    prayers = load_jsonl("prayers_corpus.jsonl")

    all_pairs = []

    print("Generating drop pairs...")
    all_pairs.extend(generate_drop_pairs(monsters))

    print("Generating item value pairs...")
    all_pairs.extend(generate_item_pairs(items))

    print("Generating monster stat pairs...")
    all_pairs.extend(generate_monster_stats_pairs(monsters))

    print("Generating prayer pairs...")
    all_pairs.extend(generate_prayer_pairs(prayers))

    # Save everything together
    output_path = os.path.join(PROCESSED_DIR, "training_pairs.json")
    with open(output_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Successfully generated a total of {len(all_pairs)} multi-intent training pairs!")


if __name__ == "__main__":
    generate_training_data()
