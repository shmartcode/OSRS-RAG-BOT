import json
import random
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def load_jsonl(filename):
    data = []
    path = PROCESSED_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise FileNotFoundError(f"Could not find corpus file at {path}")
    return data


def generate_drop_pairs(monsters):
    pairs = []
    for monster in monsters:
        drops = monster.get("drops", [])
        monster_text = monster.get("text", "")
        if not drops or not isinstance(drops, list):
            continue

        for drop in drops:
            drop_name = drop.get("name", "").lower() if isinstance(drop, dict) else ""
            if not drop_name:
                continue

            queries = [
                f"What monsters drop {drop_name}?",
                f"Who drops {drop_name}?",
                f"{drop_name} drop source",
                f"Where to get {drop_name}",
            ]

            drop_tokens = [t for t in drop_name.split() if len(t) > 3]
            candidate_negatives = []

            for candidate in monsters:
                if candidate.get("id") == monster.get("id"):
                    continue
                cand_drops = [d.get("name", "").lower() for d in candidate.get("drops", []) if isinstance(d, dict)]
                if drop_name in cand_drops:
                    continue

                cand_name = candidate.get("name", "").lower()
                if any(token in cand_name for token in drop_tokens):
                    candidate_negatives.append(candidate)

            if candidate_negatives:
                negative_monster = random.choice(candidate_negatives)
            else:
                negative_monster = random.choice(monsters)

            negative_text = negative_monster.get("text", "")

            for q in queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": monster_text,
                        "negative": negative_text,
                    }
                )
    return pairs


def generate_item_pairs(items):
    pairs = []
    for item in items:
        item_name = item.get("name", "").lower()
        item_text = item.get("text", "")
        if not item_name or not item_text:
            continue

        high_alch = item.get("highalch")
        if high_alch:
            queries = [
                f"What is the high alch value of {item_name}?",
                f"{item_name} high alch value",
                f"How much does {item_name} alch for?",
            ]

            name_tokens = [t for t in item_name.split() if len(t) > 3]
            cand_negatives = [i for i in items if i.get("id") != item.get("id") and any(t in i.get("name", "").lower() for t in name_tokens)]
            neg_item = random.choice(cand_negatives) if cand_negatives else random.choice(items)

            for q in queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": item_text,
                        "negative": neg_item.get("text", ""),
                    }
                )

        reqs = item.get("requirement_stats", {})
        if reqs:
            req_queries = [
                f"What are the requirements to wield {item_name}?",
                f"What stats do I need for {item_name}?",
                f"{item_name} requirements",
            ]
            neg_item = random.choice(items)
            for q in req_queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": item_text,
                        "negative": neg_item.get("text", ""),
                    }
                )

    return pairs


def generate_monster_stats_pairs(monsters):
    pairs = []
    for monster in monsters:
        monster_name = monster.get("name", "").lower()
        monster_text = monster.get("text", "")
        if not monster_name or not monster_text:
            continue

        combat_level = monster.get("combat_level")
        slayer_level = monster.get("slayer_level")

        if combat_level:
            queries = [
                f"What are the stats and combat level of {monster_name}?",
                f"{monster_name} weaknesses and stats",
                f"What is the combat level of {monster_name}?",
            ]
            neg_monster = random.choice(monsters)
            for q in queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": monster_text,
                        "negative": neg_monster.get("text", ""),
                    }
                )

        if slayer_level:
            slayer_queries = [
                f"What slayer level do I need to kill {monster_name}?",
                f"{monster_name} slayer requirement",
            ]
            neg_monster = random.choice(monsters)
            for q in slayer_queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": monster_text,
                        "negative": neg_monster.get("text", ""),
                    }
                )

    return pairs


def generate_prayer_pairs(prayers):
    pairs = []
    for prayer in prayers:
        prayer_name = prayer.get("name", "").lower()
        prayer_text = prayer.get("text", "")
        prayer_requirements = prayer.get("requirements")

        if not prayer_name or not prayer_text:
            continue

        if prayer_requirements:
            queries = [
                f"What level do I need for {prayer_name}?",
                f"What level prayer do I need for {prayer_name}?",
                f"{prayer_name} prayer level requirement",
            ]
            neg_prayer = random.choice(prayers)
            while neg_prayer.get("id") == prayer.get("id"):
                neg_prayer = random.choice(prayers)

            for q in queries:
                pairs.append(
                    {
                        "query": q,
                        "positive": prayer_text,
                        "negative": neg_prayer.get("text", ""),
                    }
                )
    return pairs


def generate_training_data():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading corpus files...")
    monsters = load_jsonl("monsters_corpus.jsonl")
    items = load_jsonl("items_corpus.jsonl")
    prayers = load_jsonl("prayers_corpus.jsonl")

    drop_pairs = generate_drop_pairs(monsters)
    item_pairs = generate_item_pairs(items)
    monster_stat_pairs = generate_monster_stats_pairs(monsters)
    prayer_pairs = generate_prayer_pairs(prayers)

    max_drop_samples = min(len(drop_pairs), (len(item_pairs) + len(monster_stat_pairs)) * 2)
    sampled_drop_pairs = random.sample(drop_pairs, max_drop_samples)

    all_pairs = sampled_drop_pairs + item_pairs + monster_stat_pairs + prayer_pairs
    random.shuffle(all_pairs)

    # 90/10 Train/Validation Split
    split_idx = int(len(all_pairs) * 0.9)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    train_path = PROCESSED_DIR / "train_pairs.json"
    val_path = PROCESSED_DIR / "val_pairs.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_pairs, f, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_pairs, f, indent=2)

    print(f"Dataset summary:")
    print(f"  - Drop Pairs: {len(sampled_drop_pairs)} (sampled down from" f" {len(drop_pairs)})")
    print(f"  - Item Pairs: {len(item_pairs)}")
    print(f"  - Monster Stat Pairs: {len(monster_stat_pairs)}")
    print(f"  - Prayer Pairs: {len(prayer_pairs)}")
    print(f"Total pairs generated: {len(all_pairs)}")
    print(f"Saved {len(train_pairs)} training pairs to {train_path}")
    print(f"Saved {len(val_pairs)} validation pairs to {val_path}")


if __name__ == "__main__":
    generate_training_data()
