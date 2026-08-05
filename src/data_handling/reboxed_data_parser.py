# github url for item/weapon properties "https://github.com/0xNeffarion/osrsreboxed-db"
## Item properties i care about: id, name, members, tradeable, stackable, stacked,
#                               notable, noted, cost, lowalch, highalch, weight, quest_item, examine,
# and two dictionaries: equipment, weapon. each containing the bonuses when equipment said equipment/weapon

import json
from osrsreboxed import items_api, monsters_api, prayers_api
from pathlib import Path

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"


def parse_osrs_items(output_filepath=OUTPUT_DIR / "items_corpus.jsonl"):
    """Loads items from osrsreboxed, extracts key stats/metadata, and writes them out to a JSONL file."""
    print("Loading items database...")
    items = items_api.load()

    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for item in items:
            # Build a flat, structured dictionary for the item
            item_record = {
                "id": item.id,
                "name": item.name,
                "category": "Item",
                "examine": getattr(item, "examine", None),
                "members": getattr(item, "members", False),
                "tradeable": getattr(item, "tradeable", False),
                "value": getattr(item, "cost", 0),
                "lowalch": getattr(item, "lowalch", 0),
                "highalch": getattr(item, "highalch", 0),
                "weight": getattr(item, "weight", 0),
                "quest_item": getattr(item, "quest_item", False),
                "equipment_stats": {},
                # "weapon_stats": {},
                "weapon_stats": item.weapon.__dict__.copy() if hasattr(item, "weapon") and item.weapon else {},
                "requirement_stats": {},
                "text": "",
            }
            eq = getattr(item, "equipment", None)
            if eq:
                eq_dict = eq.__dict__.copy()
                item_record["requirement_stats"] = eq_dict.pop("requirements", {}) or {}
                item_record["equipment_stats"] = eq_dict

            item_record["text"] = flatten_to_text(item_record)
            # Write out as a single line JSON object
            f.write(json.dumps(item_record) + "\n")
            count += 1

    print(f"Successfully processed and saved {count} items to {output_filepath}")


def parse_osrs_monsters(output_filepath=OUTPUT_DIR / "monsters_corpus.jsonl"):
    """Loads monsters from osrsreboxed, extracts combat attributes, and writes them out to a JSONL file."""
    print("Loading monsters database...")
    monsters = monsters_api.load()

    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for monster in monsters:

            raw_drops = getattr(monster, "drops", []) or []
            formatted_drops = []
            for drop in raw_drops:
                # If drops are objects, convert them; if they're already dicts, this handles both safely
                if hasattr(drop, "__dict__"):
                    formatted_drops.append(drop.__dict__)
                else:
                    formatted_drops.append(drop)
            monster_record = {
                "id": monster.id,
                "name": monster.name,
                "category": "Monster",
                "examine": getattr(monster, "examine", None),
                "members": getattr(monster, "members", False),
                "combat_level": getattr(monster, "combat_level", None),
                "hitpoints": getattr(monster, "hitpoints", None),
                "max_hit": getattr(monster, "max_hit", None),
                "attack_type": getattr(monster, "attack_type", []),
                "attack_speed": getattr(monster, "attack_speed", None),
                "aggressive": getattr(monster, "aggressive", False),
                "poisonous": getattr(monster, "poisonous", False),
                "attributes": getattr(monster, "attributes", []),
                "slayer_level": getattr(monster, "slayer_level", None),
                "slayer_exp": getattr(monster, "slayer_exp", None),
                "slayer_masters": getattr(monster, "slayer_masters", []),
                "attack_level": getattr(monster, "attack_level", 1),
                "strength_level": getattr(monster, "strength_level", 1),
                "defence_level": getattr(monster, "defence_level", 1),
                "magic_level": getattr(monster, "magic_level", 1),
                "ranged_level": getattr(monster, "ranged_level", 1),
                "attack_bonus": getattr(monster, "attack_bonus", 0),
                "strength_bonus": getattr(monster, "strength_bonus", 0),
                "attack_magic": getattr(monster, "attack_magic", 0),
                "magic_bonus": getattr(monster, "magic_bonus", 0),
                "attack_ranged": getattr(monster, "attack_ranged", 0),
                "ranged_bonus": getattr(monster, "ranged_bonus", 0),
                "defence_stab": getattr(monster, "defence_stab", 0),
                "defence_slash": getattr(monster, "defence_slash", 0),
                "defence_crush": getattr(monster, "defence_crush", 0),
                "defence_magic": getattr(monster, "defence_magic", 0),
                "defence_ranged": getattr(monster, "defence_ranged", 0),
                "drops": formatted_drops,
                # Construct natural language summary for text-based model indexing
                # "text": (f"Monster: {monster.name}. Examine: {getattr(monster, 'examine', 'N/A')}."),
                "text": "",
            }

            monster_record["text"] = flatten_to_text(monster_record)
            f.write(json.dumps(monster_record) + "\n")
            count += 1

    print(f"Successfully processed and saved {count} monsters to" f" {output_filepath}")


def parse_osrs_prayers(output_filepath=OUTPUT_DIR / "prayers_corpus.jsonl"):
    """Loads prayers from osrsreboxed, extracts drain rates and stat bonuses, and writes them out to a JSONL file."""
    print("Loading prayers database...")
    prayers = prayers_api.load()

    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for prayer in prayers:
            # Build a flat, structured dictionary for the prayer
            prayer_record = {
                "id": getattr(prayer, "id", None),
                "name": prayer.name,
                "category": "Prayer",
                "members": getattr(prayer, "members", False),
                "description": getattr(prayer, "description", None),
                "drain_per_minute": getattr(prayer, "drain_per_minute", 0),
                "requirements": getattr(prayer, "requirements", {}),
                "bonuses": getattr(prayer, "bonuses", {}),
                "icon": getattr(prayer, "icon", None),
                # "book": getattr(prayer, "book", "standard"),
                # Construct a natural language summary string for vector embeddings
                # "text": (f"Prayer: {prayer.name}. Description: {getattr(prayer, "description", None)}"),
                "text": "",
            }

            # If the prayer has specific stat modifier bonuses, capture them
            # if hasattr(prayer, "bonuses") and prayer.bonuses:
            #     bonuses = prayer.bonuses
            #     prayer_record["bonuses"] = {
            #         "attack": getattr(bonuses, "attack", 0),
            #         "strength": getattr(bonuses, "strength", 0),
            #         "defence": getattr(bonuses, "defence", 0),
            #         "ranged": getattr(bonuses, "ranged", 0),
            #         "magic": getattr(bonuses, "magic", 0),
            #     }

            prayer_record["text"] = flatten_to_text(prayer_record)
            f.write(json.dumps(prayer_record) + "\n")
            count += 1

    print(f"Successfully processed and saved {count} prayers to {output_filepath}")


def flatten_to_text(data):
    """Recursively converts nested dicts, lists, and primitives into a clean text string."""
    parts = []

    if isinstance(data, dict):
        for key, value in data.items():
            # Skip internal or unhelpful keys if needed (e.g., raw IDs or URLs)
            if key in ["id", "icon", "wiki_url", "text"]:
                continue
            # Format key-value pairs nicely
            sub_text = flatten_to_text(value)
            if sub_text:
                parts.append(f"{key.replace('_', ' ')}: {sub_text}")

    elif isinstance(data, list):
        for item in data:
            sub_text = flatten_to_text(item)
            if sub_text:
                parts.append(sub_text)

    else:
        # Primitive values (strings, numbers, booleans)
        if data is not None and str(data).strip():
            parts.append(str(data))

    return ", ".join(parts)


if __name__ == "__main__":
    parse_osrs_items()
    parse_osrs_monsters()
    parse_osrs_prayers()
