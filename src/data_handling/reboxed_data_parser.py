import json
from pathlib import Path
from osrsreboxed import items_api, monsters_api, prayers_api

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_osrs_items(output_filepath=OUTPUT_DIR / "items_corpus.jsonl"):
    print("Loading items database...")
    items = items_api.load()
    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for item in items:
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
                "weapon_stats": (item.weapon.__dict__.copy() if hasattr(item, "weapon") and item.weapon else {}),
                "requirement_stats": {},
            }
            eq = getattr(item, "equipment", None)
            if eq:
                eq_dict = eq.__dict__.copy()
                item_record["requirement_stats"] = eq_dict.pop("requirements", {}) or {}
                item_record["equipment_stats"] = eq_dict

            # # Flatten ALL item attributes (including equipment_stats, weapon_stats, requirement_stats)
            # body = flatten_to_text(item_record)

            # # Explicit Entity Lead + Full Stats Text
            # item_record["text"] = f"Item: {item.name}. {body}"

            f.write(json.dumps(item_record) + "\n")
            count += 1
    print(f"Successfully saved {count} items.")


def parse_osrs_monsters(output_filepath=OUTPUT_DIR / "monsters_corpus.jsonl"):
    print("Loading monsters database...")
    monsters = monsters_api.load()
    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for monster in monsters:
            raw_drops = getattr(monster, "drops", []) or []
            formatted_drops = [d.__dict__ if hasattr(d, "__dict__") else d for d in raw_drops]

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
            }

            # # Flatten ALL monster attributes (combat stats, bonuses, levels, drops)
            # body = flatten_to_text(monster_record)

            # # Explicit Entity Lead + Full Monster Text
            # monster_record["text"] = f"Monster: {monster.name}. {body}"

            f.write(json.dumps(monster_record) + "\n")
            count += 1
    print(f"Successfully saved {count} monsters.")


def parse_osrs_prayers(output_filepath=OUTPUT_DIR / "prayers_corpus.jsonl"):
    print("Loading prayers database...")
    prayers = prayers_api.load()
    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for prayer in prayers:
            prayer_record = {
                "id": getattr(prayer, "id", None),
                "name": prayer.name,
                "category": "Prayer",
                "members": getattr(prayer, "members", False),
                "description": getattr(prayer, "description", None),
                "drain_per_minute": getattr(prayer, "drain_per_minute", 0),
                "requirements": getattr(prayer, "requirements", {}),
                "bonuses": getattr(prayer, "bonuses", {}),
            }

            # body = flatten_to_text(prayer_record)
            # prayer_record["text"] = f"Prayer: {prayer.name}. {body}"

            f.write(json.dumps(prayer_record) + "\n")
            count += 1
    print(f"Successfully saved {count} prayers.")


if __name__ == "__main__":
    parse_osrs_items()
    parse_osrs_monsters()
    parse_osrs_prayers()
