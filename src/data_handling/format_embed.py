import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def build_monster_text(record: dict) -> str:
    """Builds a comprehensive text block containing all monster attributes for embeddings and LLM context."""
    parts = []

    # 1. Identity & Core Overview
    name = record.get("name") or record.get("title", "Unknown Monster")
    combat = record.get("combat_level", "N/A")
    hp = record.get("hitpoints", "N/A")
    max_hit = record.get("max_hit", "N/A")
    parts.append(f"[Monster] {name} | Combat Level: {combat} | HP: {hp} | Max Hit: {max_hit}")

    if examine := record.get("examine"):
        parts.append(f"Examine: {examine}")

    # 2. General Attributes & Flags
    flags = []
    flags.append(f"Members: {'Yes' if record.get('members') else 'No'}")
    if record.get("aggressive"):
        flags.append("Aggressive: Yes")
    if record.get("poisonous"):
        flags.append("Poisonous: Yes")
    if record.get("slayer_level"):
        flags.append(f"Slayer Level Req: {record['slayer_level']}")
    if record.get("slayer_exp"):
        flags.append(f"Slayer XP: {record['slayer_exp']}")
    parts.append("General: " + ", ".join(flags))

    # 3. Combat Parameters & Classifications
    atk_types = record.get("attack_type", [])
    if isinstance(atk_types, list) and atk_types:
        parts.append(f"Attack Type(s): {', '.join(atk_types)}")
    if record.get("attack_speed"):
        parts.append(f"Attack Speed: {record['attack_speed']} ticks")

    attrs = record.get("attributes", [])
    if isinstance(attrs, list) and attrs:
        parts.append(f"Attributes: {', '.join(attrs)}")

    masters = record.get("slayer_masters", [])
    if isinstance(masters, list) and masters:
        parts.append(f"Slayer Masters: {', '.join(masters)}")

    # 4. Levels & Combat Stats
    level_fields = [
        ("Attack", record.get("attack_level")),
        ("Strength", record.get("strength_level")),
        ("Defence", record.get("defence_level")),
        ("Magic", record.get("magic_level")),
        ("Ranged", record.get("ranged_level")),
    ]
    active_levels = [f"{lbl}: {val}" for lbl, val in level_fields if val is not None]
    if active_levels:
        parts.append("Levels: " + ", ".join(active_levels))

    # 5. Offensive Bonuses & Defensive Stats (filtering out 0s where appropriate, keeping key ones)
    offensive = []
    for lbl, key in [
        ("Melee Atk", "attack_bonus"),
        ("Melee Str", "strength_bonus"),
        ("Magic Atk", "attack_magic"),
        ("Magic Bonus", "magic_bonus"),
        ("Ranged Atk", "attack_ranged"),
        ("Ranged Bonus", "ranged_bonus"),
    ]:
        val = record.get(key, 0)
        if val != 0:
            offensive.append(f"{lbl}: {val:+d}" if isinstance(val, int) else f"{lbl}: {val}")
    if offensive:
        parts.append("Offensive Stats: " + ", ".join(offensive))

    defensive = []
    for lbl, key in [
        ("Stab Def", "defence_stab"),
        ("Slash Def", "defence_slash"),
        ("Crush Def", "defence_crush"),
        ("Magic Def", "defence_magic"),
        ("Ranged Def", "defence_ranged"),
    ]:
        val = record.get(key, 0)
        defensive.append(f"{lbl}: {val:+d}" if isinstance(val, int) else f"{lbl}: {val}")
    if defensive:
        parts.append("Defensive Stats: " + ", ".join(defensive))

    # 6. Drop Table Summary
    drop_strs = []
    for d in record.get("drops", []):
        d_name = d.get("name")
        rarity = d.get("rarity")
        if d_name and rarity is not None:
            if rarity >= 1.0:
                drop_strs.append(d_name)
            else:
                pct = f"{rarity * 100:.2f}%".rstrip("0").rstrip(".")
                drop_strs.append(f"{d_name} ({pct}%)")

    drops_summary = ", ".join(drop_strs) if drop_strs else "None"
    parts.append(f"Drops: {drops_summary}")

    return "\n".join(parts)


def build_item_text(record: dict) -> str:
    """Builds a comprehensive text block containing all item attributes for embeddings and LLM context."""
    parts = []

    # 1. Base Identity
    name = record.get("name") or record.get("title", "Unknown Item")
    parts.append(f"[Item] {name}")

    if examine := record.get("examine"):
        parts.append(f"Examine: {examine}")

    # 2. Economy & Properties
    details = []
    details.append(f"Members: {'Yes' if record.get('members') else 'No'}")
    details.append(f"Tradeable: {'Yes' if record.get('tradeable') else 'No'}")
    if record.get("quest_item"):
        details.append("Quest Item: Yes")
    if value := record.get("value"):
        details.append(f"Base Value: {value:,} gp")
    if lowalch := record.get("lowalch"):
        details.append(f"Low Alch: {lowalch:,} gp")
    if highalch := record.get("highalch"):
        details.append(f"High Alch: {highalch:,} gp")
    if weight := record.get("weight"):
        details.append(f"Weight: {weight} kg")
    parts.append("Properties: " + ", ".join(details))

    # 3. Requirement Stats
    reqs = record.get("requirement_stats", {})
    if isinstance(reqs, dict) and reqs:
        req_list = [f"{k.replace('_', ' ').title()} {v}" for k, v in reqs.items() if v]
        if req_list:
            parts.append(f"Requirements: {', '.join(req_list)}")

    # 4. Equipment Stats (filtering out 0 values)
    eq = record.get("equipment_stats", {})
    if isinstance(eq, dict) and eq:
        active_eq = []
        for stat_name, stat_val in eq.items():
            if isinstance(stat_val, (int, float)) and stat_val != 0:
                formatted_val = f"+{stat_val}" if stat_val > 0 else f"{stat_val}"
                active_eq.append(f"{stat_name.replace('_', ' ').title()}: {formatted_val}")
            elif isinstance(stat_val, str) and stat_val:
                active_eq.append(f"{stat_name.title()}: {stat_val}")
        if active_eq:
            parts.append("Equipment Stats: " + ", ".join(active_eq))

    # 5. Weapon Stats
    wpn = record.get("weapon_stats", {})
    if isinstance(wpn, dict) and wpn:
        wpn_list = []
        for k, v in wpn.items():
            if v:
                clean_key = k.replace("_", " ").title()
                if isinstance(v, list):
                    wpn_list.append(f"{clean_key}: {', '.join(map(str, v))}")
                elif isinstance(v, dict):
                    sub_str = ", ".join([f"{sk}: {sv}" for sk, sv in v.items()])
                    wpn_list.append(f"{clean_key}: [{sub_str}]")
                else:
                    wpn_list.append(f"{clean_key}: {v}")
        if wpn_list:
            parts.append("Weapon Properties: " + ", ".join(wpn_list))

    return "\n".join(parts)


def build_prayer_text(record: dict) -> str:
    """Builds a comprehensive text block containing all prayer attributes for embeddings and LLM context."""
    parts = []

    name = record.get("name", "Unknown Prayer")
    parts.append(f"[Prayer] {name}")

    if desc := record.get("description"):
        parts.append(f"Description: {desc}")

    parts.append(f"Members: {'Yes' if record.get('members') else 'No'}")
    parts.append(f"Drain Rate: {record.get('drain_per_minute', 0)} points/min")

    reqs = record.get("requirements", {})
    if isinstance(reqs, dict) and reqs:
        req_str = ", ".join([f"{k.replace('_', ' ').title()} {v}" for k, v in reqs.items() if v])
        parts.append(f"Requirements: {req_str}")
    else:
        parts.append("Requirements: None")

    bonuses = record.get("bonuses", {})
    if isinstance(bonuses, dict) and bonuses:
        bonus_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in bonuses.items() if v])
        parts.append(f"Bonuses: {bonus_str}")

    return "\n".join(parts)


def process_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}", flush=True)

    model_name = "shmartcode/fine_tuned_osrs_embedder_v2"
    model = SentenceTransformer(model_name, device=device)
    print(f"Loading embedding model: {model_name}...")

    corpus_files = [
        "data/processed/clean_wiki_articles.jsonl",
        "data/processed/items_corpus.jsonl",
        "data/processed/monsters_corpus.jsonl",
        "data/processed/prayers_corpus.jsonl",
        "data/processed/ge_prices_corpus.jsonl",
    ]

    for file_name in corpus_files:
        if not os.path.exists(file_name):
            print(f"Skipping {file_name} (File not found)")
            continue

        print(f"\nProcessing corpus: {file_name}")
        documents = []
        metadata = []

        with open(file_name, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)

                    # 1. Format text based on domain category
                    if "monsters" in file_name:
                        category = "Monster"
                        text = build_monster_text(record)
                    elif "items" in file_name:
                        category = "Item"
                        text = build_item_text(record)
                    elif "prayers" in file_name:
                        category = "Prayer"
                        text = build_prayer_text(record)
                    elif "ge_prices" in file_name:
                        category = "GE_Price"
                        text = f"[GE Price] {record.get('name')} | Price: {record.get('price')} gp"
                    else:
                        category = "Wiki"
                        text = record.get("text", record.get("content", ""))

                    if text:
                        documents.append(text)
                        title = record.get("title", record.get("name", "Unknown"))

                        # 2. Preserve raw structured fields in metadata entry for Python logic
                        meta_entry = {
                            "source": file_name,
                            "id": record.get("id"),
                            "title": title,
                            "category": category,
                            "text": text,
                        }

                        # Preserve entity-specific fields for programmatic access
                        if "items" in file_name:
                            meta_entry["equipment_stats"] = record.get("equipment_stats", {})
                            meta_entry["weapon_stats"] = record.get("weapon_stats", {})
                            meta_entry["requirement_stats"] = record.get("requirement_stats", {})
                            meta_entry["highalch"] = record.get("highalch", 0)

                        elif "monsters" in file_name:
                            if "drops" in record:
                                meta_entry["drops"] = record["drops"]
                            if "slayer_level" in record:
                                meta_entry["slayer_level"] = record["slayer_level"]
                            meta_entry["combat_level"] = record.get("combat_level")
                            meta_entry["hitpoints"] = record.get("hitpoints")

                        elif "prayers" in file_name:
                            meta_entry["drain_per_minute"] = record.get("drain_per_minute", 0)
                            meta_entry["requirements"] = record.get("requirements", {})

                        metadata.append(meta_entry)

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line {line_num} in {file_name}: {e}")

        if documents:
            print(f"Generating embeddings for {len(documents)} records in {file_name}...")
            embeddings = model.encode(
                documents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            print(f"Finished {file_name}. Embedding shape: {embeddings.shape}")

            base_name = file_name.replace(".jsonl", "")
            emb_path = f"{base_name}_embeddings.npy"
            meta_path = f"{base_name}_metadata.json"

            np.save(emb_path, embeddings)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, ensure_ascii=False, indent=2)

            print(f"Saved embeddings to {emb_path} and metadata to {meta_path}.")


if __name__ == "__main__":
    process_embedding()
