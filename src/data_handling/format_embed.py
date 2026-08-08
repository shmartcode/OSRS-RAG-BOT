import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def build_monster_text(record: dict) -> str:
    """Builds a compact, token-dense text representation for monsters (<150 tokens)."""
    title = record.get("name", record.get("title", "Unknown"))
    combat = record.get("combat_level", "N/A")
    hp = record.get("hitpoints", "N/A")
    slayer = record.get("slayer_level", 0)

    attrs = ", ".join(record.get("attributes", [])) if record.get("attributes") else "None"
    masters = ", ".join(record.get("slayer_masters", [])) if record.get("slayer_masters") else "None"
    examine = record.get("examine", "")

    # Format drops into concise "Item (Rarity%)" pairs
    drop_strs = []
    for d in record.get("drops", []):
        name = d.get("name")
        rarity = d.get("rarity")
        if name and rarity is not None:
            if rarity >= 1.0:
                drop_strs.append(name)
            else:
                pct = f"{rarity * 100:.2f}%".rstrip("0").rstrip(".")
                drop_strs.append(f"{name} ({pct}%)")

    drops_summary = ", ".join(drop_strs) if drop_strs else "None"

    return (
        f"[Monster] {title} | Combat: {combat} | HP: {hp} | Slayer: {slayer} | Attributes: {attrs} | Masters: {masters}\n"
        f"Examine: {examine}\n"
        f"Drops: {drops_summary}"
    )


def build_item_text(record: dict) -> str:
    """Builds a compact text representation for items."""
    name = record.get("name", "Unknown")
    examine = record.get("examine", "")
    members = record.get("members", False)
    value = record.get("value", 0)
    highalch = record.get("highalch", 0)

    reqs = record.get("requirement_stats", {})
    req_str = ", ".join([f"{k}: {v}" for k, v in reqs.items()]) if reqs else "None"

    eq = record.get("equipment_stats", {})
    slot = eq.get("slot", "N/A") if isinstance(eq, dict) else "N/A"

    return f"[Item] {name} | Slot: {slot} | Members: {members} | HighAlch: {highalch} gp | Reqs: {req_str}\n" f"Examine: {examine}"


def build_prayer_text(record: dict) -> str:
    """Builds a compact text representation for prayers."""
    name = record.get("name", "Unknown")
    desc = record.get("description", "")
    drain = record.get("drain_per_minute", 0)
    reqs = record.get("requirements", {})
    req_str = ", ".join([f"{k}: {v}" for k, v in reqs.items()]) if reqs else "None"

    return f"[Prayer] {name} | Drain Rate: {drain}/min | Reqs: {req_str}\n" f"Description: {desc}"


def process_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}", flush=True)

    model_name = "fine_tuned_osrs_embedder_v2"
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

                        # 2. Preserve raw structured fields in metadata for Python reranking
                        meta_entry = {
                            "source": file_name,
                            "title": title,
                            "category": category,
                            "text": text,
                        }

                        # Crucial: attach raw drops array so get_monster_drop_rate_for_query() can read it
                        if "drops" in record:
                            meta_entry["drops"] = record["drops"]
                        if "slayer_level" in record:
                            meta_entry["slayer_level"] = record["slayer_level"]

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
