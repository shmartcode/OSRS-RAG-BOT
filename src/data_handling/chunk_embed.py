import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import torch


def process_embedding():
    # --- ADD THESE TWO LINES HERE ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}", flush=True)
    # --------------------------------

    # Pass the detected device into the model loader
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name, device=device)
    # print(f"Loading embedding model: {model_name}...")

    # List of your corpus files
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

                    # Dynamically build text from all properties in the record
                    properties = []
                    for key, value in record.items():
                        if value is not None and value != "":
                            if isinstance(value, (dict, list)):
                                value_str = json.dumps(value)
                            else:
                                value_str = str(value)
                            properties.append(f"{key}: {value_str}")

                    text = " | ".join(properties)

                    if text:
                        documents.append(text)
                        # Grab a descriptive name/title for metadata if available
                        title = record.get("title", record.get("name", "Unknown"))
                        # metadata.append({"source": file_name, "title": title})

                        if "monsters" in file_name:
                            category = "Monster"
                        elif "items" in file_name:
                            category = "Item"
                        elif "prayers" in file_name:
                            category = "Prayer"
                        elif "ge_prices" in file_name:
                            category = "GE_Price"
                        else:
                            category = "Wiki"

                        metadata.append(
                            {
                                "source": file_name,
                                "title": title,
                                "category": category,
                                "text": text,
                            }
                        )

                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSON line {line_num} in {file_name}: {e}")

        if documents:
            print(f"Generating embeddings for {len(documents)} records in {file_name}...")
            embeddings = model.encode(
                documents,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            print(f"Finished {file_name}. Embedding shape: {embeddings.shape}")

            # Save embeddings and metadata to your processed data folder
            base_name = file_name.replace(".jsonl", "")
            emb_path = f"{base_name}_embeddings.npy"
            meta_path = f"{base_name}_metadata.json"

            np.save(emb_path, embeddings)
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(metadata, mf, ensure_ascii=False, indent=2)

            print(f"Saved embeddings to {emb_path} and metadata to {meta_path}.")

        else:
            print(f"No valid documents found in {file_name}.")


if __name__ == "__main__":
    process_embedding()
