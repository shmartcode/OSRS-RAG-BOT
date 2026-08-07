import json
import os
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
MODEL_PATH = "fine_tuned_osrs_embedder_epoch2"
PROCESSED_DIR = "data/processed"
INDEX_PATH = os.path.join(PROCESSED_DIR, "vector_index.faiss")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_all_metadata():
    """
    Loads and merges all metadata JSON files in the order that matches
    how build_vector_index.py stacked the numpy embedding arrays.
    """
    embedding_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith("_embeddings.npy")])
    combined_metadata = []

    for emb_file in embedding_files:
        meta_file = emb_file.replace("_embeddings.npy", "_metadata.json")
        meta_path = os.path.join(PROCESSED_DIR, meta_file)

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_metadata.extend(data)
        else:
            print(f"Warning: Metadata file {meta_file} missing for {emb_file}")

    return combined_metadata


def query_index(query_text, model, index, metadata, top_k=3):
    """
    Encodes a user prompt, searches FAISS, and returns formatted top-K results.
    """
    # 1. Encode and normalize query vector (matches train/encode behavior)
    query_vector = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    # 2. Search FAISS index
    scores, indices = index.search(query_vector, top_k)

    print(f"\n=======================================================")
    print(f"Query: '{query_text}'")
    print(f"=======================================================")

    for i in range(top_k):
        idx = indices[0][i]
        score = scores[0][i]

        if idx < len(metadata):
            item = metadata[idx]
            category = item.get("category", "Unknown")
            title = item.get("title", "Unknown")
            text = item.get("text", "")

            # Truncate text preview for clean terminal output
            preview = text[:200] + "..." if len(text) > 200 else text

            print(f"\nRank {i+1} | Score: {score:.4f} | Category: [{category}]")
            print(f"Title: {title}")
            print(f"Content: {preview}")
        else:
            print(f"\nRank {i+1} | Index {idx} out of range for metadata bounds.")


def main():
    if not os.path.exists(INDEX_PATH):
        print(f"Error: Could not find FAISS index at {INDEX_PATH}")
        print("Please run build_vector_index.py first.")
        return

    print("Loading fine-tuned embedding model...")
    model = SentenceTransformer(MODEL_PATH, device=DEVICE)

    print("Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)

    print("Loading metadata records...")
    metadata = load_all_metadata()
    print(f"Loaded {index.ntotal} vectors and {len(metadata)} metadata entries.")

    # -------------------------------------------------------------------------
    # Test Queries
    # -------------------------------------------------------------------------
    test_queries = [
        "What stats are needed to wield an abyssal whip?",
        "How much damage does a dragon dagger special attack do?",
        "What monster drops the dragon boots?",
    ]

    for q in test_queries:
        query_index(q, model, index, metadata, top_k=6)


if __name__ == "__main__":
    main()
