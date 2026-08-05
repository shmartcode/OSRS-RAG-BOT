import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import glob


class LocalRAGRetriever:
    def __init__(self, processed_dir="data/processed", model_name="all-mpnet-base-v2"):
        self.processed_dir = processed_dir
        self.index_path = os.path.join(processed_dir, "vector_index.faiss")

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Run your index builder first!")
        self.index = faiss.read_index(self.index_path)

        print("Loading metadata records...")
        self.metadata_records = self._load_metadata()
        print(f"Retriever initialized successfully with {self.index.ntotal} vectors and {len(self.metadata_records)} metadata entries.")

        # processed_dir = "data/processed"
        # print("All files in directory:", os.listdir(processed_dir))

    def _load_metadata(self):
        """Loads and flattens all metadata json files matching the embeddings corpus."""
        records = []
        meta_files = sorted([f for f in os.listdir(self.processed_dir) if f.endswith("_metadata.json")])

        for file_name in meta_files:
            file_path = os.path.join(self.processed_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
        return records

    def search(self, query_text, top_k=5):
        """Encodes query, searches FAISS index, and maps back to metadata chunks."""

        intent_info = route_query_intent(query_text)
        target7_category = intent_info.get("category_filter")

        # Encode the query into a vector and cast to float32 (required by FAISS)
        query_vector = self.model.encode([query_text]).astype(np.float32)

        # Normalize the query vector for cosine similarity (IndexFlatIP)
        faiss.normalize_L2(query_vector)

        # Pull a larger candidate pool from FAISS to allow room for filtering/re-ranking
        fetch_k = max(top_k * 3, 30)
        distances, indices = self.index.search(query_vector, k=fetch_k)

        # Gather initial matches from FAISS index
        candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # Skip FAISS padding

            meta = self.metadata_records[idx] if idx < len(self.metadata_records) else None
            if meta:
                candidates.append({"score": float(score), "metadata": meta})

        # Apply intent prioritization/filtering
        if target_category:
            category_matches = [c for c in candidates if c["metadata"].get("category") == target_category]
            other_matches = [c for c in candidates if c["metadata"].get("category") != target_category]
            results = category_matches + other_matches
        else:
            results = candidates

        # Return constrained strictly to top_k
        return results[:top_k]


def route_query_intent(query: str) -> dict:
    """Analyzes the user query to detect intent and return filtering rules."""
    query_lower = query.lower()

    # Define trigger keywords for monster drops
    drop_triggers = ["drop", "drops", "drop table", "who drops", "what drops", "dropping", "drop chance"]

    # Check if the query is asking about drops
    is_drop_query = any(trigger in query_lower for trigger in drop_triggers)

    # if is_drop_query:
    #     return {
    #         "intent": "monster_drops",
    #         # Enforce a filter for the metadata category
    #         "category_filter": "Monster",
    #     }

    # Default intent for general semantic queries
    return {"intent": "general", "category_filter": None}


if __name__ == "__main__":
    # Quick test execution
    retriever = LocalRAGRetriever()
    test_query = "What monsters drop dragon bones?"
    print(f"\nRunning test search for: '{test_query}'\n")

    hits = retriever.search(test_query, top_k=10)
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Score: {hit['score']:.4f}")
        print(f"Content / Data: {json.dumps(hit['metadata'], indent=2)[:300]}...\n")
