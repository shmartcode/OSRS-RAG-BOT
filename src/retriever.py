import glob
import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# =====================================================================
# GLOBAL STATELESS HELPERS (Pure functions, easy to unit test)
# =====================================================================


def route_query_intent(query: str) -> dict:
    """Analyzes user query to detect intent and return filtering/ranking rules."""
    query_lower = query.lower()

    buy_triggers = [
        "where to buy",
        "where can i buy",
        "buy",
        "buy the",
        "shop",
        "store",
        "vendor",
        "ge price",
        "grand exchange",
        "cost",
        "how much is",
        "price",
        "gp",
        "worth",
    ]

    monster_triggers = [
        "drop",
        "drops",
        "drops the",
        "drop table",
        "who drops",
        "what drops",
        "dropping",
        "drop chance",
        "kill",
        "slayer",
        "boss",
        "monster",
    ]

    is_buy_query = any(trigger in query_lower for trigger in buy_triggers)
    is_monster_query = any(trigger in query_lower for trigger in monster_triggers)

    if is_buy_query:
        return {
            "intent": "purchasing",
            "exclude_category": None,
            "prefer_category": None,
        }

    if is_monster_query:
        return {
            "intent": "monster_drops",
            "exclude_category": "GE_Price",
            "prefer_category": "Monster",
        }

    return {
        "intent": "general",
        "exclude_category": "GE_Price",
        "prefer_category": None,
    }


def compute_title_boost(query_text: str, candidate_title: str, category: str) -> float:
    """Computes title score boost/penalty to prioritize base items over cosmetic variants

    and give Item metadata chunks priority over Wiki pages for basic item queries (Mod 3).
    """
    q_lower = query_text.lower().strip()
    t_lower = candidate_title.lower().strip()

    boost = 0.0

    # Base match boost
    if t_lower in q_lower or q_lower in t_lower:
        boost += 2.0

    # Slight structural advantage for raw Item corpus entries on item queries
    if category == "Item" and t_lower in q_lower:
        boost += 1.0

    # Demote variants containing parentheses e.g. (cr), (p++), (or) unless requested
    has_variant_in_title = bool(re.search(r"\(.*?\)", t_lower))
    has_variant_in_query = "(" in q_lower or ")" in q_lower

    if has_variant_in_title and not has_variant_in_query:
        boost -= 4.0

    return boost


def get_monster_drop_rate_for_query(query_text: str, metadata: dict) -> float:
    """Extracts drop rate probability for the item requested in the query."""
    if metadata.get("category") != "Monster":
        return 0.0

    drops = metadata.get("drops", [])
    text_content = str(metadata.get("text", "")).lower()
    q_lower = query_text.lower()

    stopwords = {
        "what",
        "monster",
        "monsters",
        "drops",
        "drop",
        "the",
        "a",
        "an",
        "does",
        "who",
        "is",
    }
    target_words = [w for w in re.findall(r"\b\w+\b", q_lower) if w not in stopwords and len(w) > 2]

    best_prob = 0.0

    if isinstance(drops, list):
        for drop in drops:
            item_name = ""
            rarity = None

            if isinstance(drop, dict):
                item_name = str(drop.get("item", "") or drop.get("name", "")).lower()
                rarity = drop.get("rarity", 0)
            elif isinstance(drop, str):
                item_name = drop.lower()

            if any(w in item_name for w in target_words if len(w) > 3):
                prob = parse_rarity_value(rarity)
                if prob > best_prob:
                    best_prob = prob

    if best_prob == 0.0:
        for w in target_words:
            if w in text_content and len(w) > 3:
                best_prob = 0.001

    return best_prob


def parse_rarity_value(rarity) -> float:
    """Converts numeric, fraction, or word-based rarities into probability float values."""
    if isinstance(rarity, (int, float)):
        return float(rarity)
    if isinstance(rarity, str):
        if "/" in rarity:
            try:
                num, den = rarity.split("/")
                return float(num) / float(den)
            except ValueError:
                return 0.0
        tier_map = {
            "always": 1.0,
            "common": 0.1,
            "uncommon": 0.01,
            "rare": 0.001,
            "very rare": 0.0001,
        }
        return tier_map.get(rarity.lower(), 0.0)
    return 0.0


# =====================================================================
# STATEFUL RETRIEVER CLASS
# =====================================================================


class LocalRAGRetriever:

    def __init__(
        self,
        processed_dir="data/processed",
        model_name="fine_tuned_osrs_embedder_epoch2",
        reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.processed_dir = processed_dir
        self.index_path = os.path.join(processed_dir, "vector_index.faiss")

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Loading Cross-Encoder reranker...")
        self.reranker = CrossEncoder(reranker_name)

        print("Loading FAISS index...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}.")
        self.index = faiss.read_index(self.index_path)

        print("Loading metadata records...")
        self.metadata_records = self._load_metadata()
        print(f"Retriever initialized with {len(self.metadata_records)} metadata entries.")

    def _load_metadata(self):
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
        intent_info = route_query_intent(query_text)
        exclude_cat = intent_info.get("exclude_category")
        prefer_cat = intent_info.get("prefer_category")
        intent_type = intent_info.get("intent")

        # Step 1: Broad FAISS Retrieval
        query_vector = self.model.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(query_vector)

        # Mod 1: Larger candidate pool to ensure deep Monster entries are captured
        fetch_k = max(top_k * 20, 100)
        distances, indices = self.index.search(query_vector, k=fetch_k)

        candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_records):
                candidates.append(
                    {
                        "faiss_score": float(score),
                        "metadata": self.metadata_records[idx],
                    }
                )

        # Step 2: Hard Category Exclusions
        if exclude_cat:
            candidates = [c for c in candidates if c["metadata"].get("category") != exclude_cat]

        # STRICT FILTER FOR MONSTER DROPS: Discard non-Monster chunks completely
        if intent_type == "monster_drops":
            candidates = [c for c in candidates if c["metadata"].get("category") == "Monster"]

        if not candidates:
            return []

        # Step 3: Cross-Encoder Reranking
        pairs = []
        for c in candidates:
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            text = c["metadata"].get("text", "")
            pairs.append([query_text, f"{title}: {text}"])

        rerank_scores = self.reranker.predict(pairs)

        # Step 4: Apply Title Boost & Item Priority (Mod 3)
        for c, r_score in zip(candidates, rerank_scores):
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            cat = c["metadata"].get("category", "")
            c["score"] = float(r_score) + compute_title_boost(query_text, title, cat)

        # Step 5: Category-Aware Deduplication
        deduped = []
        seen_keys = set()

        candidates.sort(key=lambda x: x["score"], reverse=True)

        for c in candidates:
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            category = c["metadata"].get("category", "")
            key = f"{title.lower()}_{category.lower()}"
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(c)

        # Step 6: Intent-Based Ranking
        if intent_type == "monster_drops":
            # Pure Monster results—rank strictly by drop rate probability and cross-encoder score
            for m in deduped:
                prob = get_monster_drop_rate_for_query(query_text, m["metadata"])
                m["drop_prob"] = prob
                m["score"] = m["score"] + (prob * 15.0)

            results = sorted(deduped, key=lambda x: x["score"], reverse=True)

        elif prefer_cat:
            preferred = [c for c in deduped if c["metadata"].get("category") == prefer_cat]
            others = [c for c in deduped if c["metadata"].get("category") != prefer_cat]
            results = preferred + others

        else:
            results = deduped

        return results[:top_k]


if __name__ == "__main__":
    retriever = LocalRAGRetriever()
    test_queries = [
        "What stats are needed to wield an abyssal whip?",
        "How much damage does a dragon dagger special attack do?",
        "What monster drops the dragon boots?",
    ]

    for q in test_queries:
        print(f"\nRunning test search for: '{q}'\n")
        intent_info = route_query_intent(q)
        print("DEBUG INTENT:", intent_info)
        hits = retriever.search(q, top_k=5)
        for i, hit in enumerate(hits, 1):
            print(f"[{i}] Score: {hit['score']:.4f}")
            print(f"Content / Data: {json.dumps(hit['metadata'], indent=2)[:300]}...\n")
