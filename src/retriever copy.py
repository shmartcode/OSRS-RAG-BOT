import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from drop_rate_formatter import enrich_retrieved_hit_with_drop_math
from aliases import resolve_query_aliases

# =====================================================================
# GLOBAL STATELESS HELPERS
# =====================================================================


def route_query_intent(query: str) -> dict:
    """Analyzes user query to detect intent and return filtering/ranking rules."""
    query_lower = query.lower()

    BUY_TRIGGERS = [
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

    MONSTER_TRIGGERS = ["stats", "slayer", "boss", "monster"]

    MONSTER_THAT_DROP_TRIGGERS = [
        "drop",
        "drops",
        "drops the",
        "drop table",
        "who drops",
        "what drops",
        "dropping",
        "drop chance",
        "monster that drop",
        "monsters that drop",
    ]
    STRATEGY_KEYWORDS = [
        "how to kill",
        "how to beat",
        "strategy",
        "strategies",
        "guide",
        "setup",
        "gear",
        "method",
        "solo",
        "duo",
        "5:0",
        "6:0",
        "7:0",
    ]

    is_strategy_query = any(trigger in query_lower for trigger in STRATEGY_KEYWORDS)
    is_monsters_that_drop = any(trigger in query_lower for trigger in MONSTER_THAT_DROP_TRIGGERS)
    is_buy_query = any(trigger in query_lower for trigger in BUY_TRIGGERS)
    is_monster_query = any(trigger in query_lower for trigger in MONSTER_TRIGGERS)

    if is_monsters_that_drop:
        return {
            "intent": "monster_drops",
            "exclude_category": "GE_Price",
            "prefer_category": "Monster",
        }

    if is_buy_query:
        return {
            "intent": "purchasing",
            "exclude_category": None,
            "prefer_category": None,
        }

    if is_monster_query:
        return {
            "intent": "monster",
            "exclude_category": "GE_Price",
            "prefer_category": "Monster",
        }

    if is_strategy_query:
        return {
            "intent": "stategy",
            "exclude_category": None,
            "prefer_category": None,
        }

    return {
        "intent": "general",
        "exclude_category": "GE_Price",
        "prefer_category": None,
    }


def compute_title_boost(query_text: str, candidate_title: str, category: str) -> float:
    """Computes title score boost/penalty to prioritize base items over variants."""
    q_lower = query_text.lower().strip()
    t_lower = candidate_title.lower().strip()

    boost = 0.0

    if t_lower in q_lower or q_lower in t_lower:
        boost += 2.0

    if category == "Item" and t_lower in q_lower:
        boost += 1.0

    has_variant_in_title = bool(re.search(r"\(.*?\)", t_lower))
    has_variant_in_query = "(" in q_lower or ")" in q_lower

    if has_variant_in_title and not has_variant_in_query:
        boost -= 4.0

    return boost


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
        model_name="fine_tuned_osrs_embedder_v2",
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

    def _extract_target_item_name(self, query: str) -> str:
        """search helper that extracts the target item keyword from the query text"""
        q_lower = query.lower()
        stopwords = [
            "what monsters drop",
            "monster that drops",
            "monsters that drop",
            "what drops the",
            "what drops",
            "who drops the",
            "who drops",
            "dropped by",
            "drop rate of",
            "drop chance of",
            "drop rate",
            "from",
            "the",
        ]
        clean_q = q_lower
        for word in stopwords:
            clean_q = clean_q.replace(word, " ")

        return clean_q.strip()

    def _try_aggregated_drop_search(self, query_text: str, top_k_monsters: int = 5) -> list[dict]:
        """search helper that compiles a list of monsters that drop a queried item"""
        """Scans metadata directly to aggregate all monsters dropping the requested item."""
        target_item = self._extract_target_item_name(query_text)
        if not target_item or len(target_item) < 3:
            return []

        matching_monsters = []
        seen_monsters = set()  # Track unique monster names

        for record in self.metadata_records:
            if record.get("category") != "Monster":
                continue

            raw_monster_name = record.get("title") or record.get("name", "Unknown")
            # Standardize name (e.g. removes parenthetical locations if needed, or normalizes case)
            base_monster_name = raw_monster_name.strip()

            if base_monster_name.lower() in seen_monsters:
                continue

            drops = record.get("drops", [])
            if not isinstance(drops, list):
                continue

            for drop in drops:
                item_name = ""
                rarity_val = 0.0

                if isinstance(drop, dict):
                    item_name = str(drop.get("item", "") or drop.get("name", "")).lower()
                    rarity_val = parse_rarity_value(drop.get("rarity", 0))
                elif isinstance(drop, str):
                    item_name = drop.lower()

                # Match target item in drop entry
                if target_item in item_name or item_name in target_item:
                    one_in_x = round(1.0 / rarity_val) if rarity_val > 0 else 0
                    pct = rarity_val * 100

                    matching_monsters.append(
                        {
                            "monster": base_monster_name,
                            "item_name": item_name.title(),
                            "rarity": rarity_val,
                            "one_in_x": one_in_x,
                            "percentage": pct,
                        }
                    )
                    seen_monsters.add(base_monster_name.lower())
                    break  # Found drop match for this monster entry

        if not matching_monsters:
            return []

        # Sort monsters by highest probability (descending)
        matching_monsters.sort(key=lambda x: x["rarity"], reverse=True)
        top_matches = matching_monsters[:top_k_monsters]

        # Build synthetic aggregated text block for LLM context
        summary_lines = [
            f"AGGREGATED DROP ANALYSIS FOR '{target_item.title()}' " f"(Found in {len(matching_monsters)} unique monster drop table(s)):",
            "Monsters ranked by highest drop rate:",
        ]

        for idx, m in enumerate(top_matches, start=1):
            rate_str = f"1 in {m['one_in_x']:,} ({m['percentage']:.3f}%)" if m["one_in_x"] > 0 else "Unknown rate"
            summary_lines.append(f"{idx}. {m['monster']} — Drop Rate: {rate_str}")

        text_payload = "\n".join(summary_lines)

        synthetic_hit = {
            "score": 10.0,
            "text": text_payload,
            "metadata": {
                "title": f"Drop Sources for {target_item.title()}",
                "category": "Aggregated_Drop_Summary",
                "calculated_drop_summary": text_payload,
                "text": text_payload,
            },
        }

        return [synthetic_hit]

    def _vector_search_and_rerank(self, query_text: str, intent_info: dict, top_k: int = 5) -> list[dict]:
        """search helper that runs FAISS search, category exlusions,
        cross-encoder rerank, title boost, deduplication, preferred category boosting, and item drop rate math"""
        exclude_cat = intent_info.get("exclude_category")
        prefer_cat = intent_info.get("prefer_category")

        # 1. FAISS Search
        query_vector = self.model.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(query_vector)

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

        # 2. Hard Category Exclusions - excludes categories based on query. ie dont use wiki source if we want item results
        if exclude_cat:
            candidates = [c for c in candidates if c["metadata"].get("category") != exclude_cat]

        if not candidates:
            return []

        # 3. Cross-Encoder Rerank - to rerank based on query and text pairs.
        pairs = []
        for c in candidates:
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            text = c["metadata"].get("text", "")
            pairs.append([query_text, f"{title}: {text}"])

        rerank_scores = self.reranker.predict(pairs)

        # 4. Title Boost - ensure the title of the source aligns with keywords in the query text
        for c, r_score in zip(candidates, rerank_scores):
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            cat = c["metadata"].get("category", "")
            c["score"] = float(r_score) + compute_title_boost(query_text, title, cat)

        # 5. Category-Aware Deduplication
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

        # 6. Category Preference Sorting
        if prefer_cat:
            preferred = [c for c in deduped if c["metadata"].get("category") == prefer_cat]
            others = [c for c in deduped if c["metadata"].get("category") != prefer_cat]
            results = preferred + others
        else:
            results = deduped

        final_hits = results[:top_k]

        # 7. Enrich Monster Hits with Math - adjusts meta data so drop rate numbers are sensible for the user
        for hit in final_hits:
            if hit.get("metadata", {}).get("category") == "Monster":
                enrich_retrieved_hit_with_drop_math(hit, query_text)

        return final_hits

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Main search function."""
        canonical_query = resolve_query_aliases(query_text)
        intent_info = route_query_intent(canonical_query)

        # 1. Fast Path: Aggregated metadata drop search
        if intent_info.get("intent") == "monster_drops":
            aggregated_hits = self._try_aggregated_drop_search(canonical_query)
            if aggregated_hits:
                return aggregated_hits

        # 2. Standard Path: Vector Search + Rerank
        return self._vector_search_and_rerank(canonical_query, intent_info, top_k=top_k)


# =====================================================================
# TESTING ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    retriever = LocalRAGRetriever()

    test_queries = [
        "what monsters drop dragon boots",
        "what drops draconic visage",
        "How much damage does a dragon dagger special attack do?",
        "What stats are needed to wield an abyssal whip?",
        "How do i kill bandos?",
    ]

    for q in test_queries:
        print(f"\n==========================================")
        print(f"SEARCH QUERY: '{q}'")
        print(f"==========================================")
        hits = retriever.search(q, top_k=5)

        for i, hit in enumerate(hits, 1):
            print(f"[{i}] Category: {hit['metadata'].get('category')}")
            print(f"Title: {hit['metadata'].get('title')}")
            print(f"Content Preview:\n{hit['metadata'].get('text', '')[:250]}\n")
