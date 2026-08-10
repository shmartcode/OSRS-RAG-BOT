import json
import os
import re
import time
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from src.rag.aliases import resolve_query_aliases
from src.rag.context_formatter import build_rag_prompt
from src.rag.drop_rate_formatter import enrich_retrieved_hit_with_drop_math

# =====================================================================
# GLOBAL STATELESS HELPERS
# =====================================================================


def route_query_intent(query: str) -> dict:
    """Analyzes user query to detect intent and return filtering/ranking rules."""
    query_lower = query.lower()

    BUY_KEYWORDS = [
        "where to buy",
        "where can i buy",
        "buy",
        "buy the",
        "ge price",
        "grand exchange",
        "cost",
        "how much is",
        "price",
        "gp",
        "worth",
    ]

    SHOP_KEYWORDS = [
        "shop",
        "store",
        "vendor",
        "buy from shop",
        "buy from shops",
        "what shops",
    ]

    MONSTER_KEYWORDS = [
        "hp",
        "hitpoints",
        "combat level",
        "slayer",
        "slayer level",
        "xp",
        "defence level",
        "weakness",
        "immune",
        "aggressive",
        "poisonous",
        "boss",
        "combat stats",
    ]

    MONSTER_THAT_DROP_KEYWORDS = [
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

    ITEM_KEYWORDS = [
        "bonus",
        "bonuses",
        "attack speed",
        "ticks",
        "slash",
        "stab",
        "crush",
        "ranged strength",
        "melee strength",
        "magic damage",
        "prayer bonus",
        "weight",
        "equip",
        "slot",
        "combat stats",
        "style",
        "attack bonuses",
        "defense bonuses",
        "other bonuses",
    ]

    SPELL_COST_KEYWORDS = [
        "rune cost",
        "runes to cast",
        "how many runes",
        "what runes",
        "which runes",
    ]

    AMBIGUOUS_STATS = ["stats"]

    is_strategy_query = any(keyword in query_lower for keyword in STRATEGY_KEYWORDS)
    is_monsters_that_drop = any(keyword in query_lower for keyword in MONSTER_THAT_DROP_KEYWORDS)
    is_buy_query = any(keyword in query_lower for keyword in BUY_KEYWORDS)
    is_shop_query = any(keyword in query_lower for keyword in SHOP_KEYWORDS)
    is_monster_query = any(keyword in query_lower for keyword in MONSTER_KEYWORDS)
    is_item_query = any(keyword in query_lower for keyword in ITEM_KEYWORDS)
    is_ambiguous_stats = any(keyword in query_lower for keyword in AMBIGUOUS_STATS)
    is_spell_cost_query = any(keyword in query_lower for keyword in SPELL_COST_KEYWORDS)

    if is_strategy_query:
        target_entity = query.lower()
        for kw in STRATEGY_KEYWORDS:
            target_entity = target_entity.replace(kw, "")
        target_entity = target_entity.replace("how", "").replace("do i", "").strip()

        return {
            "intent": "strategy",
            "is_strategy_query": True,
            "prefer_category": "Strategy",
            "target_entity": target_entity,
        }

    if is_monsters_that_drop:
        return {
            "intent": "monster_drops",
            "is_strategy_query": False,
            "prefer_category": "Monster",
        }

    if is_buy_query:
        return {
            "intent": "purchasing",
            "is_strategy_query": False,
            "prefer_category": "GE_Price",
        }

    if is_shop_query:
        return {
            "intent": "purchasing",
            "is_strategy_query": False,
            "prefer_category": None,
        }

    if is_monster_query:
        return {
            "intent": "monster",
            "is_strategy_query": False,
            "prefer_category": "Monster",
        }

    if is_item_query:
        return {
            "intent": "item_bonuses",
            "is_strategy_query": False,
            "prefer_category": "Item",
        }

    if is_ambiguous_stats:
        return {"intent": "ambiguous_stats", "is_strategy_query": False, "prefer_category": "Monster or Item"}

    if is_item_query:
        return {
            "intent": "spell runes cost",
            "is_strategy_query": False,
            "prefer_category": None,
        }

    return {
        "intent": "general",
        "is_strategy_query": False,
        "prefer_category": None,
    }


def compute_title_boost(query_text: str, candidate_title: str, category: str) -> float:
    """Computes title score boost/penalty to prioritize exact matches and base items."""
    q_lower = query_text.lower().strip()
    t_lower = candidate_title.lower().strip()

    boost = 0.0

    achievement_terms = [
        "combat achievement",
        "combat diary",
        "grandmaster",
        "master",
        "elite",
        "hard task",
        "ca/",
    ]
    has_achievement_title = any(term in t_lower for term in achievement_terms)
    query_asks_achievement = any(term in q_lower for term in ["achievement", "task", "diary", "grandmaster", "ca"])

    if has_achievement_title and not query_asks_achievement:
        return -10.0

    if "/strategies" in t_lower or "/guide" in t_lower or category == "Strategy":
        boost += 3.0
        return boost

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
        model_name="shmartcode/osrs-embedder-v2",
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
        """Extracts target item keyword from natural language queries."""
        clean_q = query.lower().strip()

        prefix_pattern = r"^(?:what|which|who|where|how|can you|tell me|list)\s+(?:monsters?|mobs?|bosses?|enemies?|npcs?)?\s*(?:that|can|do|will|would)?\s*(?:drop|drops|dropped|give|gives|yield|yields|farm|get|obtain|find|kill for)\s+(?:the|a|an)?\s*"
        clean_q = re.sub(prefix_pattern, "", clean_q)

        suffix_pattern = r"\s+(?:dropped by|drops from|obtained from|farmed from|from|sources?)$"
        clean_q = re.sub(suffix_pattern, "", clean_q)

        filler_patterns = [
            r"\bdrop\s+rates?\s+(?:of|for)?\b",
            r"\bdrop\s+chances?\s+(?:of|for)?\b",
            r"\bwhere\s+to\s+(?:get|find|farm|obtain)\b",
            r"\bhow\s+to\s+(?:get|find|farm|obtain)\b",
            r"\bwhat\s+(?:drops|gives)\b",
            r"\bwho\s+(?:drops|gives)\b",
        ]
        for pattern in filler_patterns:
            clean_q = re.sub(pattern, " ", clean_q)

        clean_q = re.sub(r"^(?:the|a|an)\s+", "", clean_q)
        clean_q = clean_q.strip("? .!,").strip()

        preserved_plurals = (
            "ss",
            "boots",
            "gloves",
            "vambs",
            "chaps",
            "logs",
            "bones",
            "barrows",
            "coins",
        )
        if clean_q.endswith("s") and not clean_q.endswith(preserved_plurals):
            clean_q = clean_q[:-1]

        return clean_q.strip()

    def _try_aggregated_drop_search(self, query_text: str, top_k_monsters: int = 10) -> list[dict]:
        """Scans metadata directly to aggregate all monsters dropping the requested item."""
        target_item = self._extract_target_item_name(query_text)
        if not target_item or len(target_item) < 3:
            return []

        matching_monsters = []
        seen_monsters = set()

        for record in self.metadata_records:
            if record.get("category") != "Monster":
                continue

            raw_monster_name = record.get("title") or record.get("name", "Unknown")
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
                    break

        if not matching_monsters:
            return []

        matching_monsters.sort(key=lambda x: x["rarity"], reverse=True)
        top_matches = matching_monsters[:top_k_monsters]

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

    def _apply_intent_boosting(self, candidates: list[dict], intent_info: dict) -> list[dict]:
        """Applies soft score adjustments based on intent."""
        is_strategy = intent_info.get("is_strategy_query", False)
        prefer_cat = intent_info.get("prefer_category")
        target_entity = (intent_info.get("target_entity") or "").strip().lower()

        boosted_candidates = []
        for orig in candidates:
            c = dict(orig)  # Shallow copy to prevent mutating raw reference
            title = (c["metadata"].get("title") or c["metadata"].get("name", "")).lower().strip()
            category = c["metadata"].get("category", "")

            if is_strategy:
                if "/strategies" in title or "strategy" in title or "guide" in title or category == "Strategy":
                    c["score"] += 8.0
                elif category == "Monster" or (target_entity and (target_entity in title or title in target_entity)):
                    c["score"] += 4.0
                elif category == "Drop_Table" or "achievement" in title or "combat achievement" in title:
                    c["score"] -= 10.0

            elif prefer_cat and category == prefer_cat:
                c["score"] += 1.5

            elif prefer_cat and category in prefer_cat:
                c["score"] += 1.5

            boosted_candidates.append(c)

        boosted_candidates.sort(key=lambda x: x["score"], reverse=True)
        return boosted_candidates

    def _vector_search_and_rerank(self, query_text: str, intent_info: dict, top_k: int = 5) -> list[dict]:
        """Runs FAISS search, Cross-Encoder rerank, and title boost."""
        t0 = time.perf_counter()

        query_vector = self.model.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(query_vector)

        fetch_k = max(top_k * 10, 50)
        distances, indices = self.index.search(query_vector, k=fetch_k)
        t_faiss = time.perf_counter() - t0

        raw_candidates = []
        for score, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_records):
                raw_candidates.append(
                    {
                        "score": float(score),
                        "faiss_score": float(score),
                        "metadata": self.metadata_records[idx],
                    }
                )

        if not raw_candidates:
            return []

        t1 = time.perf_counter()

        pre_boosted = self._apply_intent_boosting(raw_candidates, intent_info)
        candidates_to_rerank = pre_boosted[:15]

        pairs = []
        for c in candidates_to_rerank:
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            text = c["metadata"].get("text", "")
            pairs.append([query_text, f"{title}: {text}"])

        rerank_scores = self.reranker.predict(pairs)
        t_rerank = time.perf_counter() - t1

        for c, r_score in zip(candidates_to_rerank, rerank_scores):
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            cat = c["metadata"].get("category", "")
            c["score"] = float(r_score) + compute_title_boost(query_text, title, cat)

        candidates_to_rerank.sort(key=lambda x: x["score"], reverse=True)

        deduped = []
        seen_keys = set()
        for c in candidates_to_rerank:
            title = c["metadata"].get("title") or c["metadata"].get("name", "")
            category = c["metadata"].get("category", "")
            key = f"{title.lower()}_{category.lower()}"
            if key not in seen_keys:
                seen_keys.add(key)
                deduped.append(c)

        final_hits = deduped[:top_k]

        for hit in final_hits:
            if hit.get("metadata", {}).get("category") == "Monster":
                enrich_retrieved_hit_with_drop_math(hit, query_text)

        print(f"[LATENCY] FAISS: {t_faiss*1000:.1f}ms | Reranker ({len(candidates_to_rerank)} items): {t_rerank*1000:.1f}ms")

        return final_hits

    def search(self, query_text: str, top_k: int = 5) -> list[dict]:
        """Main search entrypoint."""
        canonical_query = resolve_query_aliases(query_text)
        intent_info = route_query_intent(canonical_query)

        if intent_info.get("intent") == "monster_drops":
            aggregated_hits = self._try_aggregated_drop_search(canonical_query)
            if aggregated_hits:
                return aggregated_hits

        return self._vector_search_and_rerank(canonical_query, intent_info, top_k=top_k)
