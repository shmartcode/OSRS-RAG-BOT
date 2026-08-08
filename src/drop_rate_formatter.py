import math
from typing import Dict, List, Optional


def format_drop_probability(rarity: float) -> str:
    """Converts a raw float probability (e.g. 0.0001) into human-readable OSRS drop rates."""
    if rarity <= 0:
        return "Guaranteed / Never (0%)"
    if rarity >= 1.0:
        return "100% (Guaranteed)"

    # Calculate "1 in X"
    one_in_x = round(1.0 / rarity)
    percentage = rarity * 100

    # Format percentage neatly without trailing zeros
    if percentage < 0.01:
        pct_str = f"{percentage:.4f}%".rstrip("0").rstrip(".")
    else:
        pct_str = f"{percentage:.2f}%".rstrip("0").rstrip(".")

    return f"1 in {one_in_x:,} ({pct_str} chance | ~{one_in_x:,} kills expected)"


def extract_exact_drop_info(drops: List[Dict], target_item_name: str) -> Optional[Dict]:
    """Finds a specific drop in the monster's metadata drops array and computes exact stats."""
    target_lower = target_item_name.lower().strip()

    for drop in drops:
        drop_name = drop.get("name", "").lower()

        # Check for exact or substring match (e.g., "visage" in "draconic visage")
        if target_lower in drop_name or drop_name in target_lower:
            rarity = float(drop.get("rarity", 0.0))
            quantity = str(drop.get("quantity", "1"))
            rolls = int(drop.get("rolls", 1))

            # Adjust effective rarity if the monster has multiple drop rolls per kill
            effective_rarity = 1.0 - ((1.0 - rarity) ** rolls) if rolls > 1 else rarity

            return {
                "item_name": drop.get("name"),
                "raw_rarity": rarity,
                "effective_rarity": effective_rarity,
                "formatted_rate": format_drop_probability(effective_rarity),
                "quantity": quantity,
                "rolls_per_kill": rolls,
            }

    return None


def enrich_retrieved_hit_with_drop_math(hit: Dict, query: str) -> Dict:
    """Modifies a retrieved hit dict to include explicit drop rate math if applicable."""
    meta = hit.get("metadata", {})
    drops = meta.get("drops", [])

    if not drops:
        return hit

    # Simple heuristic to extract item intent from queries like "draconic visage drop rate"
    query_words = [w for w in query.lower().split() if w not in ["what", "is", "the", "drop", "rate", "of", "from", "chance", "for", "how", "rare"]]
    target_keyword = " ".join(query_words).strip()

    if target_keyword:
        drop_info = extract_exact_drop_info(drops, target_keyword)
        if drop_info:
            # Store calculated math inside metadata for the LLM prompt generator
            hit["metadata"]["calculated_drop_summary"] = (
                f"EXACT DROP MATH -> Item: {drop_info['item_name']} | " f"Rate: {drop_info['formatted_rate']} | " f"Quantity: {drop_info['quantity']}"
            )

    return hit
