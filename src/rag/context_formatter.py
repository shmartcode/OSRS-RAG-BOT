def format_hits_for_llm(hits: list[dict], max_text_len: int = 1200) -> str:
    """Formats retrieved query hits into a clean, structured context block for LLM prompts."""
    if not hits:
        return "No relevant OSRS wiki entries found."

    formatted_blocks = []

    for idx, hit in enumerate(hits, start=1):
        meta = hit.get("metadata", {})
        title = meta.get("title") or meta.get("name", "Untitled Entry")

        category = meta.get("category", "General")
        text = meta.get("text", "").strip()

        # Apply truncation only to secondary sources if they exceed max_text_len
        if idx > 1 and len(text) > max_text_len:
            text = text[:max_text_len] + "...\n[Content Truncated]"

        # Check for aggregated drop math summary if present
        drop_summary = meta.get("calculated_drop_summary")

        block = [f"--- SOURCE [{idx}]: {title} (Category: {category}) ---"]

        if drop_summary:
            block.append(f"AGGREGATED DROP DATA:\n{drop_summary}")

        block.append(f"CONTENT:\n{text}")
        formatted_blocks.append("\n".join(block))

    return "\n\n".join(formatted_blocks)


def build_rag_prompt(user_query: str, hits: list[dict]) -> str:
    """Combines system instructions, structured context, and user query into final LLM prompt."""
    context_block = format_hits_for_llm(hits)

    prompt = f"""You are an expert Old School RuneScape (OSRS) assistant. Answer the user's query accurately using ONLY the provided context entries below across our categories (Wiki, Monsters, Prayers, Items). If the context does not contain enough information, state clearly what is missing.

=== RETRIEVED OSRS CONTEXT ===
{context_block}
==============================

USER QUERY: {user_query}

INSTRUCTIONS:
1. Ground your answer in the provided context.
2. EXECUTIVE SUMMARY FIRST: Provide a clean, scannable summary answering the core question first (e.g., stats, primary BiS gear, core requirements, or exact drop rates). Keep initial summaries concise.
3. PROGRESSIVE DISCLOSURE: Do not dump exhaustive inventory layouts, phase-by-phase mechanics, or low-tier budget substitutes all at once unless explicitly requested.
4. FOLLOW-UP ELICITATION: Always end your response by offering 2-3 specific follow-up topics the user can explore next (e.g., Detailed Inventory Setup, Specific Phase Mechanics, Budget Gear Substitutes).

ANSWER:"""

    return prompt
