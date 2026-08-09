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

    prompt = f"""You are an expert Old School RuneScape (OSRS) assistant. Answer the user's query accurately using ONLY the provided wiki context entries below. If the context does not contain enough information, state clearly what is missing.

=== RETRIEVED WIKI CONTEXT ===
{context_block}
==============================

USER QUERY: {user_query}

INSTRUCTIONS:
1. Ground your answer in the provided context.
2. For combat/boss strategy queries, outline inventory setups, gear choices, and core kill mechanics.
3. For drop rate queries, state exact drop rates or odds clearly.
4. Keep the response concise, clear, and formatted with Markdown.

ANSWER:"""

    return prompt
