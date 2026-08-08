# eval_retriever.py
from retriever import LocalRAGRetriever

retriever = LocalRAGRetriever()

# Test queries paired with expected top hits and metadata requirements
TEST_CASES = [
    {
        "query": "what is the drop rate of draconic visage from iron dragons",
        "expected_top_title": "Iron dragon",
        "expected_category": "Monster",
        "must_contain_metadata": ["calculated_drop_summary"],
        "expected_drop_substring": "1 in 10,000",
    },
    {
        "query": "dragon boots stats",
        "expected_top_title": "Dragon boots",
        "expected_category": "Item",
    },
    {
        "query": "abyssal whip high alch value",
        "expected_top_title": "Abyssal whip",
        "expected_category": "Item",
    },
    {
        "query": "what drops dragon boots",
        "expected_top_title": "Spiritual mage",
        "expected_category": "Monster",
    },
]


def run_evals():
    passed = 0
    for case in TEST_CASES:
        query = case["query"]
        results = retriever.search(query, top_k=5)

        if not results:
            print(f"❌ FAIL: '{query}' returned no results.")
            continue

        top_hit = results[0]
        title = top_hit.get("metadata", {}).get("title")
        category = top_hit.get("metadata", {}).get("category")

        # Check title/entity ranking
        if title != case["expected_top_title"]:
            print(f"❌ RANK FAIL: '{query}' -> Expected #{case['expected_top_title']}, got #{title}")
            continue

        # Check category match
        if category != case["expected_category"]:
            print(f"❌ CATEGORY FAIL: '{query}' -> Expected category {case['expected_category']}, got {category}")
            continue

        # Check exact drop math enrichment
        if "must_contain_metadata" in case:
            meta = top_hit.get("metadata", {})
            missing = [key for key in case["must_contain_metadata"] if key not in meta]
            if missing:
                print(f"❌ METADATA FAIL: '{query}' missing metadata keys: {missing}")
                continue

            summary = meta.get("calculated_drop_summary", "")
            if case["expected_drop_substring"] not in summary:
                print(f"❌ MATH FAIL: '{query}' expected '{case['expected_drop_substring']}' in summary, got '{summary}'")
                continue

        print(f"✅ PASS: '{query}' -> Ranked #{title} correctly.")
        passed += 1

    print(f"\nCompleted: {passed}/{len(TEST_CASES)} tests passed.")


if __name__ == "__main__":
    run_evals()
