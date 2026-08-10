import time
from typing import Dict, List
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
# Adjust imports based on your project structure
from src.rag.retriever import LocalRAGRetriever
from src.rag.retriever import route_query_intent

# Define test suite covering critical OSRS RAG edge cases
TEST_SUITE: Dict[str, List[Dict[str, str]]] = {
    "1. Spell Costs & Templates": [
        {
            "query": "What runes are required for High Level Alchemy?",
            "expected_tokens": ["Nature", "Fire", "1", "5"],
            "expected_category": None,
        },
        {
            "query": "What are the rune costs for Ice Barrage?",
            "expected_tokens": ["Blood", "Death", "Water"],
            "expected_category": None,
        },
    ],
    "2. Item Stats & Properties": [
        {
            "query": "What are the stats of an Abyssal whip?",
            "expected_tokens": ["Slash", "82", "Melee strength"],
            "expected_category": "Item",
        },
        {
            "query": "How much slash attack bonus does a Dragon dagger have?",
            "expected_tokens": ["Slash", "dagger"],
            "expected_category": "Item",
        },
        {
            "query": "What is the attack speed and bonuses of the Toxic blowpipe?",
            "expected_tokens": ["Ranged", "speed"],
            "expected_category": "Item",
        },
    ],
    "3. High-Volume Drop Tables": [
        {
            "query": "What monsters drop rune scimitars?",
            "expected_tokens": ["Fire giant", "drop"],
            "expected_category": "Monster",
        },
        {
            "query": "What monsters drop a Dragon spear?",
            "expected_tokens": ["drop"],
            "expected_category": "Monster",
        },
    ],
    "4. Hard Numbers & Requirements": [
        {
            "query": "What level Smithing do I need to make a Rune Platebody?",
            "expected_tokens": ["99", "Smithing"],
            "expected_category": None,
        },
        {
            "query": "How many Gold nuggets do I need for the Prospector outfit?",
            "expected_tokens": ["180", "Prospector"],
            "expected_category": None,
        },
        {
            "query": "What level Thieving is required for Master Farmers?",
            "expected_tokens": ["38", "Thieving"],
            "expected_category": None,
        },
    ],
    "5. Non-Combat Acquisition & Shops": [
        {
            "query": "Where can I buy Nature runes from shops?",
            "expected_tokens": ["Mage Arena", "shop"],
            "expected_category": None,
        },
        {
            "query": "Where can I get a Rune scimitar without killing monsters?",
            "expected_tokens": ["Smithing"],
            "expected_category": None,
        },
    ],
    "6. Hallucination Resistance & Mechanics": [
        {
            "query": "What monsters drop the Torva full helm?",
            "expected_tokens": ["Nex"],
            "expected_category": "Monster",
        },
        {
            "query": "What drops the Fire Cape?",
            "expected_tokens": ["TzTok-Jad", "Fight Caves"],
            "expected_category": "Monster",
        },
    ],
}


def run_pipeline_tests():
    print("==================================================")
    print("     OSRS RAG Pipeline Benchmark & Query Test     ")
    print("==================================================\n")

    retriever = LocalRAGRetriever()
    total_tests = 0
    passed_tests = 0

    start_time = time.time()

    for category, tests in TEST_SUITE.items():
        print(f"\n--- Category: {category} ---")

        for test in tests:
            total_tests += 1
            query = test["query"]
            expected_tokens = test["expected_tokens"]
            expected_category = test.get("expected_category")

            # 1. Test Router Intent (if applicable)
            actual_category = route_query_intent(query).get("prefer_category") if route_query_intent else "N/A"
            actual_intent = route_query_intent(query).get("intent") if route_query_intent else "N/A"
            if route_query_intent:
                if actual_category == "Monster or Item":
                    catgeory_match = expected_category in actual_category
                else:
                    catgeory_match = actual_category == expected_category

            # 2. Test Retrieval
            # Choose top_k dynamically if intent is drop_table
            top_k = 6 if actual_intent == "monster_drops" else 3
            results = retriever.search(query, top_k=top_k)

            # Combine retrieved chunk texts into one string for inspection
            retrieved_text = " ".join([str(chunk["metadata"].get("text", "")) for chunk in results])

            # Check if expected key tokens exist in retrieved context
            found_tokens = [token for token in expected_tokens if token.lower() in retrieved_text.lower()]
            token_match = len(found_tokens) == len(expected_tokens)

            # Determine pass/fail status
            test_passed = token_match and catgeory_match
            if test_passed:
                passed_tests += 1
                status = "✅ PASS"

            else:
                status = "❌ FAIL"

            # Print concise results per query
            print(f"\n[{status}] Query: '{query}'")
            print(f"  └─ Category: {actual_category} (Expected: {expected_category})")
            print(f"  └─ Tokens Found: {len(found_tokens)}/{len(expected_tokens)} -> {found_tokens}")

            if not test_passed:
                top_title = results[0]["metadata"].get("title", "Unknown") if results else "None"
                snippet = results[0]["metadata"].get("text", "")[:200] if results else "No context"
                print(f"  └─ Top Match Title: {top_title}")
                print(f"  └─ Snippet Preview: {snippet}...")

    elapsed = round(time.time() - start_time, 2)
    print("\n==================================================")
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed in {elapsed}s")
    print("==================================================")


if __name__ == "__main__":
    run_pipeline_tests()
