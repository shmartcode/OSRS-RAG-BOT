import json
import requests
from pathlib import Path
import time

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"


def fetch_ge_market_data(output_filepath=OUTPUT_DIR / "ge_prices_corpus.jsonl"):
    """Fetches item mapping and latest live prices from the OSRS Wiki API, combines them, and exports a unified GE dataset."""

    headers = {"User-Agent": ("OSRS-RAG-Learning-Project (contact martin.816@wright.edu)")}

    print("Fetching Grand Exchange mapping data...")
    mapping_url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
    mapping_response = requests.get(mapping_url, headers=headers)
    if mapping_response.status_code != 200:
        print("Failed to fetch item mapping.")
        return

    mapping_data = mapping_response.json()
    # Convert list to a dictionary keyed by item ID for fast lookups
    items_map = {item["id"]: item for item in mapping_data}

    print("Fetching latest GE prices...")
    latest_url = "https://prices.runescape.wiki/api/v1/osrs/latest"
    latest_response = requests.get(latest_url, headers=headers)
    if latest_response.status_code != 200:
        print("Failed to fetch latest prices.")
        return

    prices_data = latest_response.json().get("data", {})

    count = 0
    with open(output_filepath, "w", encoding="utf-8") as f:
        for item_id_str, price_info in prices_data.items():
            item_id = int(item_id_str)
            meta = items_map.get(item_id, {})

            name = meta.get("name", f"Unknown Item {item_id}")
            high_price = price_info.get("high")  # Instant sell price
            low_price = price_info.get("low")  # Instant buy price
            high_time = price_info.get("highTime")
            low_time = price_info.get("lowTime")

            # Calculate margin if both prices exist
            margin = (high_price - low_price) if (high_price and low_price) else 0

            ge_record = {
                "id": item_id,
                "name": name,
                "category": "GrandExchange",
                "members": meta.get("members", False),
                "buy_limit": meta.get("limit", None),
                "high_price": high_price,
                "low_price": low_price,
                "margin": margin,
                "high_time": high_time,
                "low_time": low_time,
                "examine": meta.get("examine", None),
                # Structured text summary for your vector embeddings
                "text": (
                    f"Grand Exchange Item: {name}. Members: {meta.get('members', False)}. "
                    f"Buy Limit: {meta.get('limit', 'N/A')}. Latest Sell Price (High):"
                    f" {high_price} gp. Latest Buy Price (Low): {low_price} gp."
                    f" Current Margin: {margin} gp."
                ),
            }

            f.write(json.dumps(ge_record) + "\n")
            count += 1

    print(f"Successfully processed and saved {count} GE market records to" f" {output_filepath}")


if __name__ == "__main__":
    fetch_ge_market_data()
