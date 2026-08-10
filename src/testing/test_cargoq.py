# import json
# import requests

# HEADERS = {"User-Agent": "OSRS_RAG_Pipeline_Test/1.0 (contact@example.com)"}


# def test_wiki_parse(monster_name: str):
#     url = "https://oldschool.runescape.wiki/api.php"
#     params = {"action": "parse", "page": monster_name, "prop": "wikitext", "format": "json"}

#     print(f"\n--- Testing Wikitext Parse for: '{monster_name}' ---")
#     res = requests.get(url, params=params, headers=HEADERS).json()
#     wikitext = res.get("parse", {}).get("wikitext", {}).get("*", "")

#     if wikitext:
#         print(f"Successfully retrieved {len(wikitext)} characters of wikitext for {monster_name}.")
#         # Show first 300 chars (Infobox preview)
#         print("Preview:\n", wikitext)
#     else:
#         print("Failed to fetch wikitext.")


# test_wiki_parse("Adamant dragon")


import json
import re
import requests

HEADERS = {"User-Agent": "OSRS_RAG_Pipeline_Test/1.0 (contact@example.com)"}


def fetch_raw_wikitext(monster_name: str) -> str:
    """Fetches the complete raw wikitext for a given Wiki page."""
    url = "https://oldschool.runescape.wiki/api.php"
    params = {"action": "parse", "page": monster_name, "prop": "wikitext", "format": "json"}

    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code != 200:
        print(f"HTTP Error {response.status_code}")
        return ""

    data = response.json()
    if "error" in data:
        print(f"Wiki API Error: {data['error'].get('info')}")
        return ""

    return data.get("parse", {}).get("wikitext", {}).get("*", "")


# def parse_osrs_wikitext(monster_name: str, wikitext: str) -> dict:
#     """
#     Parses OSRS Wiki wikitext into structured monster attributes and drops.
#     """
#     parsed_data = {"name": monster_name, "attributes": {}, "drops": []}

#     # 1. Extract {{Infobox Monster ... }} block
#     # Matches {{Infobox Monster up to the balancing closing }}
#     infobox_match = re.search(r"\{\{Infobox Monster\s*\n(.*?)\n\}\}", wikitext, re.DOTALL | re.IGNORECASE)
#     if infobox_match:
#         infobox_body = infobox_match.group(1)
#         for line in infobox_body.split("\n"):
#             line = line.strip()
#             if line.startswith("|") and "=" in line:
#                 # Remove leading '|' and split on first '='
#                 key, val = line[1:].split("=", 1)
#                 # Clean Wiki links [[Item]] -> Item
#                 clean_val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", val.strip())
#                 parsed_data["attributes"][key.strip().lower()] = clean_val

#     # 2. Extract standard drops: {{DropsLine|name=...|quantity=...|rarity=...}}
#     # Matches {{DropsLine ... }} and {{DropsLineClue ... }}
#     drop_lines = re.findall(r"\{\{DropsLine(?:Clue)?\s*\|([^}]+)\}\}", wikitext, re.IGNORECASE)

#     for drop_str in drop_lines:
#         drop_dict = {}
#         # Split attributes by pipe '|'
#         parts = drop_str.split("|")
#         for part in parts:
#             if "=" in part:
#                 k, v = part.split("=", 1)
#                 # Strip Wiki markup link brackets [[ ]]
#                 clean_v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", v.strip())
#                 drop_dict[k.strip().lower()] = clean_v

#         # Handle {{DropsLineClue|type=elite|rarity=1/320}} template structure
#         if "type" in drop_dict and "name" not in drop_dict:
#             drop_dict["name"] = f"Clue scroll ({drop_dict['type']})"

#         if drop_dict.get("name"):
#             parsed_data["drops"].append(drop_dict)

#     return parsed_data

STANDARD_HERBS = [
    "Grimy guam leaf",
    "Grimy marrentill",
    "Grimy tarromin",
    "Grimy harralander",
    "Grimy ranarr weed",
    "Grimy irit leaf",
    "Grimy avantoe",
    "Grimy kwuarm",
    "Grimy cadantine",
    "Grimy lantadyme",
    "Grimy dwarf weed",
]


def parse_osrs_wikitext(monster_name: str, wikitext: str) -> dict:
    parsed_data = {"name": monster_name, "attributes": {}, "drops": []}

    # 1. Extract {{Infobox Monster}} parameters
    infobox_match = re.search(r"\{\{Infobox Monster\s*\n(.*?)\n\}\}", wikitext, re.DOTALL | re.IGNORECASE)
    if infobox_match:
        infobox_body = infobox_match.group(1)
        for line in infobox_body.split("\n"):
            line = line.strip()
            if line.startswith("|") and "=" in line:
                key, val = line[1:].split("=", 1)
                clean_val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", val.strip())
                parsed_data["attributes"][key.strip().lower()] = clean_val

    # 2. Extract standard drops: {{DropsLine}} & {{DropsLineClue}}
    drop_lines = re.findall(r"\{\{DropsLine(?:Clue)?\s*\|([^}]+)\}\}", wikitext, re.IGNORECASE)
    for drop_str in drop_lines:
        drop_dict = {}
        parts = drop_str.split("|")
        for part in parts:
            if "=" in part:
                k, v = part.split("=", 1)
                clean_v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", v.strip())
                drop_dict[k.strip().lower()] = clean_v

        if "type" in drop_dict and "name" not in drop_dict:
            drop_dict["name"] = f"Clue scroll ({drop_dict['type']})"

        if drop_dict.get("name"):
            parsed_data["drops"].append(drop_dict)

    # 3. Handle Herb Macro Tables: {{UsefulHerbDropLines|8/110|unnoted=yes}}
    herb_match = re.search(r"\{\{UsefulHerbDropLines\|([^|}]+)", wikitext, re.IGNORECASE)
    if herb_match:
        herb_rarity = herb_match.group(1).strip()
        for herb in STANDARD_HERBS:
            parsed_data["drops"].append(
                {"name": herb, "quantity": "1", "rarity": f"{herb_rarity} (Combined Herb Table)", "note": "Derived from UsefulHerbDropLines template"}
            )

    # 4. Handle Rare Drop Table: {{RareDropTable|1/110|...}}
    rdt_match = re.search(r"\{\{RareDropTable\|([^|}]+)", wikitext, re.IGNORECASE)
    if rdt_match:
        rdt_rarity = rdt_match.group(1).strip()
        parsed_data["drops"].append(
            {"name": "Rare drop table", "quantity": "1", "rarity": rdt_rarity, "note": "Rolls on the global OSRS Rare Drop Table"}
        )

    return parsed_data


# --- Run the Test ---
for name in ["Adamant dragon", "Rune dragon"]:
    print(f"\n================ Fetching: '{name}' ================")
    wikitext = fetch_raw_wikitext(name)

    if wikitext:
        print(f"Success! Fetched {len(wikitext)} characters of raw wikitext.")

        parsed = parse_osrs_wikitext(name, wikitext)
        print(f"Extracted Infobox Parameters: {len(parsed['attributes'])}")
        print(f"Extracted Drop Entries: {len(parsed['drops'])}")

        # Show a quick snippet of extracted data
        print("\nSample Infobox Attributes:")
        for k in ["combat", "hitpoints", "max hit", "attack style", "slayerreq"][:5]:
            if k in parsed["attributes"]:
                print(f"  - {k}: {parsed['attributes'][k]}")

        print("\nSample Drops (First 3):")
        for drop in parsed["drops"]:
            print(f"  - {drop.get('name', 'Unknown')}: {drop.get('quantity', '1')} (Rarity: {drop.get('rarity', 'N/A')})")
