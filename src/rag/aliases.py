OSRS_ALIAS_MAP = {
    # Bosses & Monsters
    "bandos": "General Graardor",
    "graardor": "General Graardor",
    "armadyl": "Kree'arra",
    "kreearra": "Kree'arra",
    "kree": "Kree'arra",
    "zammy": "K'ril Tsutsaroth",
    "kril": "K'ril Tsutsaroth",
    "sara": "Commander Zilyana",
    "zilyana": "Commander Zilyana",
    "corp": "Corporeal Beast",
    "vork": "Vorkath",
    "money dragon": "Vorkath",
    "zul": "Zulrah",
    "sire": "Abyssal Sire",
    "cerb": "Cerberus",
    "thermo": "Thermonuclear smoke devil",
    "kq": "Kalphite Queen",
    "kbd": "King Black Dragon",
    "mole": "Giant Mole",
    "muspah": "Phantom Muspah",
    "duke": "Duke Sucellus",
    "levi": "The Leviathan",
    "whisperer": "The Whisperer",
    "vardorvis": "Vardorvis",
    # Raids & Minigames
    "cox": "Chambers of Xeric",
    "tob": "Theatre of Blood",
    "toa": "Tombs of Amascut",
    "cg": "Corrupted Gauntlet",
    "gauntlet": "The Gauntlet",
    # Common Items & Drops
    "visage": "Draconic visage",
    "dfs": "Dragonfire shield",
    "whip": "Abyssal whip",
    "bcp": "Bandos chestplate",
    "tassets": "Bandos tassets",
    "tassys": "Bandos tassets",
    "agn": "Armadyl godsword",
    "ags": "Armadyl godsword",
    "sgs": "Saradomin godsword",
    "bgs": "Bandos godsword",
    "zgs": "Zamorak godsword",
    "fang": "Osmumten's fang",
    "shadow": "Tumeken's shadow",
    "scythe": "Scythe of vitur",
    "tbow": "Twisted bow",
}


import re


def resolve_query_aliases(query: str, alias_map: dict = OSRS_ALIAS_MAP) -> str:
    """Replaces known OSRS nicknames/abbreviations in the query with their canonical names."""
    resolved_query = query.lower()

    # Sort aliases by length descending so longer phrases replace first (e.g., "money dragon" before "dragon")
    sorted_aliases = sorted(alias_map.keys(), key=len, reverse=True)

    for alias in sorted_aliases:
        canonical_name = alias_map[alias]
        # Use word boundaries so "sara" doesn't match inside "saradomin"
        pattern = r"\b" + re.escape(alias) + r"\b"
        resolved_query = re.sub(pattern, canonical_name.lower(), resolved_query)

    return resolved_query
