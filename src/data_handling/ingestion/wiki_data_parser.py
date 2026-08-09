import gzip
import json
from pathlib import Path
import re
import mwparserfromhell

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "raw/pages_content_compressed"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clean_wiki_articles.jsonl"


def preprocess_osrs_templates(parsed_code):
    """Processes OSRS MediaWiki templates in the AST before stripping code.

    Preserves skill levels, item names in equipment slots, and slot rankings (1-5).
    """
    # 1. First Pass: Process leaf-level icon/item templates first so nested items expand
    # Handles: {{Item|Torva full helm}} -> "Torva full helm"
    # Handles: {{Skill|Attack|80+}} -> "Attack 80+"
    for template in parsed_code.filter_templates():
        name = template.name.strip().lower()

        if name in [
            "skill",
            "scp",
            "item",
            "ico",
            "icon",
            "s",
            "i",
            "use",
            "req",
            "requirement",
            "level",
            "levels",
        ]:
            extracted_parts = []
            for param in template.params:
                val = str(param.value).strip()
                key = str(param.name).strip()

                if not val:
                    continue

                if not key.isdigit():
                    extracted_parts.append(f"{key}: {val}")
                else:
                    extracted_parts.append(val)

            if extracted_parts:
                parsed_code.replace(template, f" {' '.join(extracted_parts)} ")

        elif name in ["cbl", "combat"]:
            if template.has(1):
                val = str(template.get(1).value).strip()
                parsed_code.replace(template, f" Combat level {val} ")

    # 2. Second Pass: Process Equipment Setups / Gear Tables with ranked slots
    for template in parsed_code.filter_templates():
        name = template.name.strip().lower()

        if any(k in name for k in ["equipment", "gear", "setup", "lineup", "loadout"]):
            gear_items = []
            for param in template.params:
                # Recursively parse the parameter value to convert internal templates or wiki links
                val_ast = mwparserfromhell.parse(str(param.value))

                # Expand any remaining sub-templates inside the slot
                for sub_t in val_ast.filter_templates():
                    sub_parts = [str(p.value).strip() for p in sub_t.params if str(p.value).strip()]
                    if sub_parts:
                        val_ast.replace(sub_t, " ".join(sub_parts))

                val = val_ast.strip_code().strip()
                key = str(param.name).strip().lower()

                if not val or key in ["caption", "image", "notes", "style"]:
                    continue

                # Parse ranked slot keys (e.g., head1 -> Head (BiS), head2 -> Head (Alt 1))
                # Match keys ending in numbers 1-5
                slot_match = re.match(r"^([a-z]+)(\d)$", key)
                if slot_match:
                    slot_name = slot_match.group(1).capitalize()
                    rank_num = int(slot_match.group(2))

                    if rank_num == 1:
                        rank_label = "BiS"
                    else:
                        rank_label = f"Alt {rank_num - 1}"

                    gear_items.append(f"{slot_name} ({rank_label}): {val}")
                elif not key.isdigit():
                    gear_items.append(f"{key.capitalize()}: {val}")
                else:
                    gear_items.append(val)

            if gear_items:
                parsed_code.replace(
                    template,
                    f"\nEquipment Setup:\n" + "\n".join(gear_items) + "\n",
                )


def parse_downloaded_wikitext():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(INPUT_DIR.glob("batch_*.json.gz"))
    print(f"Found {len(batch_files)} compressed batches to process.")

    processed_count = 0
    skipped_redirects = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for batch_file in batch_files:
            with gzip.open(batch_file, "rt", encoding="utf-8") as f:
                data = json.load(f)

            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if "missing" in page_data:
                    continue

                title = page_data.get("title", "Unknown Title")
                revisions = page_data.get("revisions", [])

                if not revisions:
                    continue

                wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")

                try:
                    parsed = mwparserfromhell.parse(wikitext)

                    # --- STEP 1: PREPROCESS TEMPLATES TO PRESERVE ICONS/STATS ---
                    preprocess_osrs_templates(parsed)

                    # --- STEP 2: STRIP REMAINING WIKICODE ---
                    clean_text = parsed.strip_code()

                    # --- STEP 3: REGEX CLEANUP FOR MARGINAL ARTIFACTS ---
                    # Strip leading alignment keywords
                    clean_text = re.sub(
                        r"^(?:left|right|center|none|thumb)(?:\|?\d+x?\d*px)?",
                        "",
                        clean_text,
                        flags=re.IGNORECASE,
                    )

                    # Catch standalone pixel dimensions left at start
                    clean_text = re.sub(r"^\d+x?\d*px\|?", "", clean_text, flags=re.IGNORECASE)

                    # Remove leftover HTML attributes like style="..." or class="..."
                    clean_text = re.sub(
                        r'\b(?:style|class|align|valign|width|height|rowspan|colspan)\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]+)',
                        "",
                        clean_text,
                        flags=re.IGNORECASE,
                    )

                    # Normalize excess blank lines
                    clean_text = re.sub(r"\n\s*\n", "\n\n", clean_text).strip()
                except Exception:
                    continue

                # Filter out MediaWiki redirect pages
                if clean_text.lower().startswith("redirect"):
                    skipped_redirects += 1
                    continue

                if not clean_text:
                    continue

                processed_count += 1

                article_record = {"title": title, "content": clean_text}
                out_f.write(json.dumps(article_record) + "\n")

    print(f"Pages parsed and processed: {processed_count}")
    print(f"Skipped redirects: {skipped_redirects}")


if __name__ == "__main__":
    parse_downloaded_wikitext()
