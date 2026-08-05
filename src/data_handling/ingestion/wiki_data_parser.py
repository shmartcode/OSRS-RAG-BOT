import gzip
import json
from pathlib import Path
import mwparserfromhell  # Optional, but highly recommended for cleaning wikitext
import re

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "raw/pages_content_compressed"
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_FILE = OUTPUT_DIR / "clean_wiki_articles.jsonl"


def parse_downloaded_wikitext():
    # Ensure the output folder exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(INPUT_DIR.glob("batch_*.json.gz"))
    print(f"Found {len(batch_files)} compressed batches to process.")

    processed_count = 0
    skipped_redirects = 0

    # Open the output file in write mode ("w")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for batch_file in batch_files:
            with gzip.open(batch_file, "rt", encoding="utf-8") as f:  # open compressed file, rt is read text mode
                data = json.load(f)

            # MediaWiki structure: query -> pages -> { page_id: { title, revisions: [...] } }
            # searches data dictionary for 'query'. if found returns the dictionary found, if not returns empty dict '{}'
            # then within there searches for pages, if found will return dictionary of pages (articles and ids)
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                # Skip error pages or missing entries if any exist
                if "missing" in page_data:
                    continue

                title = page_data.get("title", "Unknown Title")
                revisions = page_data.get("revisions", [])

                if not revisions:  # if there are no revisions then there is no text (there should be at least one revision...)
                    continue

                # Extract the raw MediaWiki markup
                wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")

                # --- CLEANING THE WIKITEXT ---
                # Use mwparserfromhell to strip out templates, tables, and wiki syntax
                #########################
                try:
                    parsed = mwparserfromhell.parse(wikitext)
                    clean_text = parsed.strip_code()

                    # Strip leading alignment keywords even without a trailing pipe (e.g., "center300x150px", "left120px")
                    clean_text = re.sub(
                        r"^(?:left|right|center|none|thumb)(?:\|?\d+x?\d*px)?",
                        "",
                        clean_text,
                        flags=re.IGNORECASE,
                    )

                    # Also catch standalone pixel dimensions left at the start
                    clean_text = re.sub(r"^\d+x?\d*px\|?", "", clean_text, flags=re.IGNORECASE)

                    # Remove leftover HTML attributes like style="..." or class="..."
                    clean_text = re.sub(
                        r'\b(?:style|class|align|valign|width|height|rowspan|colspan)\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]+)',
                        "",
                        clean_text,
                        flags=re.IGNORECASE,
                    )

                    # Clean up excess blank lines resulting from removals
                    clean_text = re.sub(r"\n\s*\n", "\n\n", clean_text).strip()
                except Exception:
                    continue

                ###########################
                # Filter out MediaWiki redirect pages
                if clean_text.lower().startswith("redirect"):
                    skipped_redirects += 1
                    continue

                # Clean up excessive newlines or whitespace
                clean_text = clean_text.strip()
                if not clean_text:
                    continue

                processed_count += 1

                # Package the title and cleaned text into a dictionary
                article_record = {"title": title, "content": clean_text}

                # Write each article as a single JSON line to the file
                out_f.write(json.dumps(article_record) + "\n")

                # Example: Print the first 2 processed articles just to see them
            #     if processed_count <= 4:
            #         print(f"\n--- Article: {title} ---")
            #         print(clean_text[:400] + "..." if len(clean_text) > 400 else clean_text)
            #     if processed_count >= 100:
            #         break
            # if processed_count >= 100:
            #     break

    print(f"Pages parsed and processed: {processed_count}")


if __name__ == "__main__":
    parse_downloaded_wikitext()
