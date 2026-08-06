import json
from pathlib import Path
import requests
import time
import gzip

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


class OSRSDataFetcher:
    def __init__(self):
        self.headers = {"User-Agent": ("OSRS-RAG-Learning-Project (contact martin.816@wright.edu)")}

    def fetch_all_wiki_pages(self):
        url = "https://oldschool.runescape.wiki/api.php"
        all_pages = []
        apcontinue = None

        print("Crawling OSRS Wiki page index...")
        past_length = 0

        while True:
            params = {
                "action": "query",
                "format": "json",
                "list": "allpages",
                "apnamespace": "0",  # Main namespace (articles, guides, items, NPCs)
                "aplimit": "500",  # Max batch size allowed per request
            }
            if apcontinue:
                params["apcontinue"] = apcontinue

            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Collect page titles from this batch
            pages = data.get("query", {}).get("allpages", [])
            for page in pages:
                all_pages.append({"pageid": page["pageid"], "title": page["title"]})

            curr_length = len(all_pages)
            if curr_length > past_length + 10000:
                past_length += 10000
                print(f"Fetched {past_length} titles so far...")

            # Check if there is another batch
            if "continue" in data and "apcontinue" in data["continue"]:
                apcontinue = data["continue"]["apcontinue"]
            else:
                break

        # Save the full index of pages
        index_path = DATA_DIR / "wiki_page_index.json"
        index_path.write_text(json.dumps(all_pages, indent=4))
        print(f"Crawl complete. Total pages indexed: {len(all_pages)}")
        return all_pages

    def chunk_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def fetch_page_contents_compressed(self):
        url = "https://oldschool.runescape.wiki/api.php"
        headers = {
            "User-Agent": "OSRSRAGProject/1.0 (Contact: your_email@example.com)",
            "Accept-Encoding": "gzip",
        }
        INDEX_FILE = DATA_DIR / "wiki_page_index.json"
        OUTPUT_DIR = DATA_DIR / "pages_content_compressed"
        OUTPUT_DIR.mkdir(exist_ok=True)

        if not INDEX_FILE.exists():
            print("Index file not found. Run the title crawler first.")
            return

        pages = json.loads(INDEX_FILE.read_text())
        print(f"Loaded {len(pages)} pages. Starting compressed download...")

        batches = list(self.chunk_list(pages, 50))
        total_batches = len(batches)
        completed_batches = set()
        for file_path in OUTPUT_DIR.glob("batch_*.json.gz"):
            # Extract the index number from 'batch_0042.json.gz'
            try:
                idx = int(file_path.stem.split("_")[1].split(".")[0])
                completed_batches.add(idx)  # track batch numbers in case data fetcher stops without completing
            except ValueError:
                continue

        print(f"Found {len(completed_batches)}/{total_batches} batches already" " downloaded.")
        print("Starting resumable download...")

        for i, batch in enumerate(batches):
            if i in completed_batches:  # skip if batch already downloaded
                continue
            titles = "|".join([p["title"] for p in batch])

            params = {
                "action": "query",
                "format": "json",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": titles,
            }

            success = False
            retries = 3

            # Basic retry loop for network hiccups
            while retries > 0 and not success:
                try:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    response.raise_for_status()

                    # 3. Save directly as a compressed .json.gz file
                    batch_file = OUTPUT_DIR / f"batch_{i:04d}.json.gz"
                    with gzip.open(batch_file, "wt", encoding="utf-8") as f:
                        f.write(response.text)

                    success = True
                except Exception as e:
                    retries -= 1
                    print(f"Error fetching batch {i}: {e}")

            # if retries < 1:
            if not success:
                print(f"Failed to download batch {i} after multiple attempts. Stopping.")
                break

            print(f"Progress: Batch {i+1}/{len(batches)} saved (Compressed)" f" ({((i+1)/len(batches))*100:.1f}%)...")

            time.sleep(0.3)

        print("All wiki article contents downloaded and compressed successfully.")


if __name__ == "__main__":

    fetcher = OSRSDataFetcher()
    fetcher.fetch_all_wiki_pages()
    fetcher.fetch_page_contents_compressed()
