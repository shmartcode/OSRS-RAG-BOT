import requests


def fetch_all_osrs_wiki_titles():
    url = "https://oldschool.runescape.wiki/api.php"
    titles = []
    params = {
        "action": "query",
        "format": "json",
        "list": "allpages",
        "aplimit": "max",
    }

    session = requests.Session()

    while True:
        response = session.get(url, params=params)
        data = response.json()

        for page in data["query"]["allpages"]:
            titles.append(page["title"])

        # MediaWiki uses 'continue' tokens when there are more results to fetch
        if "continue" in data:
            params["apcontinue"] = data["continue"]["apcontinue"]
        else:
            break

    print(f"Fetched {len(titles)} titles from the OSRS Wiki.")
    return titles


if __name__ == "__main__":
    all_titles = fetch_all_osrs_wiki_titles()
    # Optional: save them to a file
    with open("wiki_page_titles.txt", "w", encoding="utf-8") as f:
        for title in all_titles:
            f.write(title + "\n")
