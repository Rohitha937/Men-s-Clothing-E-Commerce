import csv
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "static" / "licensed_product_images"
MANIFEST_PATH = BASE_DIR / "data" / "licensed_image_manifest.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MrStorefrontImageCollector/1.0 (educational project)"
PER_CATEGORY_TARGET = 48

QUERIES = {
    "shirt": [
        "shirt white background",
        "polo shirt white background",
        "shirt isolated clothing",
        "folded shirt clothing",
        "formal shirt product photo",
    ],
    "tshirt": [
        "t-shirt white background",
        "tshirt isolated clothing",
        "t shirt product photo",
        "short sleeve shirt isolated",
        "tee shirt white background",
    ],
    "jeans": [
        "jeans white background",
        "blue jeans isolated",
        "denim pants product photo",
        "jeans clothing isolated",
        "jeans apparel product photo",
    ],
    "trousers": [
        "trousers white background",
        "pants isolated clothing",
        "formal trousers product photo",
        "chinos white background",
        "slacks isolated clothing",
    ],
}

KEYWORDS = {
    "shirt": ["shirt", "polo"],
    "tshirt": ["shirt", "t-shirt", "tshirt", "tee"],
    "jeans": ["jeans", "denim"],
    "trousers": ["trouser", "trousers", "pants", "chino", "slacks"],
}


def commons_session():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def load_manifest():
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_manifest(rows):
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category",
                "filename",
                "title",
                "source_url",
                "description_url",
                "license_source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def search_files(session, search_term, offset=0):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "gsroffset": offset,
        "prop": "imageinfo",
        "iiprop": "url|mime",
        "origin": "*",
    }
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def choose_image_url(page):
    info = (page.get("imageinfo") or [{}])[0]
    url = info.get("url")
    if not url:
        return None
    if not re.search(r"\.(jpg|jpeg|png|webp)$", url, re.I):
        return None
    return url


def category_count(rows, category):
    return sum(1 for row in rows if row["category"] == category)


def next_filename(category, url, index):
    extension = ".jpg" if url.lower().endswith((".jpg", ".jpeg")) else ".png"
    return f"{category}-{index:02d}{extension}"


def relevant_title(category, title):
    lowered = title.lower()
    return any(token in lowered for token in KEYWORDS[category])


def existing_titles(rows, category):
    return {row["title"] for row in rows if row["category"] == category}


def fetch_category(session, category, manifest_rows):
    titles = existing_titles(manifest_rows, category)
    current_count = category_count(manifest_rows, category)
    for query in QUERIES[category]:
        if current_count >= PER_CATEGORY_TARGET:
            break
        for offset in (0, 50, 100, 150):
            if current_count >= PER_CATEGORY_TARGET:
                break
            data = search_files(session, query, offset)
            pages = (data.get("query") or {}).get("pages", {})
            for page in pages.values():
                if current_count >= PER_CATEGORY_TARGET:
                    break
                title = page.get("title", "")
                url = choose_image_url(page)
                if not url or title in titles or not relevant_title(category, title):
                    continue

                filename = next_filename(category, url, current_count + 1)
                output_path = OUT_DIR / filename
                try:
                    response = session.get(url, timeout=60)
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    if exc.response is not None and exc.response.status_code == 429:
                        print("rate limited on", title, "sleeping before retry...")
                        time.sleep(20)
                        continue
                    raise

                output_path.write_bytes(response.content)
                manifest_rows.append(
                    {
                        "category": category,
                        "filename": filename,
                        "title": title,
                        "source_url": url,
                        "description_url": f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'))}",
                        "license_source": "Wikimedia Commons",
                    }
                )
                titles.add(title)
                current_count += 1
                print(category, current_count, filename)
                save_manifest(manifest_rows)
                time.sleep(2)


def main():
    session = commons_session()
    manifest_rows = load_manifest()
    for category in ("shirt", "tshirt", "jeans", "trousers"):
        fetch_category(session, category, manifest_rows)
    print("done")
    for category in ("shirt", "tshirt", "jeans", "trousers"):
        print(category, category_count(manifest_rows, category))


if __name__ == "__main__":
    main()
