
import requests
import json
import csv
from pathlib import Path

BASE_URL = "https://qipedc.moet.gov.vn"
API_URL = f"{BASE_URL}/dictionary/getAll"

OUTPUT_DIR = Path("QIPEDC_Dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": f"{BASE_URL}/dictionary",
    "X-Requested-With": "XMLHttpRequest",
}

response = requests.post(
    API_URL,
    data={"group": 20, "text": ""},
    headers=headers,
    timeout=60
)

response.raise_for_status()
result = response.json()

records = result.get("data", [])

print(f"Total records: {len(records)}")

# Save raw API response
with open(OUTPUT_DIR / "qipedc_api_raw.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# Normalize metadata
rows = []

for item in records:
    item_id = str(item.get("_id", ""))

    rows.append({
        "source_id": "QIPEDC",
        "source_sample_id": item_id,
        "gloss_original": item.get("word", ""),
        "description": item.get("description", ""),
        "has_image": item.get("i", ""),
        "video_url": f"{BASE_URL}/videos/{item_id}.mp4",
        "thumbnail_url": f"{BASE_URL}/thumbs/{item_id}.png",
        "illustration_url": (
            f"{BASE_URL}/Anh/{item_id[:5]}.png"
            if item.get("i") == "true" else ""
        ),
        "access_status": "not_checked",
        "license_status": "not_checked",
        "label_status": "unverified",
    })

if rows:
    with open(
        OUTPUT_DIR / "qipedc_metadata.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print(f"Saved JSON: {OUTPUT_DIR}/qipedc_api_raw.json")
print(f"Saved CSV: {OUTPUT_DIR}/qipedc_metadata.csv")

if rows:
    print("\nFirst record:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))
