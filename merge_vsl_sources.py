from pathlib import Path
import pandas as pd
import json
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent

SOURCE1 = ROOT / "VSL_Dataset/clean_vocabulary_manifest.csv"
SOURCE2 = ROOT / "QIPEDC_Dataset/qipedc_enriched_manifest.csv"
MEDIA = ROOT / "VSL_Dataset/media_manifest.csv"

OUT = ROOT / "VSL_Dataset"
OUT.mkdir(exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def clean(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_url(url):
    """Remove query strings so equivalent media URLs can match."""
    url = clean(url)
    if not url:
        return ""
    try:
        p = urlsplit(url)
        return urlunsplit((
            p.scheme.lower(),
            p.netloc.lower(),
            p.path.rstrip("/"),
            "",
            ""
        ))
    except Exception:
        return url


# --------------------------------------------------
# 1. Load source manifests
# --------------------------------------------------

df1 = read_csv(SOURCE1)
df2 = read_csv(SOURCE2)
media = read_csv(MEDIA)

print("SOURCE 1:", len(df1))
print("SOURCE 2:", len(df2))
print("MEDIA:", len(media))


# --------------------------------------------------
# 2. Build media lookup
# --------------------------------------------------

media_lookup = {}

for _, row in media.iterrows():
    key = normalize_url(row.get("media_url", ""))
    if key and key not in media_lookup:
        media_lookup[key] = {
            "media_local_path": row.get("local_path", ""),
            "media_sha256": row.get("sha256", ""),
            "media_download_status": row.get("status", ""),
            "media_last_error": row.get("last_error", ""),
            "media_updated_at": row.get("updated_at", ""),
        }


# --------------------------------------------------
# 3. Normalize source 1
# --------------------------------------------------

records1 = []

for _, r in df1.iterrows():
    video_url = clean(r.get("video_url", ""))

    record = r.to_dict()

    record.update({
        "sample_id": clean(r.get("sample_id", "")),
        "source_id": "TUDIEN",
        "source_sample_id": clean(r.get("video_id", "")),
        "gloss_original": clean(r.get("gloss_candidate", "")),
        "description": "",
        "access_status": clean(r.get("media_status", "")),
        "label_status": clean(r.get("concept_review_status", "")),
        "training_eligibility": clean(r.get("training_eligible", "")),
        "region_source": "source_metadata_claim",
        "overlay_region_clean": "",
        "ocr_gloss_usable": "",
        "region_available": "",
    })

    record.update(media_lookup.get(normalize_url(video_url), {
        "media_local_path": "",
        "media_sha256": "",
        "media_download_status": "",
        "media_last_error": "",
        "media_updated_at": "",
    }))

    records1.append(record)


# --------------------------------------------------
# 4. Normalize source 2
# --------------------------------------------------

records2 = []

for _, r in df2.iterrows():
    record = r.to_dict()

    record.update({
        "sample_id": clean(r.get("sample_id", "")),
        "source_id": "QIPEDC",
        "source_sample_id": clean(r.get("source_sample_id", "")),
        "gloss_original": clean(r.get("gloss_original", "")),
        "description": clean(r.get("description", "")),
        "source_page_url": "",
        "video_id": clean(r.get("source_sample_id", "")),
        "region_source": (
            "video_overlay"
            if clean(r.get("region_ocr_status", "")) == "detected"
            else "unknown"
        ),
    })

    record.update(media_lookup.get(
        normalize_url(record.get("video_url", "")),
        {
            "media_local_path": "",
            "media_sha256": "",
            "media_download_status": "",
            "media_last_error": "",
            "media_updated_at": "",
        }
    ))

    records2.append(record)


# --------------------------------------------------
# 5. Merge without deleting any records
# --------------------------------------------------

master = pd.DataFrame(records1 + records2)

# Standard fields first; source-specific fields remain afterward.
standard_columns = [
    "sample_id",
    "source_id",
    "source_sample_id",
    "gloss_original",
    "description",
    "video_id",
    "video_url",
    "source_page_url",
    "thumbnail_url",
    "illustration_url",
    "region_claimed",
    "region_source",
    "overlay_region_clean",
    "access_status",
    "license_status",
    "label_status",
    "training_eligibility",
    "signer_id",
    "signer_group_id",
    "recording_session_id",
    "media_local_path",
    "media_sha256",
    "media_download_status",
    "ocr_gloss_usable",
    "region_available",
]

columns = standard_columns + [
    c for c in master.columns if c not in standard_columns
]
master = master.reindex(columns=columns)

# Check IDs; do not silently remove duplicates.
if master["sample_id"].eq("").any():
    raise ValueError("Some records have empty sample_id.")

duplicate_ids = master[
    master["sample_id"].duplicated(keep=False)
]

if not duplicate_ids.empty:
    duplicate_ids.to_csv(
        OUT / "master_manifest_duplicate_ids.csv",
        index=False
    )
    raise ValueError(
        f"Duplicate sample_id found: {len(duplicate_ids)} rows. "
        "See master_manifest_duplicate_ids.csv"
    )

# --------------------------------------------------
# 6. Save master manifest and summary
# --------------------------------------------------

output = OUT / "master_manifest.csv"
master.to_csv(output, index=False, encoding="utf-8-sig")

summary = {
    "source_1": {
        "name": "tudienngonngukyhieu",
        "records": len(df1),
    },
    "source_2": {
        "name": "QIPEDC",
        "records": len(df2),
    },
    "total_records": len(master),
    "unique_sample_ids": int(master["sample_id"].nunique()),
    "records_by_source": {
        str(k): int(v)
        for k, v in master["source_id"].value_counts().items()
    },
    "records_with_video_url": int(
        master["video_url"].fillna("").ne("").sum()
    ),
    "records_with_local_media": int(
        master["media_local_path"].fillna("").ne("").sum()
    ),
    "records_with_sha256": int(
        master["media_sha256"].fillna("").ne("").sum()
    ),
    "duplicate_records_removed": 0,
}

with open(OUT / "master_manifest_summary.json", "w",
          encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\nMERGE COMPLETED")
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"\nOutput: {output}")
