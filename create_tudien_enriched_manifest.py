
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


OUTPUT = Path("VSL_Dataset/tudien_vsl_enriched_manifest.csv")

SOURCE_ID = "TUDIEN_NNKH"


def get_column(df, aliases, required=False):
    for name in aliases:
        if name in df.columns:
            return df[name].fillna("").astype(str).str.strip()

    if required:
        raise ValueError(
            f"Missing required column. Expected one of: {aliases}"
        )

    return pd.Series([""] * len(df), index=df.index)


def make_sample_id(row):
    video_id = row["video_id"]
    if video_id:
        return f"TUDIEN:{video_id}"

    token = row["video_url"] or row["source_page_url"] or str(row.name)
    digest = hashlib.sha256(
        f"{SOURCE_ID}|{token}".encode("utf-8")
    ).hexdigest()[:16]

    return f"TUDIEN:{digest}"


def normalize_region(value):
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return "unknown"

    if "bắc" in text:
        return "Miền Bắc"
    if "nam" in text:
        return "Miền Nam"
    if "trung" in text:
        return "Miền Trung"
    if "toàn quốc" in text or "cả nước" in text:
        return "Toàn quốc"

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the original tudienngonngukyhieu metadata CSV"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    raw = pd.read_csv(input_path, dtype=str).fillna("")

    # Preserve every original field
    df = raw.copy()

    # Standardized fields for cross-source comparison
    df["source_id"] = SOURCE_ID

    df["source_sample_id"] = get_column(
        df, ["vimeo_id", "video_id", "sample_id"]
    )

    df["gloss_original"] = get_column(
        df, ["gloss_clean", "gloss_original", "gloss", "label"],
        required=True
    )

    df["video_id"] = get_column(
        df, ["vimeo_id", "video_id"]
    )

    df["video_url"] = get_column(
        df, ["vimeo_url", "video_url", "media_url"]
    )

    df["source_page_url"] = get_column(
        df, ["source_page_url", "detail_url", "page_url"]
    )

    df["region_claimed"] = get_column(
        df, ["region_claimed", "region", "region_label"]
    )

    df["signer_id"] = get_column(
        df, ["signer_id", "signer", "person_id"]
    )

    df["license_status"] = get_column(
        df, ["license_status", "license", "copyright_status"]
    )

    df["access_status"] = get_column(
        df, ["video_accessible", "access_status"]
    )

    df["label_status"] = get_column(
        df, ["sign_matches_label", "label_status"]
    )

    df["duplicate_source_flag"] = get_column(
        df, ["is_duplicate_content", "is_duplicate"]
    )

    df["notes"] = get_column(df, ["notes", "note"])

    # Normalize region without overwriting original claim
    df["region_normalized"] = df["region_claimed"].apply(
        normalize_region
    )

    # Stable sample identity
    df["sample_id"] = df.apply(make_sample_id, axis=1)

    # Keep training eligibility undecided
    df["training_eligibility"] = "pending_audit"

    # Check IDs
    if df["sample_id"].duplicated().any():
        duplicates = df.loc[
            df["sample_id"].duplicated(keep=False), "sample_id"
        ].tolist()
        raise ValueError(
            f"Duplicate sample IDs detected: {duplicates[:10]}"
        )

    # Check missing identifiers and video URLs
    missing_video_id = (df["video_id"] == "").sum()
    missing_video_url = (df["video_url"] == "").sum()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("=" * 50)
    print("TUDIEN ENRICHED MANIFEST CREATED")
    print("=" * 50)
    print("Input records:", len(df))
    print("Unique sample IDs:", df["sample_id"].nunique())
    print("Missing video IDs:", missing_video_id)
    print("Missing video URLs:", missing_video_url)
    print("\nRegion distribution:")
    print(df["region_normalized"].value_counts().to_string())
    print("\nOutput:", OUTPUT)


if __name__ == "__main__":
    main()
