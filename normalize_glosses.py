"""
normalize_glosses.py
Normalizes raw gloss strings from VSL & QIPEDC manifests while strictly
preserving Vietnamese diacritics and semantic meaning.
"""

import re
import unicodedata
from pathlib import Path
import pandas as pd

# Paths to manifest files
INPUT_MANIFEST = Path("VSL_Dataset/master_manifest_downloaded.csv")
FALLBACK_MANIFEST = Path("VSL_Dataset/master_manifest_normalized.csv")
OUTPUT_PATH = Path("VSL_Dataset/standardized_glosses.csv")
SUMMARY_PATH = Path("VSL_Dataset/gloss_vocabulary_summary.csv")


def normalize_vietnamese_gloss(gloss: str) -> str:
    """Cleans and standardizes a Vietnamese gloss label.
    Preserves all Vietnamese diacritics (Unicode NFC).
    """
    if not isinstance(gloss, str) or not gloss.strip():
        return ""

    # Ensure NFC Unicode composition
    text = unicodedata.normalize("NFC", gloss.strip().lower())

    # Replace multiple spaces / underscores / dashes with single space
    text = re.sub(r"[\t\n\r_]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove trailing/leading special characters but preserve letters, numbers, spaces
    # Vietnamese characters are within \w in Python 3 regex
    text = re.sub(r"[^\w\s\(\)\-\.,/]", "", text)
    text = text.strip()

    return text


def extract_base_concept(gloss: str) -> str:
    """Extracts base concept removing parenthetical annotations if needed,
    e.g. '0 (số không)' -> 'số không', 'π (số pi)' -> 'số pi'
    """
    if not gloss:
        return ""
    # Match patterns like "0 (số không)" -> "số không"
    m = re.match(r"^[\d\W_]+\((.+)\)$", gloss)
    if m:
        inner = m.group(1).strip()
        if inner:
            return inner
    # Clean inner parentheses e.g. "con trai (không có)" -> "con trai"
    text = re.sub(r"\(không có\)", "", gloss).strip()
    return text


def main():
    if INPUT_MANIFEST.exists():
        manifest_path = INPUT_MANIFEST
    elif FALLBACK_MANIFEST.exists():
        manifest_path = FALLBACK_MANIFEST
    else:
        raise FileNotFoundError("Could not find downloaded or normalized master manifest.")

    print(f"Loading master manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path, dtype=str, low_memory=False).fillna("")

    print(f"Loaded {len(df)} rows.")

    # Determine original gloss column
    gloss_col = "gloss_original" if "gloss_original" in df.columns else "gloss_candidate"

    df["raw_gloss"] = df[gloss_col]
    df["normalized_gloss"] = df["raw_gloss"].apply(normalize_vietnamese_gloss)
    df["base_concept"] = df["normalized_gloss"].apply(extract_base_concept)

    # Filter out empty glosses
    df_valid = df[df["normalized_gloss"] != ""].copy()
    print(f"Valid records with non-empty gloss: {len(df_valid)}")

    # Standardize columns for downstream training and grammar engines
    out_cols = [
        "sample_id",
        "source_id",
        "raw_gloss",
        "normalized_gloss",
        "base_concept",
        "region_claimed",
        "category_candidates_raw",
        "media_local_path",
        "media_download_status",
        "media_duration_sec",
    ]
    # Retain available columns
    available_cols = [c for c in out_cols if c in df_valid.columns]
    standardized_df = df_valid[available_cols]

    standardized_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved standardized glosses to {OUTPUT_PATH} ({len(standardized_df)} samples)")

    # Vocabulary summary
    vocab_summary = (
        standardized_df.groupby("normalized_gloss")
        .agg(
            sample_count=("sample_id", "count"),
            sources=("source_id", lambda s: ", ".join(sorted(set(s)))),
            regions=("region_claimed", lambda r: ", ".join(sorted(set(filter(None, r))))),
            has_local_media=(
                "media_local_path",
                lambda p: sum(1 for x in p if str(x).strip() and Path(str(x)).exists()),
            ),
        )
        .reset_index()
        .sort_values(by="sample_count", ascending=False)
    )

    vocab_summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved vocabulary summary to {SUMMARY_PATH} ({len(vocab_summary)} unique glosses)")
    print("\nTop 15 Most Frequent Glosses:")
    print(vocab_summary.head(15)[["normalized_gloss", "sample_count", "has_local_media"]])


if __name__ == "__main__":
    main()
