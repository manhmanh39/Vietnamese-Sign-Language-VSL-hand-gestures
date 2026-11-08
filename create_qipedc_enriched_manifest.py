
import pandas as pd
from pathlib import Path

# Input files
META_PATH = Path("QIPEDC_Dataset/qipedc_metadata.csv")
OCR_PATH = Path("QIPEDC_Dataset/qipedc_overlay_audit_normalized.csv")

# Output file
OUTPUT_PATH = Path("QIPEDC_Dataset/qipedc_enriched_manifest.csv")

KEY = "source_sample_id"


def main():
    metadata = pd.read_csv(META_PATH, dtype=str).fillna("")
    ocr = pd.read_csv(OCR_PATH, dtype=str).fillna("")

    # 1. Validate primary keys
    for name, df in [("Metadata", metadata), ("OCR", ocr)]:
        if KEY not in df.columns:
            raise ValueError(f"{name} missing key: {KEY}")

        if df[KEY].duplicated().any():
            raise ValueError(f"{name} contains duplicate IDs")

    ids_meta = set(metadata[KEY])
    ids_ocr = set(ocr[KEY])

    if ids_meta != ids_ocr:
        raise ValueError(
            f"ID mismatch: "
            f"metadata_only={len(ids_meta - ids_ocr)}, "
            f"ocr_only={len(ids_ocr - ids_meta)}"
        )

    # 2. Select original metadata fields
    metadata_fields = [
        "source_id",
        "source_sample_id",
        "gloss_original",
        "description",
        "has_image",
        "video_url",
        "thumbnail_url",
        "illustration_url",
        "access_status",
        "license_status",
        "label_status",
    ]

    # 3. Select OCR fields
    ocr_fields = [
        "source_sample_id",
        "overlay_gloss",
        "overlay_region",
        "region_claimed",
        "gloss_match",
        "overlay_status",
        "overlay_region_raw",
        "overlay_region_clean",
        "region_ocr_status",
        "video_http_status",
    ]

    missing_meta = [
        col for col in metadata_fields
        if col not in metadata.columns
    ]
    missing_ocr = [
        col for col in ocr_fields
        if col not in ocr.columns
    ]

    if missing_meta or missing_ocr:
        raise ValueError(
            f"Missing metadata columns: {missing_meta}\n"
            f"Missing OCR columns: {missing_ocr}"
        )

    metadata = metadata[metadata_fields].copy()
    ocr = ocr[ocr_fields].copy()

    # 4. Merge using the stable source ID
    enriched = metadata.merge(
        ocr,
        on=KEY,
        how="inner",
        validate="one_to_one",
        suffixes=("", "_ocr"),
    )

    # 5. Preserve source identity
    enriched.insert(
        0,
        "sample_id",
        "QIPEDC:" + enriched[KEY].astype(str)
    )

    # 6. Add dataset eligibility flags
    enriched["ocr_gloss_usable"] = (
        enriched["gloss_match"] == "yes"
    )

    enriched["region_available"] = (
        enriched["overlay_region_clean"].isin(
            ["Miền Bắc", "Miền Trung", "Miền Nam", "Toàn quốc"]
        )
    )

    # Do not infer training eligibility from OCR alone.
    enriched["training_eligibility"] = "pending_audit"

    # 7. Sort deterministically
    enriched = enriched.sort_values(
        by=KEY,
        kind="stable"
    ).reset_index(drop=True)

    # 8. Final integrity checks
    if len(enriched) != len(metadata):
        raise ValueError(
            f"Unexpected row count: {len(enriched)} "
            f"(expected {len(metadata)})"
        )

    if enriched[KEY].duplicated().any():
        raise ValueError("Duplicate IDs after merge")

    # 9. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 50)
    print("QIPEDC ENRICHED MANIFEST CREATED")
    print("=" * 50)
    print("Total records:", len(enriched))
    print("Unique sample IDs:", enriched["sample_id"].nunique())
    print(
        "OCR gloss usable:",
        enriched["ocr_gloss_usable"].sum()
    )
    print(
        "Region available:",
        enriched["region_available"].sum()
    )
    print(
        "Pending training audit:",
        (enriched["training_eligibility"] == "pending_audit").sum()
    )
    print("Output:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
