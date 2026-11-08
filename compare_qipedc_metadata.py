
import pandas as pd
from pathlib import Path

META = Path("QIPEDC_Dataset/qipedc_metadata.csv")
OCR = Path("QIPEDC_Dataset/qipedc_overlay_audit_normalized.csv")
OUTPUT = Path("QIPEDC_Dataset/qipedc_metadata_compare_issues.csv")

a = pd.read_csv(META, dtype=str).fillna("")
b = pd.read_csv(OCR, dtype=str).fillna("")

KEY = "source_sample_id"

# 1. Check key integrity
print("Metadata rows:", len(a))
print("OCR rows:", len(b))
print("Metadata duplicate IDs:", a[KEY].duplicated().sum())
print("OCR duplicate IDs:", b[KEY].duplicated().sum())

ids_a = set(a[KEY])
ids_b = set(b[KEY])

print("Missing from OCR:", len(ids_a - ids_b))
print("Missing from metadata:", len(ids_b - ids_a))

# 2. Compare original metadata fields
fields = [
    "source_id",
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

common = ids_a & ids_b
a = a.set_index(KEY)
b = b.set_index(KEY)

issues = []

for sample_id in sorted(common):
    for field in fields:
        if field not in a.columns or field not in b.columns:
            continue

        va = " ".join(a.at[sample_id, field].split())
        vb = " ".join(b.at[sample_id, field].split())

        if va != vb:
            issues.append({
                "source_sample_id": sample_id,
                "field": field,
                "metadata_value": va,
                "ocr_file_value": vb,
            })

issues_df = pd.DataFrame(
    issues,
    columns=[
        "source_sample_id", "field",
        "metadata_value", "ocr_file_value"
    ]
)

issues_df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

# 3. OCR quality summary
print("\n--- OCR quality ---")
print(b["gloss_match"].value_counts(dropna=False).to_string())
print("\n--- Region summary ---")
print(b["overlay_region_clean"].value_counts(dropna=False).to_string())

print("\n--- Metadata comparison ---")
print("Compared IDs:", len(common))
print("Field mismatches:", len(issues_df))
print("Affected records:", issues_df["source_sample_id"].nunique()
      if len(issues_df) else 0)
print("Issues saved:", OUTPUT)
