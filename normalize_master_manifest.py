import pandas as pd
from pathlib import Path

BASE = Path("VSL_Dataset")

INPUT = BASE / "master_manifest_deduplicated.csv"
OUTPUT = BASE / "master_manifest_normalized.csv"

df = pd.read_csv(INPUT, dtype=str).fillna("")

# Normalize duplicate status.
if "duplicate_status" not in df.columns:
    df["duplicate_status"] = "not_assessed"

# Remove conflicting legacy eligibility columns.
for col in ["training_eligible", "training_eligibility"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Initialize canonical eligibility.
df["training_eligibility"] = "pending_media"

# TUDIEN: references only, no full media verification yet.
tudien = df["source_id"].eq("TUDIEN")
df.loc[tudien, "training_eligibility"] = "pending_media"

# QIPEDC: OCR-enriched records have passed metadata extraction,
# but media accessibility and annotation quality still need auditing.
qipedc = df["source_id"].eq("QIPEDC")
df.loc[qipedc, "training_eligibility"] = "pending_audit"

# Preserve duplicate exclusions.
duplicates = df["duplicate_status"].eq("duplicate_candidate")
df.loc[duplicates, "training_eligibility"] = "excluded_duplicate"

# Report.
print("=" * 65)
print("NORMALIZED MASTER MANIFEST")
print("=" * 65)

print("\nRecords:", len(df))
print("\nEligibility by source:")
print(pd.crosstab(
    df["source_id"],
    df["training_eligibility"],
    margins=True
).to_string())

print("\nDuplicate status:")
print(df["duplicate_status"].value_counts().to_string())

# Integrity checks.
assert len(df) == 12172, "Unexpected record count"
assert df["sample_id"].nunique() == len(df), "Duplicate sample IDs"
assert df["training_eligibility"].notna().all()
assert df.loc[duplicates, "training_eligibility"].eq(
    "excluded_duplicate"
).all()

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("\nSaved:", OUTPUT.resolve())
print("Original manifests unchanged.")
