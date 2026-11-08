#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

MASTER_PATH = BASE / "VSL_Dataset/master_manifest.csv"
CANDIDATE_PATH = (
    BASE / "VSL_Dataset/duplicate_candidates_for_manual_audit.csv"
)

OUTPUT_PATH = BASE / "VSL_Dataset/master_manifest_deduplicated.csv"
REPORT_PATH = BASE / "VSL_Dataset/deduplication_report.csv"

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

master = pd.read_csv(MASTER_PATH, dtype=str)
candidates = pd.read_csv(CANDIDATE_PATH, dtype=str)

print("=" * 65)
print("APPLY CROSS-SOURCE DUPLICATE EXCLUSIONS")
print("=" * 65)

print("Master records:", len(master))
print("Candidate pairs:", len(candidates))

# --------------------------------------------------
# 2. Validate candidate pairs
# --------------------------------------------------

required = [
    "TUDIEN_sample_id",
    "QIPEDC_sample_id",
    "gloss",
]

for col in required:
    if col not in candidates.columns:
        raise ValueError(f"Missing candidate column: {col}")

if candidates["TUDIEN_sample_id"].duplicated().any():
    raise ValueError("A TUDIEN video appears in multiple candidate pairs.")

if candidates["QIPEDC_sample_id"].duplicated().any():
    raise ValueError("A QIPEDC video appears in multiple candidate pairs.")

tudien_ids = set(candidates["TUDIEN_sample_id"].str.strip())
qipedc_ids = set(candidates["QIPEDC_sample_id"].str.strip())

# --------------------------------------------------
# 3. Identify master manifest ID column
# --------------------------------------------------

id_col = "sample_id"

if id_col not in master.columns:
    raise ValueError("master_manifest.csv has no sample_id column.")

master_ids = set(master[id_col].astype(str).str.strip())

missing_tudien = tudien_ids - master_ids
missing_qipedc = qipedc_ids - master_ids

if missing_tudien:
    raise ValueError(
        f"{len(missing_tudien)} TUDIEN IDs missing from master manifest: "
        f"{sorted(missing_tudien)}"
    )

if missing_qipedc:
    raise ValueError(
        f"{len(missing_qipedc)} QIPEDC IDs missing from master manifest: "
        f"{sorted(missing_qipedc)}"
    )

# --------------------------------------------------
# 4. Initialize duplicate metadata
# --------------------------------------------------

new_columns = {
    "duplicate_status": "not_assessed",
    "duplicate_of": "",
    "duplicate_evidence": "",
    "training_eligible": "pending_audit",
}

for col, default in new_columns.items():
    if col not in master.columns:
        master[col] = default

# --------------------------------------------------
# 5. Mark duplicate TUDIEN records
# --------------------------------------------------

pair_lookup = {}

for _, row in candidates.iterrows():
    tudien_id = str(row["TUDIEN_sample_id"]).strip()
    qipedc_id = str(row["QIPEDC_sample_id"]).strip()

    pair_lookup[tudien_id] = qipedc_id

for tudien_id, qipedc_id in pair_lookup.items():
    mask = master[id_col].astype(str).str.strip() == tudien_id

    master.loc[mask, "duplicate_status"] = "duplicate_candidate"
    master.loc[mask, "duplicate_of"] = qipedc_id

    master.loc[mask, "duplicate_evidence"] = (
        "same_gloss+QIPEDC_marker+region_match+strong_pHash"
    )

    master.loc[mask, "training_eligible"] = "False"

# --------------------------------------------------
# 6. Mark corresponding QIPEDC records
# --------------------------------------------------

for qipedc_id in qipedc_ids:
    mask = master[id_col].astype(str).str.strip() == qipedc_id

    master.loc[mask, "duplicate_status"] = "retained_original"
    master.loc[mask, "duplicate_of"] = ""
    master.loc[mask, "duplicate_evidence"] = (
        "preferred_source_record_for_cross_source_pair"
    )

    # Do not automatically enable training:
    # QIPEDC records still require quality/license/label audit.
    master.loc[mask, "training_eligible"] = "pending_audit"

# --------------------------------------------------
# 7. Validate
# --------------------------------------------------

marked_duplicates = (
    master["duplicate_status"] == "duplicate_candidate"
).sum()

retained_originals = (
    master["duplicate_status"] == "retained_original"
).sum()

if marked_duplicates != len(tudien_ids):
    raise ValueError("Unexpected number of marked TUDIEN duplicates.")

if retained_originals != len(qipedc_ids):
    raise ValueError("Unexpected number of retained QIPEDC records.")

# --------------------------------------------------
# 8. Save deduplicated manifest
# --------------------------------------------------

master.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 9. Create report
# --------------------------------------------------

report = pd.DataFrame([
    {
        "metric": "original_master_records",
        "value": len(master),
    },
    {
        "metric": "duplicate_pairs",
        "value": len(candidates),
    },
    {
        "metric": "unique_tudien_duplicates",
        "value": len(tudien_ids),
    },
    {
        "metric": "unique_qipedc_retained",
        "value": len(qipedc_ids),
    },
    {
        "metric": "records_marked_duplicate",
        "value": marked_duplicates,
    },
    {
        "metric": "records_marked_retained_original",
        "value": retained_originals,
    },
    {
        "metric": "remaining_manifest_records",
        "value": len(master),
    },
])

report.to_csv(
    REPORT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# 10. Summary
# --------------------------------------------------

print("\nDeduplication summary:")
print(report.to_string(index=False))

print("\nDuplicate status distribution:")
print(master["duplicate_status"].value_counts().to_string())

print("\nTraining eligibility distribution:")
print(master["training_eligible"].value_counts().to_string())

print("\nSaved:")
print(OUTPUT_PATH)
print(REPORT_PATH)

print("\nOriginal master_manifest.csv remains unchanged.")
print("Done.")
