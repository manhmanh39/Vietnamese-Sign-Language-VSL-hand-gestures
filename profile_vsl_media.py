import pandas as pd
import numpy as np
import json
from pathlib import Path

audit_path = "VSL_Dataset/media_integrity_audit.csv"
manifest_path = "VSL_Dataset/master_manifest_downloaded.csv"

out_dir = Path("VSL_Dataset")
audit = pd.read_csv(audit_path, low_memory=False)
manifest = pd.read_csv(manifest_path, low_memory=False)

df = audit.merge(
    manifest[[
        "sample_id",
        "gloss_original",
        "region_claimed",
        "region_source",
        "overlay_region_clean",
        "training_eligibility",
        "duplicate_status",
        "license_status"
    ]],
    on="sample_id",
    how="left",
    validate="one_to_one"
)

# Numeric conversion
numeric_cols = [
    "duration_sec", "width", "height",
    "fps", "file_size_mb"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["resolution"] = (
    df["width"].astype("Int64").astype(str)
    + "x"
    + df["height"].astype("Int64").astype(str)
)

df["megapixels"] = (
    df["width"] * df["height"] / 1_000_000
)

# Save full profile
df.to_csv(
    out_dir / "media_profile.csv",
    index=False
)

# Summary function
def describe(series):
    x = series.dropna()
    if len(x) == 0:
        return {}
    return {
        "count": int(x.count()),
        "min": float(x.min()),
        "p25": float(x.quantile(.25)),
        "median": float(x.median()),
        "mean": float(x.mean()),
        "p75": float(x.quantile(.75)),
        "max": float(x.max())
    }

summary = {
    "total_records": int(len(df)),
    "valid_records": int(
        df["integrity_status"].eq("valid").sum()
    ),
    "by_source": {},
    "duration_sec": describe(df["duration_sec"]),
    "fps": describe(df["fps"]),
    "width": describe(df["width"]),
    "height": describe(df["height"]),
    "file_size_mb": describe(df["file_size_mb"]),
    "resolution_distribution": (
        df["resolution"].value_counts(dropna=False).to_dict()
    ),
    "codec_distribution": (
        df["video_codec"].value_counts(dropna=False).to_dict()
    ),
    "missing_metadata": {
        col: int(df[col].isna().sum())
        for col in numeric_cols
    }
}

for source, group in df.groupby("source_id"):
    summary["by_source"][source] = {
        "records": int(len(group)),
        "duration_sec": describe(group["duration_sec"]),
        "fps": describe(group["fps"]),
        "file_size_mb": describe(group["file_size_mb"]),
        "resolution_distribution": (
            group["resolution"].value_counts(dropna=False).to_dict()
        )
    }

with open(
    out_dir / "media_profile_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

# Potential outliers: report only, do not exclude
issues = df[
    df["duration_sec"].isna()
    | (df["duration_sec"] <= 0)
    | (df["duration_sec"] < 1)
    | (df["width"].isna())
    | (df["height"].isna())
    | (df["fps"].isna())
    | (df["fps"] <= 0)
].copy()

issues.to_csv(
    out_dir / "media_profile_issues.csv",
    index=False
)

print("=" * 60)
print("MEDIA PROFILE SUMMARY")
print("=" * 60)

print("\nTotal:", len(df))
print("\nBy source:")
print(df["source_id"].value_counts())

print("\nDuration (seconds):")
print(df["duration_sec"].describe(
    percentiles=[.25, .5, .75, .9, .95]
))

print("\nFPS:")
print(df["fps"].describe())

print("\nResolution:")
print(df["resolution"].value_counts(dropna=False).to_string())

print("\nVideo codec:")
print(df["video_codec"].value_counts(dropna=False).to_string())

print("\nFile size (MB):")
print(df["file_size_mb"].describe())

print("\nPotential metadata/duration issues:", len(issues))

print("\nSaved:")
print("- VSL_Dataset/media_profile.csv")
print("- VSL_Dataset/media_profile_summary.json")
print("- VSL_Dataset/media_profile_issues.csv")
