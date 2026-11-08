
import pandas as pd
import re
import unicodedata
from pathlib import Path
from collections import defaultdict

root = Path(__file__).resolve().parent
df = pd.read_csv(
    root / "VSL_Dataset/master_manifest.csv",
    dtype=str
).fillna("")

def normalize(text, remove_accents=False):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    if remove_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(
            c for c in text
            if unicodedata.category(c) != "Mn"
        )
        text = text.replace("đ", "d")
    return text

df["gloss_exact"] = df["gloss_original"].apply(
    lambda x: normalize(x)
)
df["gloss_candidate_key"] = df["gloss_original"].apply(
    lambda x: normalize(x, True)
)

# Exact normalized matches
exact = (
    df[df["gloss_exact"] != ""]
    .groupby(["gloss_exact", "source_id"])
    .agg(
        record_count=("sample_id", "count"),
        original_labels=("gloss_original",
                         lambda x: " | ".join(sorted(set(x)))),
        regions=("region_claimed",
                 lambda x: " | ".join(sorted(set(v for v in x if v))))
    )
    .reset_index()
)

exact_wide = exact.pivot(
    index="gloss_exact",
    columns="source_id",
    values="record_count"
).fillna(0)

for source in ["TUDIEN", "QIPEDC"]:
    if source not in exact_wide:
        exact_wide[source] = 0

overlap = exact_wide[
    (exact_wide["TUDIEN"] > 0) &
    (exact_wide["QIPEDC"] > 0)
].reset_index()

overlap.to_csv(
    root / "VSL_Dataset/gloss_exact_overlap.csv",
    index=False,
    encoding="utf-8-sig"
)

# Accent-insensitive candidates: manual review only
candidate_groups = (
    df[df["gloss_candidate_key"] != ""]
    .groupby("gloss_candidate_key")
    .agg(
        sources=("source_id", lambda x: set(x)),
        labels=("gloss_original",
                lambda x: " | ".join(sorted(set(x)))),
        record_count=("sample_id", "count")
    )
    .reset_index()
)

candidates = candidate_groups[
    candidate_groups["sources"].apply(
        lambda x: "TUDIEN" in x and "QIPEDC" in x
    )
].copy()

candidates["sources"] = candidates["sources"].apply(
    lambda x: " | ".join(sorted(x))
)

candidates.to_csv(
    root / "VSL_Dataset/gloss_accent_insensitive_candidates.csv",
    index=False,
    encoding="utf-8-sig"
)

print("TOTAL RECORDS:", len(df))
print("EXACT OVERLAPPING GLOSSES:", len(overlap))
print("ACCENT-INSENSITIVE CANDIDATES:", len(candidates))
print("Saved:")
print("- VSL_Dataset/gloss_exact_overlap.csv")
print("- VSL_Dataset/gloss_accent_insensitive_candidates.csv")
