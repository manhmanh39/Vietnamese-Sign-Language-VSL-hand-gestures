import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


SOURCE_ID = "tudienngonngukyhieu"
TITLE_PREFIX = "Hướng dẫn thực hiện ký hiệu cho từ "
REVIEW_FIELDS = (
    "canonical_gloss_id", "variant_id", "category_review_status", "signer_id",
    "signer_group_id", "recording_session_id", "concept_review_status",
    "variant_review_status", "video_accessible", "is_vsl", "sign_matches_label",
    "visual_quality", "video_sha256", "perceptual_hash", "duplicate_group_id",
    "duplicate_type", "duplicate_review_status", "is_duplicate_content",
    "license_status", "training_eligible", "review_notes",
)


def normalize_gloss(value):
    value = unicodedata.normalize("NFC", value or "")
    return " ".join(value.split())


def clean_gloss(record):
    title = normalize_gloss(record.get("title", ""))
    if title.casefold().startswith(TITLE_PREFIX.casefold()):
        title = normalize_gloss(title[len(TITLE_PREFIX):])
    if title:
        return title
    page_text = normalize_gloss(record.get("page_text", ""))
    return normalize_gloss(page_text.split(" - Ngôn Ngữ ký Hiệu", 1)[0])


def gloss_key(gloss):
    return normalize_gloss(gloss).casefold()


def provisional_gloss_id(key):
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12].upper()
    return f"G-{digest}"


def extract_vimeo_id(url):
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host != "vimeo.com" and not host.endswith(".vimeo.com"):
        return ""
    for segment in reversed(parsed.path.strip("/").split("/")):
        if re.fullmatch(r"\d+", segment):
            return segment
    return ""


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_existing_reviews(path):
    by_sample_id = {}
    by_source_video = {}
    if not path.exists():
        return by_sample_id, by_source_video
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            by_sample_id[row.get("sample_id", "")] = row
            by_source_video[(row.get("source_id", ""), row.get("source_page_url", ""), row.get("video_id", ""))] = row
    return by_sample_id, by_source_video


def has_manual_review(row):
    fields = (
        "canonical_gloss_id", "variant_id", "signer_id", "signer_group_id",
        "recording_session_id", "video_accessible", "is_vsl", "sign_matches_label",
        "visual_quality", "video_sha256", "perceptual_hash", "duplicate_group_id",
        "duplicate_type", "duplicate_review_status", "is_duplicate_content", "review_notes",
    )
    if any((row.get(field) or "").strip() for field in fields):
        return True
    return (
        row.get("category_review_status", "pending") not in ("", "pending")
        or row.get("concept_review_status", "pending") not in ("", "pending")
        or row.get("variant_review_status", "pending") not in ("", "pending")
        or row.get("license_status", "not_checked") not in ("", "not_checked")
        or row.get("training_eligible", "no").casefold() in {"yes", "true", "1"}
    )


def main():
    parser = argparse.ArgumentParser(description="Create a provenance-preserving VSL metadata training manifest.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("VSL_Dataset"))
    args = parser.parse_args()

    metadata_path = args.dataset_dir / "vsl_metadata.jsonl"
    media_manifest_path = args.dataset_dir / "media_manifest.csv"
    clean_manifest_path = args.dataset_dir / "clean_vocabulary_manifest.csv"
    reviews_by_id, reviews_by_source_video = load_existing_reviews(clean_manifest_path)
    existing_manifest_rows_matched = 0
    manual_review_rows_preserved = 0
    media_status = {}
    with media_manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            media_status[row.get("media_url", "")] = row.get("status", "")

    samples = []
    labels = {}
    label_pages = defaultdict(set)
    label_videos = defaultdict(set)
    label_regions = defaultdict(set)
    missing_gloss = 0
    missing_vimeo = 0
    duplicate_vimeo_pages = Counter()
    key_by_id = {}

    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("page_type") != "vocabulary":
                continue

            gloss = clean_gloss(record)
            if not gloss:
                missing_gloss += 1
                gloss = "<missing gloss>"
            key = gloss_key(gloss)
            candidate_id = provisional_gloss_id(key)
            previous_key = key_by_id.setdefault(candidate_id, key)
            if previous_key != key:
                raise ValueError(f"Provisional gloss ID collision: {candidate_id}")

            label = labels.setdefault(key, {
                "provisional_gloss_id": candidate_id,
                "gloss_candidate": gloss,
            })
            page_url = record.get("page_url", "")
            if page_url:
                label_pages[key].add(page_url)
            regions = record.get("regions") or []
            label_regions[key].update(region for region in regions if isinstance(region, str) and region.strip())
            media = record.get("media", [])
            videos = {}
            for item in media:
                video_url = item.get("url", "")
                video_id = extract_vimeo_id(video_url)
                if video_id:
                    videos[video_id] = video_url
            label_videos[key].update(videos)
            if not videos:
                missing_vimeo += 1

            sample_media = videos.items() if videos else [("", "")]
            for video_id, video_url in sample_media:
                if video_id:
                    duplicate_vimeo_pages[video_id] += 1
                sample_id = f"VSL-{video_id}" if video_id else "VSL-NOVIDEO-" + hashlib.sha256(page_url.encode("utf-8")).hexdigest()[:12].upper()
                sample = {
                    "sample_id": sample_id,
                    "source_id": SOURCE_ID,
                    "source_page_url": page_url,
                    "video_id": video_id,
                    "video_url": video_url,
                    "media_status": media_status.get(video_url, "not_in_manifest") if video_url else "missing_video_reference",
                    "provisional_gloss_id": candidate_id,
                    "gloss_candidate": gloss,
                    "canonical_gloss_id": "",
                    "variant_id": "",
                    "category_candidates_raw": "; ".join(record.get("categories") or []),
                    "category_review_status": "pending",
                    "region_claimed": "; ".join(regions),
                    "signer_id": "",
                    "signer_group_id": "",
                    "recording_session_id": "",
                    "concept_review_status": "pending",
                    "variant_review_status": "pending",
                    "video_accessible": "",
                    "is_vsl": "",
                    "sign_matches_label": "",
                    "visual_quality": "",
                    "video_sha256": "",
                    "perceptual_hash": "",
                    "duplicate_group_id": "",
                    "duplicate_type": "",
                    "duplicate_review_status": "not_checked",
                    "is_duplicate_content": "",
                    "license_status": "not_checked",
                    "training_eligible": "no",
                    "review_notes": "",
                    "split": "",
                }
                prior = reviews_by_id.get(sample_id) or reviews_by_source_video.get((SOURCE_ID, page_url, video_id), {})
                if prior:
                    existing_manifest_rows_matched += 1
                    prior_gloss_changed = prior.get("gloss_candidate", "") != gloss
                    prior_video_changed = prior.get("video_url", "") != video_url
                    for field in REVIEW_FIELDS:
                        if field in prior:
                            sample[field] = prior[field]
                    if prior_gloss_changed:
                        sample.update({
                            "canonical_gloss_id": "",
                            "variant_id": "",
                            "concept_review_status": "pending",
                            "variant_review_status": "pending",
                            "sign_matches_label": "",
                            "training_eligible": "no",
                        })
                    if prior.get("category_candidates_raw", "") != sample["category_candidates_raw"]:
                        sample["category_review_status"] = "pending"
                    if prior_video_changed:
                        sample.update({
                            "video_accessible": "",
                            "is_vsl": "",
                            "sign_matches_label": "",
                            "visual_quality": "",
                            "video_sha256": "",
                            "perceptual_hash": "",
                            "duplicate_group_id": "",
                            "duplicate_type": "",
                            "duplicate_review_status": "not_checked",
                            "is_duplicate_content": "",
                            "license_status": "not_checked",
                            "training_eligible": "no",
                        })
                    if has_manual_review(sample) and not prior_gloss_changed and not prior_video_changed:
                        manual_review_rows_preserved += 1
                samples.append(sample)

    sample_fields = [
        "sample_id", "source_id", "source_page_url", "video_id", "video_url", "media_status",
        "provisional_gloss_id", "gloss_candidate", "canonical_gloss_id", "variant_id",
        "category_candidates_raw", "category_review_status", "region_claimed", "signer_id", "signer_group_id",
        "recording_session_id", "concept_review_status",
        "variant_review_status", "video_accessible", "is_vsl", "sign_matches_label", "visual_quality",
        "video_sha256", "perceptual_hash", "duplicate_group_id", "duplicate_type",
        "duplicate_review_status", "is_duplicate_content", "license_status",
        "training_eligible", "review_notes", "split",
    ]
    write_csv(args.dataset_dir / "clean_vocabulary_manifest.csv", sample_fields, samples)

    catalog_rows = []
    for key, label in sorted(labels.items(), key=lambda item: item[1]["gloss_candidate"].casefold()):
        catalog_rows.append({
            "provisional_gloss_id": label["provisional_gloss_id"],
            "gloss_candidate": label["gloss_candidate"],
            "record_count": len(label_pages[key]),
            "unique_vimeo_ids": len(label_videos[key]),
            "region_claimed_values": "; ".join(sorted(label_regions[key])),
            "concept_review_status": "pending",
            "canonical_gloss_id": "",
        })
    write_csv(
        args.dataset_dir / "provisional_gloss_catalog.csv",
        ["provisional_gloss_id", "gloss_candidate", "record_count", "unique_vimeo_ids", "region_claimed_values", "concept_review_status", "canonical_gloss_id"],
        catalog_rows,
    )

    count_bins = Counter()
    for key in labels:
        count = len(label_videos[key])
        bucket = "0" if count == 0 else "1" if count == 1 else "2-5" if count <= 5 else "6-10" if count <= 10 else ">10"
        count_bins[bucket] += 1
    summary = {
        "source_id": SOURCE_ID,
        "vocabulary_sample_rows": len(samples),
        "provisional_exact_gloss_count": len(labels),
        "records_without_gloss": missing_gloss,
        "records_without_vimeo_reference": missing_vimeo,
        "duplicate_vimeo_ids_across_records": sum(count > 1 for count in duplicate_vimeo_pages.values()),
        "provisional_labels_by_video_count": {bucket: count_bins.get(bucket, 0) for bucket in ("0", "1", "2-5", "6-10", ">10")},
        "license_status_not_checked": sum(row["license_status"] == "not_checked" for row in samples),
        "training_eligible_samples": sum(row["training_eligible"].strip().casefold() in {"yes", "true", "1"} for row in samples),
        "existing_manifest_rows_matched": existing_manifest_rows_matched,
        "manual_review_rows_preserved": manual_review_rows_preserved,
        "region_claimed_is_source_metadata_not_signer_geography": True,
        "normalization": "Unicode NFC, trim/collapse whitespace, case-insensitive exact match only; no fuzzy, synonym, or semantic merging.",
        "limitations": [
            "Provisional gloss IDs group only exact normalized source labels and are not reviewed canonical concept IDs.",
            "Variant IDs and signer IDs require human/video review and are intentionally blank.",
            "Vimeo URLs are references only; this script does not check playback, download videos, or determine usage rights.",
            "Category values are preserved as raw candidates because legacy extraction may include navigation or related-page links.",
            "No samples are marked training-eligible until label, variant, accessibility, and license reviews are complete.",
        ],
    }
    (args.dataset_dir / "clean_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()