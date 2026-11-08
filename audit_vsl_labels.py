import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse


TITLE_PREFIX = "Hướng dẫn thực hiện ký hiệu cho từ "
SCREENING_MIN_VIDEOS = 6


def clean_gloss(record):
    title = " ".join(unicodedata.normalize("NFC", record.get("title") or "").split())
    if title.casefold().startswith(TITLE_PREFIX.casefold()):
        title = " ".join(unicodedata.normalize("NFC", title[len(TITLE_PREFIX):]).split())
    if title:
        return title
    page_text = " ".join(unicodedata.normalize("NFC", record.get("page_text") or "").split())
    return " ".join(page_text.split(" - Ngôn Ngữ ký Hiệu", 1)[0].split())


def vimeo_id(url):
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host != "vimeo.com" and not host.endswith(".vimeo.com"):
        return None
    for segment in reversed(parsed.path.strip("/").split("/")):
        if re.fullmatch(r"\d+", segment):
            return segment
    return None


def video_bin(count):
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 5:
        return "2-5"
    if count <= 10:
        return "6-10"
    return ">10"


def dump_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Audit VSL vocabulary labels and media references.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("VSL_Dataset"))
    args = parser.parse_args()

    metadata_path = args.dataset_dir / "vsl_metadata.jsonl"
    manifest_path = args.dataset_dir / "media_manifest.csv"
    labels = defaultdict(lambda: {
        "gloss_clean": "",
        "record_count": 0,
        "vimeo_ids": set(),
        "page_urls": set(),
        "regions": defaultdict(lambda: {"record_count": 0, "vimeo_ids": set(), "page_urls": set()}),
    })
    media_occurrences = Counter()
    media_urls = defaultdict(set)
    vocabulary_records = 0
    vocabulary_without_vimeo_id = 0
    vocabulary_with_multiple_vimeo_ids = 0
    vimeo_page_counts = Counter()

    with metadata_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("page_type") != "vocabulary":
                continue

            vocabulary_records += 1
            gloss = clean_gloss(record)
            if not gloss:
                gloss = "<missing gloss>"
            key = gloss.casefold()
            group = labels[key]
            group["gloss_clean"] = group["gloss_clean"] or gloss
            group["record_count"] += 1
            page_url = record.get("page_url", "")
            if page_url:
                group["page_urls"].add(page_url)

            ids = set()
            for item in record.get("media", []):
                media_type = item.get("media_type", "<missing>")
                media_occurrences[media_type] += 1
                media_url = item.get("url", "")
                if media_url:
                    media_urls[media_type].add(media_url)
                identifier = vimeo_id(media_url)
                if identifier:
                    ids.add(identifier)

            if not ids:
                vocabulary_without_vimeo_id += 1
            if len(ids) > 1:
                vocabulary_with_multiple_vimeo_ids += 1
            group["vimeo_ids"].update(ids)
            vimeo_page_counts.update(ids)

            regions = record.get("regions") or ["<unspecified>"]
            regions = sorted(set(region.strip() for region in regions if isinstance(region, str) and region.strip()))
            if not regions:
                regions = ["<unspecified>"]
            for region in regions:
                regional = group["regions"][region]
                regional["record_count"] += 1
                regional["vimeo_ids"].update(ids)
                if page_url:
                    regional["page_urls"].add(page_url)

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))

    manifest_type_status = Counter((row.get("media_type", ""), row.get("status", "")) for row in manifest_rows)
    manifest_vimeo_ids = {
        identifier
        for row in manifest_rows
        if (identifier := vimeo_id(row.get("media_url", "")))
    }
    jsonl_vimeo_ids = set(vimeo_page_counts)
    bins = Counter(video_bin(len(group["vimeo_ids"])) for group in labels.values())
    region_totals = Counter()
    for group in labels.values():
        for region, regional in group["regions"].items():
            region_totals[region] += regional["record_count"]

    def label_row(group):
        region_records = {name: data["record_count"] for name, data in sorted(group["regions"].items())}
        region_videos = {name: len(data["vimeo_ids"]) for name, data in sorted(group["regions"].items())}
        video_count = len(group["vimeo_ids"])
        return {
            "gloss_clean": group["gloss_clean"],
            "record_count": group["record_count"],
            "unique_vimeo_ids": video_count,
            "video_count_bin": video_bin(video_count),
            "screening_candidate_ge_6": video_count >= SCREENING_MIN_VIDEOS,
            "region_record_counts": json.dumps(region_records, ensure_ascii=False, sort_keys=True),
            "region_unique_vimeo_ids": json.dumps(region_videos, ensure_ascii=False, sort_keys=True),
            "vimeo_ids": ";".join(sorted(group["vimeo_ids"], key=lambda value: (len(value), value))),
            "page_urls": ";".join(sorted(group["page_urls"])),
        }

    sorted_labels = sorted(labels.values(), key=lambda group: (-len(group["vimeo_ids"]), group["gloss_clean"].casefold()))
    label_rows = [label_row(group) for group in sorted_labels]
    fields = [
        "gloss_clean", "record_count", "unique_vimeo_ids", "video_count_bin",
        "screening_candidate_ge_6", "region_record_counts", "region_unique_vimeo_ids",
        "vimeo_ids", "page_urls",
    ]
    dump_csv(args.dataset_dir / "label_distribution.csv", fields, label_rows)
    dump_csv(
        args.dataset_dir / "label_candidates_ge_6.csv",
        fields,
        [row for row in label_rows if row["screening_candidate_ge_6"]],
    )

    regional_rows = []
    for group in sorted(labels.values(), key=lambda item: item["gloss_clean"].casefold()):
        for region, data in sorted(group["regions"].items()):
            regional_rows.append({
                "gloss_clean": group["gloss_clean"],
                "region": region,
                "record_count": data["record_count"],
                "unique_vimeo_ids": len(data["vimeo_ids"]),
                "vimeo_ids": ";".join(sorted(data["vimeo_ids"], key=lambda value: (len(value), value))),
                "page_urls": ";".join(sorted(data["page_urls"])),
            })
    dump_csv(
        args.dataset_dir / "label_region_distribution.csv",
        ["gloss_clean", "region", "record_count", "unique_vimeo_ids", "vimeo_ids", "page_urls"],
        regional_rows,
    )

    summary = {
        "vocabulary_records": vocabulary_records,
        "normalized_gloss_labels": len(labels),
        "unique_vimeo_ids_in_vocabulary": len(jsonl_vimeo_ids),
        "vocabulary_records_without_vimeo_id": vocabulary_without_vimeo_id,
        "vocabulary_records_with_multiple_vimeo_ids": vocabulary_with_multiple_vimeo_ids,
        "vimeo_ids_referenced_by_multiple_vocabulary_pages": sum(count > 1 for count in vimeo_page_counts.values()),
        "label_counts_by_unique_video_bin": {name: bins.get(name, 0) for name in ("0", "1", "2-5", "6-10", ">10")},
        "screening_candidate_threshold_unique_vimeo_ids": SCREENING_MIN_VIDEOS,
        "screening_candidate_label_count": sum(len(group["vimeo_ids"]) >= SCREENING_MIN_VIDEOS for group in labels.values()),
        "jsonl_vocabulary_media_occurrences_by_type": dict(sorted(media_occurrences.items())),
        "jsonl_vocabulary_distinct_media_urls_by_type": {name: len(urls) for name, urls in sorted(media_urls.items())},
        "manifest_rows": len(manifest_rows),
        "manifest_rows_by_media_type_and_status": {
            f"{media_type}:{status}": count
            for (media_type, status), count in sorted(manifest_type_status.items())
        },
        "manifest_unique_vimeo_ids": len(manifest_vimeo_ids),
        "jsonl_manifest_vimeo_id_sets_match": jsonl_vimeo_ids == manifest_vimeo_ids,
        "vimeo_ids_only_in_jsonl": sorted(jsonl_vimeo_ids - manifest_vimeo_ids),
        "vimeo_ids_only_in_manifest": sorted(manifest_vimeo_ids - jsonl_vimeo_ids),
        "region_tagged_record_counts": dict(sorted(region_totals.items())),
        "notes": [
            "Labels are grouped by NFC-normalized, whitespace-collapsed, case-insensitive exact text after removing the known instructional title prefix; no synonym, fuzzy, or semantic merging is performed.",
            "A page tagged with multiple regions contributes to each region count, so regional totals overlap.",
            "At least 6 unique Vimeo IDs is only a manual-screening threshold, not evidence that a label is training-ready.",
            "A Vimeo ID is an extracted reference, not a check that the video is playable, licensed, or visually valid.",
        ],
    }
    summary_path = args.dataset_dir / "label_distribution_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# VSL Label Distribution Audit",
        "",
        f"- Vocabulary records: {vocabulary_records}",
        f"- Case-insensitive exact labels after prefix removal: {len(labels)}",
        f"- Unique Vimeo IDs in vocabulary records: {len(jsonl_vimeo_ids)}",
        f"- Manifest Vimeo IDs: {len(manifest_vimeo_ids)}; ID sets match: {jsonl_vimeo_ids == manifest_vimeo_ids}",
        f"- Labels with at least {SCREENING_MIN_VIDEOS} unique Vimeo IDs for manual screening: {summary['screening_candidate_label_count']}",
        "",
        "## Labels by Unique Video Count",
        "",
        "| Unique Vimeo IDs per label | Label count |",
        "| --- | ---: |",
    ]
    for name in ("0", "1", "2-5", "6-10", ">10"):
        lines.append(f"| {name} | {bins.get(name, 0)} |")
    lines.extend([
        "",
        "## Media Cross-Check",
        "",
        f"- JSONL vocabulary media occurrences: `{json.dumps(dict(media_occurrences), ensure_ascii=False)}`",
        f"- Distinct JSONL vocabulary media URLs by type: `{json.dumps({name: len(urls) for name, urls in sorted(media_urls.items())}, ensure_ascii=False)}`",
        f"- Manifest rows by type/status: `{json.dumps(summary['manifest_rows_by_media_type_and_status'], ensure_ascii=False)}`",
        "",
        "Regional counts overlap when one page lists multiple regions. Candidate status only prioritizes review; it does not establish licensing, accessibility, or training quality.",
        "",
        "Detailed files: `label_distribution.csv`, `label_region_distribution.csv`, and `label_candidates_ge_6.csv`.",
    ])
    (args.dataset_dir / "label_distribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary": summary,
        "outputs": [
            str(args.dataset_dir / "label_distribution_report.md"),
            str(args.dataset_dir / "label_distribution.csv"),
            str(args.dataset_dir / "label_region_distribution.csv"),
            str(args.dataset_dir / "label_candidates_ge_6.csv"),
            str(summary_path),
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()