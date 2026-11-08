import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def primary_region(record):
    for link in record.get("links", []):
        path = urlparse(link.get("url", "")).path.rstrip("/")
        if path.startswith("/vung-mien/"):
            label = " ".join((link.get("text") or "").split())
            if label:
                return [label]
    return []


def main():
    parser = argparse.ArgumentParser(description="Repair scraped VSL region tags from stored page links.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("VSL_Dataset"))
    parser.add_argument("--apply", action="store_true", help="Write the corrected regions and export dataset files.")
    args = parser.parse_args()

    db_path = args.dataset_dir / "crawler.sqlite3"
    audit_path = args.dataset_dir / "manual_video_audit.csv"
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        records = connection.execute(
            "SELECT page_url,record_json FROM records WHERE json_extract(record_json,'$.page_type')='vocabulary'"
        ).fetchall()
        audit_urls = set()
        if audit_path.exists():
            with audit_path.open(encoding="utf-8-sig", newline="") as stream:
                audit_urls = {row["source_page_url"] for row in csv.DictReader(stream)}

        updates = []
        no_region_link = 0
        old_region_counts = Counter()
        audit_mismatches = []
        corrected_by_url = {}
        for row in records:
            record = json.loads(row["record_json"])
            old_regions = record.get("regions") or []
            corrected = primary_region(record)
            old_region_counts[len(old_regions)] += 1
            if not corrected:
                no_region_link += 1
            if row["page_url"] in audit_urls and old_regions != corrected:
                audit_mismatches.append((row["page_url"], old_regions, corrected))
            corrected_by_url[row["page_url"]] = corrected
            if not args.apply:
                continue
            if "regions_scraped_before_cleanup" not in record:
                record["regions_scraped_before_cleanup"] = old_regions
            record["regions"] = corrected
            updates.append((json.dumps(record, ensure_ascii=False), row["page_url"]))

        audit_missing = audit_urls - corrected_by_url.keys()
        if audit_missing:
            raise RuntimeError(f"{len(audit_missing)} manual audit pages are missing from the metadata records.")
        if audit_mismatches:
            raise RuntimeError(f"The stored primary-region link does not match {len(audit_mismatches)} refreshed audit records.")

        summary = {
            "vocabulary_records": len(records),
            "records_with_primary_region_link": len(records) - no_region_link,
            "records_without_primary_region_link": no_region_link,
            "manual_audit_records_validated": len(audit_urls),
            "manual_audit_mismatches": len(audit_mismatches),
            "old_region_tag_count_distribution": dict(sorted(old_region_counts.items())),
            "mode": "apply" if args.apply else "dry_run",
            "rule": "Use the first /vung-mien/ link in the stored page-link order; validated against the refreshed manual audit pages.",
        }

        if args.apply:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany("UPDATE records SET record_json=? WHERE page_url=?", updates)
            connection.execute(
                """UPDATE pages SET status='done',last_error=NULL
                   WHERE kind='vocabulary' AND url IN (SELECT page_url FROM records)"""
            )
            connection.commit()

    if args.apply:
        from crawl_vsl_dictionary import Store

        store = Store(db_path)
        store.export(args.dataset_dir)
        store.conn.close()

        if audit_path.exists():
            with audit_path.open(encoding="utf-8-sig", newline="") as stream:
                audit_rows = list(csv.DictReader(stream))
            for row in audit_rows:
                regions = corrected_by_url[row["source_page_url"]]
                row["region_claimed"] = "; ".join(regions) if regions else "<unspecified>"
            with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]))
                writer.writeheader()
                writer.writerows(audit_rows)

        report_path = args.dataset_dir / "region_cleanup_report.json"
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()