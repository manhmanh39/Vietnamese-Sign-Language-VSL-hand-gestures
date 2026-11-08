#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QIPEDC dictionary metadata crawler.

Target:
    https://qipedc.moet.gov.vn/dictionary

What it does:
- Fetches the dictionary page (NOT /slang2, which previously triggered a Unity error).
- Extracts dictionary cards, labels, detail links, thumbnails and all data-* attributes.
- Downloads the page's referenced JavaScript files for endpoint/pagination inspection.
- Follows explicit HTML "next" links when present.
- Saves CSV, JSONL, raw HTML, JS assets and a crawl summary.

Important:
- Metadata only. It does NOT download videos.
- A candidate video URL inferred from a thumbnail ID is marked as inferred, not verified.
- Public accessibility does not establish permission to download/train.
- training_eligible is always "no" until the record is manually reviewed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://qipedc.moet.gov.vn"
DEFAULT_URL = f"{BASE_URL}/dictionary"
CARD_SELECTOR = 'a[data-target="#video-modal"]'

FIELDS = [
    "sample_id",
    "source_id",
    "qipedc_entry_id",
    "qipedc_media_id",
    "gloss_original",
    "description",
    "detail_url",
    "thumbnail_url",
    "video_url_observed",
    "video_url_candidate",
    "video_url_status",
    "raw_card_text",
    "raw_attributes_json",
    "page_url",
    "page_number",
    "access_status",
    "license_status",
    "label_status",
    "training_eligible",
]


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def absolute_url(value: str, base: str) -> str:
    value = clean(value)
    if not value or value.startswith(("javascript:", "data:", "blob:")):
        return ""
    return urljoin(base, value)


def sha1_short(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def media_id_from_text(value: str) -> str:
    patterns = [
        r"/thumbs/([A-Za-z0-9_-]+)\.(?:png|jpe?g|webp)(?:[?#].*)?$",
        r"/videos/([A-Za-z0-9_-]+)\.(?:mp4|mov|webm)(?:[?#].*)?$",
        r"/(?:video|videos|thumbs)/([A-Za-z0-9_-]+)(?:[/?#.]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value or "", flags=re.I)
        if match:
            return match.group(1)
    return ""


def collect_attrs(tag) -> dict[str, Any]:
    attrs = {}
    for key, value in (tag.attrs or {}).items():
        attrs[key] = value if isinstance(value, (str, int, float, bool)) else list(value)
    return attrs


def extract_video_urls(tag, page_url: str) -> list[str]:
    found = set()
    for node in [tag, *tag.find_all(True)]:
        for key, value in (node.attrs or {}).items():
            if key in {"src", "href", "data-src", "data-video", "data-url",
                       "data-video-url", "data-mp4", "poster"}:
                vals = value if isinstance(value, list) else [value]
                for item in vals:
                    u = absolute_url(str(item), page_url)
                    if u and re.search(r"\.(mp4|mov|webm)(?:[?#].*)?$", u, re.I):
                        found.add(u)
            if key == "onclick":
                text = str(value)
                for raw in re.findall(
                    r"""['"]([^'"]+\.(?:mp4|mov|webm)(?:\?[^'"]*)?)['"]""",
                    text, flags=re.I
                ):
                    found.add(absolute_url(raw, page_url))
    return sorted(found)


def parse_card(card, page_url: str, page_number: int) -> dict[str, str] | None:
    attrs = collect_attrs(card)
    detail_url = absolute_url(str(attrs.get("href", "")), page_url)

    text_parts = [
        clean(card.get_text(" ", strip=True)),
        clean(attrs.get("title", "")),
        clean(attrs.get("aria-label", "")),
    ]

    image_url = ""
    image_alt = ""
    for img in card.find_all("img"):
        image_url = absolute_url(
            img.get("data-src") or img.get("data-original") or img.get("src") or "",
            page_url,
        )
        image_alt = clean(img.get("alt") or img.get("title") or "")
        if image_url:
            break

    gloss = next((x for x in text_parts if x), "") or image_alt
    # Some dictionary cards put their label in a nearby caption/parent element.
    if not gloss:
        parent = card.parent
        if parent:
            gloss = clean(parent.get_text(" ", strip=True))

    if not gloss and not image_url and not detail_url:
        return None

    media_id = media_id_from_text(image_url) or media_id_from_text(detail_url)
    observed = extract_video_urls(card, page_url)
    candidate = ""
    candidate_status = "not_found"
    if observed:
        candidate = observed[0]
        candidate_status = "observed_in_card_html"
    elif media_id and "/thumbs/" in image_url:
        # This convention was observed during earlier inspection of QIPEDC assets.
        # Keep it as a candidate only; do not treat it as a verified playable URL.
        candidate = f"{BASE_URL}/videos/{media_id}.mp4"
        candidate_status = "inferred_from_thumbnail_id"

    raw_identity = "|".join([detail_url, gloss, image_url, media_id])
    sample_id = "QIPEDC_" + sha1_short(raw_identity)

    # Capture card and child attributes to avoid losing site-specific IDs.
    all_attrs = {
        "card": attrs,
        "children": [
            {"tag": child.name, "attrs": collect_attrs(child)}
            for child in card.find_all(True)
        ],
    }

    return {
        "sample_id": sample_id,
        "source_id": "QIPEDC",
        "qipedc_entry_id": str(
            attrs.get("data-id") or attrs.get("data-entry-id") or ""
        ),
        "qipedc_media_id": media_id,
        "gloss_original": gloss,
        "description": "",
        "detail_url": detail_url,
        "thumbnail_url": image_url,
        "video_url_observed": observed[0] if observed else "",
        "video_url_candidate": candidate,
        "video_url_status": candidate_status,
        "raw_card_text": clean(card.get_text(" ", strip=True)),
        "raw_attributes_json": json.dumps(all_attrs, ensure_ascii=False),
        "page_url": page_url,
        "page_number": str(page_number),
        "access_status": "not_checked",
        "license_status": "not_checked",
        "label_status": "unverified",
        "training_eligible": "no",
    }


def find_next_url(soup: BeautifulSoup, current_url: str) -> str:
    # Prefer semantic rel=next.
    tag = soup.find("a", attrs={"rel": re.compile(r"\bnext\b", re.I)})
    if tag and tag.get("href"):
        return absolute_url(tag["href"], current_url)

    # Otherwise accept an explicit next-page label only.
    for a in soup.find_all("a", href=True):
        label = clean(
            " ".join([
                a.get_text(" ", strip=True),
                str(a.get("aria-label", "")),
                str(a.get("title", "")),
            ])
        ).lower()
        if label in {"next", "next page", "trang sau", "tiếp", "tiếp theo", "»", "›"}:
            return absolute_url(a["href"], current_url)
    return ""


def save_outputs(records: list[dict[str, str]], out_dir: Path, summary: dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_tmp = out_dir / "qipedc_metadata.tmp.csv"
    with csv_tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    csv_tmp.replace(out_dir / "qipedc_metadata.csv")

    with (out_dir / "qipedc_metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary.update({
        "records": len(records),
        "unique_gloss_strings_casefold": len({
            clean(r["gloss_original"]).casefold()
            for r in records if r["gloss_original"]
        }),
        "unique_media_ids": len({
            r["qipedc_media_id"] for r in records if r["qipedc_media_id"]
        }),
        "records_with_thumbnail": sum(bool(r["thumbnail_url"]) for r in records),
        "records_with_observed_video_url": sum(bool(r["video_url_observed"]) for r in records),
        "records_with_inferred_video_candidate": sum(
            r["video_url_status"] == "inferred_from_thumbnail_id" for r in records
        ),
        "training_eligible": sum(r["training_eligible"] == "yes" for r in records),
        "note": (
            "Metadata-only crawl. Video URLs inferred from thumbnail IDs are candidates, "
            "not verified. Rights, access and labels require manual review."
        ),
    })
    (out_dir / "qipedc_crawl_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", default="QIPEDC_Dataset")
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--user-agent", default="Mozilla/5.0 (compatible; VSLResearchCrawler/1.0)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    assets_dir = out_dir / "site_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": args.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    records_by_id: dict[str, dict[str, str]] = {}
    visited: set[str] = set()
    page_url = args.url
    page_number = 0
    summary: dict[str, Any] = {
        "source": "QIPEDC",
        "start_url": args.url,
        "pages_fetched": 0,
        "script_assets": [],
        "warnings": [],
    }

    while page_url and page_number < args.max_pages and page_url not in visited:
        visited.add(page_url)
        page_number += 1
        print(f"[GET] page {page_number}: {page_url}")

        try:
            response = session.get(page_url, timeout=args.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            summary["warnings"].append(f"Page fetch failed: {page_url}: {exc}")
            print(f"[ERROR] {exc}")
            break

        html = response.text
        (assets_dir / f"page_{page_number:03d}.html").write_text(
            html, encoding="utf-8"
        )
        soup = BeautifulSoup(html, "html.parser")
        summary["pages_fetched"] += 1

        # The current QIPEDC dictionary page uses this modal-target selector.
        cards = soup.select(CARD_SELECTOR)
        selector_used = CARD_SELECTOR
        if not cards:
            # Fallback for a markup change: anchors that contain images and
            # reference the dictionary modal or a media-like URL.
            cards = [
                a for a in soup.find_all("a")
                if a.find("img") and (
                    a.get("data-target") or
                    re.search(r"dictionary|video|slang", str(a.get("href", "")), re.I)
                )
            ]
            selector_used = "fallback: image anchors with dictionary/video/slang hints"

        page_count = 0
        for card in cards:
            row = parse_card(card, response.url, page_number)
            if row:
                records_by_id[row["sample_id"]] = row
                page_count += 1

        print(f"[PAGE {page_number}] selector={selector_used}; cards={len(cards)}; parsed={page_count}")

        # Save each page as a checkpoint.
        save_outputs(
            list(records_by_id.values()),
            out_dir,
            {**summary, "last_page_url": response.url, "last_selector": selector_used},
        )

        # Save linked JavaScript assets. This helps identify AJAX/API endpoints
        # if the listing is populated or paginated by JavaScript.
        for script in soup.find_all("script", src=True):
            js_url = absolute_url(script["src"], response.url)
            if not js_url or urlparse(js_url).netloc not in {
                urlparse(BASE_URL).netloc, ""
            }:
                continue
            js_name = Path(urlparse(js_url).path).name or f"script_{len(summary['script_assets'])}.js"
            if not js_name.endswith(".js"):
                js_name += ".js"
            target = assets_dir / js_name
            if target.exists():
                continue
            try:
                js_response = session.get(js_url, timeout=args.timeout)
                if js_response.ok and js_response.text:
                    target.write_text(js_response.text, encoding="utf-8")
                    summary["script_assets"].append(js_url)
                    print(f"[JS] saved {js_url} -> {target}")
            except requests.RequestException as exc:
                summary["warnings"].append(f"JS fetch failed: {js_url}: {exc}")

        next_url = find_next_url(soup, response.url)
        if not next_url or next_url in visited:
            print("[STOP] No explicit HTML next-page link found.")
            break
        page_url = next_url
        time.sleep(max(args.delay, 0.5))

    # Extract likely AJAX/API endpoint strings from saved JS for debugging.
    endpoint_pattern = re.compile(
        r"""['"]([^'"]{0,250}(?:ajax|api|dictionary|video|slang)[^'"]{0,250})['"]""",
        re.I,
    )
    endpoint_lines = []
    for js_file in assets_dir.glob("*.js"):
        source = js_file.read_text(encoding="utf-8", errors="ignore")
        for match in endpoint_pattern.finditer(source):
            endpoint_lines.append(f"{js_file.name}\t{match.group(1)}")
    (out_dir / "qipedc_js_endpoint_candidates.txt").write_text(
        "\n".join(sorted(set(endpoint_lines))), encoding="utf-8"
    )

    final_summary = {
        **summary,
        "pages_fetched": page_number if summary["pages_fetched"] else 0,
        "max_pages_requested": args.max_pages,
    }
    save_outputs(list(records_by_id.values()), out_dir, final_summary)

    print("\n[DONE]")
    print(f"Metadata records: {len(records_by_id)}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"CSV: {out_dir / 'qipedc_metadata.csv'}")
    print(f"JS endpoint candidates: {out_dir / 'qipedc_js_endpoint_candidates.txt'}")
    if not records_by_id:
        print(
            "\nNo cards were parsed. Inspect site_assets/page_001.html and "
            "qipedc_js_endpoint_candidates.txt; the listing may be loaded by AJAX."
        )


if __name__ == "__main__":
    main()
