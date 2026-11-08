#!/usr/bin/env python3
"""
VSL dictionary crawler
- Discovers category, vocabulary, phrase, song and other relevant pages linked from the site.
- Follows category pagination and detail-page links.
- Saves metadata and media references to SQLite + JSONL/CSV.
- Resumes from SQLite after interruption.
- Downloads direct media files only when explicitly enabled and rights are confirmed.
- Does not bypass login, CAPTCHA, paywalls, hotlink protection or platform restrictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

DEFAULT_BASE = "https://tudienngonngukyhieu.com/"
BOT_NAME = "VSLResearchCrawler"
RELEVANT_PATHS = (
    "/phan-loai/", "/tu-ngu/", "/cau/", "/bai-hat/",
    "/chu-de/", "/bang-chu-cai/", "/a-z/", "/hoc-tieng-anh/"
)
MEDIA_EXTENSIONS = {
    ".mp4": "video", ".webm": "video", ".mov": "video", ".m4v": "video",
    ".avi": "video", ".mkv": "video",
    ".gif": "image", ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".webp": "image", ".bmp": "image", ".svg": "image",
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio", ".m4a": "audio"
}

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def canonical_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    # Keep query parameters because pagination may be query-based.
    query = urlencode(sorted(parse_qsl(p.query, keep_blank_values=True)))
    return p._replace(fragment="", query=query).geturl().rstrip("/")

def safe_name(value: str, max_len=100) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return (value or "unnamed")[:max_len]

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    })
    retry = Retry(
        total=4, connect=4, read=3, status=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            kind TEXT,
            status TEXT DEFAULT 'queued',
            attempts INTEGER DEFAULT 0,
            last_error TEXT,
            discovered_from TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS records (
            page_url TEXT PRIMARY KEY,
            record_json TEXT NOT NULL,
            saved_at TEXT
        );
        CREATE TABLE IF NOT EXISTS media (
            media_url TEXT PRIMARY KEY,
            page_url TEXT,
            media_type TEXT,
            media_role TEXT,
            local_path TEXT,
            sha256 TEXT,
            status TEXT DEFAULT 'reference_only',
            last_error TEXT,
            updated_at TEXT
        );
        """)
        self.conn.commit()

    def add_page(self, url, kind, source=None):
        self.conn.execute(
            """INSERT OR IGNORE INTO pages(url,kind,status,discovered_from,updated_at)
               VALUES(?,?,'queued',?,?)""",
            (url, kind, source, utc_now())
        )
        self.conn.commit()

    def next_page(self):
        return self.conn.execute(
            "SELECT * FROM pages WHERE status='queued' ORDER BY rowid LIMIT 1"
        ).fetchone()

    def page_status(self, url, status, error=None):
        self.conn.execute(
            """UPDATE pages SET status=?, attempts=attempts+1,
               last_error=?, updated_at=? WHERE url=?""",
            (status, error, utc_now(), url)
        )
        self.conn.commit()

    def save_record(self, url, record):
        self.conn.execute(
            """INSERT OR REPLACE INTO records(page_url,record_json,saved_at)
               VALUES(?,?,?)""",
            (url, json.dumps(record, ensure_ascii=False), utc_now())
        )
        self.conn.commit()

    def add_media(self, media):
        self.conn.execute(
            """INSERT OR IGNORE INTO media
               (media_url,page_url,media_type,media_role,status,updated_at)
               VALUES(?,?,?,?,?,?)""",
            (media["url"], media["page_url"], media["media_type"],
             media["role"], "queued" if media["download_candidate"] else "reference_only",
             utc_now())
        )
        self.conn.commit()

    def queued_media(self):
        return self.conn.execute(
            "SELECT * FROM media WHERE status='queued' ORDER BY rowid"
        ).fetchall()

    def media_update(self, url, status, local_path=None, digest=None, error=None):
        self.conn.execute(
            """UPDATE media SET status=?,local_path=?,sha256=?,last_error=?,updated_at=?
               WHERE media_url=?""",
            (status, local_path, digest, error, utc_now(), url)
        )
        self.conn.commit()

    def export(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        records = [
            json.loads(r["record_json"])
            for r in self.conn.execute("SELECT record_json FROM records ORDER BY page_url")
        ]
        with (output_dir / "vsl_metadata.jsonl").open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # CSV keeps nested media arrays as JSON strings.
        if records:
            keys = sorted({k for r in records for k in r.keys()})
            with (output_dir / "vsl_metadata.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in records:
                    writer.writerow({
                        k: json.dumps(row.get(k), ensure_ascii=False)
                        if isinstance(row.get(k), (list, dict)) else row.get(k, "")
                        for k in keys
                    })

        with (output_dir / "media_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
            rows = self.conn.execute("SELECT * FROM media ORDER BY page_url,media_url").fetchall()
            cols = [d[0] for d in self.conn.execute("SELECT * FROM media LIMIT 0").description]
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])

        with (output_dir / "crawl_summary.json").open("w", encoding="utf-8") as f:
            summary = {}
            for r in self.conn.execute("SELECT kind,status,COUNT(*) n FROM pages GROUP BY kind,status"):
                summary[f"{r['kind']}:{r['status']}"] = r["n"]
            summary["records"] = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            summary["media_items"] = self.conn.execute("SELECT COUNT(*) FROM media").fetchone()[0]
            json.dump(summary, f, ensure_ascii=False, indent=2)

class VslCrawler:
    def __init__(self, args):
        self.args = args
        self.base = args.base.rstrip("/") + "/"
        self.host = urlparse(self.base).netloc.lower()
        self.output = Path(args.output).resolve()
        self.media_dir = self.output / "media"
        self.output.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.output / "crawler.sqlite3")
        self.session = build_session(args.user_agent)
        self.last_request_at = 0.0
        self.robot_parser = RobotFileParser()
        self.robots_ready = False
        self.allowed_external_media_hosts = set(h.lower() for h in args.allowed_media_host)

        logging.basicConfig(
            filename=self.output / "crawler.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8"
        )
        self.log = logging.getLogger(BOT_NAME)

    def request(self, url, stream=False):
        delay = max(0.0, self.args.delay - (time.time() - self.last_request_at))
        if delay:
            time.sleep(delay)
        self.last_request_at = time.time()
        return self.session.get(url, timeout=(15, self.args.timeout), stream=stream)

    def check_robots(self):
        robots_url = urljoin(self.base, "/robots.txt")
        try:
            response = self.session.get(robots_url, timeout=20)
            if response.status_code == 404:
                logging.warning("robots.txt returned 404; no published rules found.")
                self.robots_ready = True
                return
            response.raise_for_status()
            self.robot_parser.parse(response.text.splitlines())
            self.robots_ready = True
            logging.info("Loaded robots.txt from %s", robots_url)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot verify robots.txt ({exc}). Stop by default; retry later or contact site owner."
            )

    def can_fetch(self, url):
        if not self.robots_ready:
            self.check_robots()
        return self.robot_parser.can_fetch(self.args.user_agent, url)

    def in_scope(self, url):
        p = urlparse(url)
        return p.scheme in ("http", "https") and p.netloc.lower() == self.host

    def relevant_kind(self, url):
        path = urlparse(url).path.lower()
        if path.startswith("/phan-loai/"):
            return "category"
        if path.startswith("/tu-ngu/"):
            return "vocabulary"
        if path.startswith("/cau/"):
            return "phrase"
        if path.startswith("/bai-hat/"):
            return "song"
        if any(path.startswith(x) for x in ("/chu-de/", "/bang-chu-cai/", "/a-z/", "/hoc-tieng-anh/")):
            return "index"
        return None

    def seed(self):
        self.store.add_page(canonical_url(self.base), "homepage", "seed")
        if not self.can_fetch(self.base):
            raise RuntimeError("robots.txt disallows crawling the homepage.")
        response = self.request(self.base)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.select("a[href]"):
            url = canonical_url(urljoin(self.base, a.get("href", "")))
            kind = self.relevant_kind(url)
            if kind and self.in_scope(url):
                self.store.add_page(url, kind, self.base)
        self.log.info("Seeded homepage and linked relevant pages.")

    def extract_media(self, soup, page_url):
        found = []
        seen = set()

        def add(raw_url, role, alt=None):
            if not raw_url:
                return
            url = canonical_url(urljoin(page_url, raw_url))
            if url in seen:
                return
            seen.add(url)
            p = urlparse(url)
            ext = Path(p.path).suffix.lower()
            media_type = MEDIA_EXTENSIONS.get(ext)
            is_iframe = role == "iframe"
            if not media_type and not is_iframe:
                # Vimeo/YouTube player links are references, not direct media files.
                if any(host in p.netloc.lower() for host in ("vimeo.com", "youtube.com", "youtu.be")):
                    media_type = "video_reference"
                else:
                    return
            direct = bool(media_type in ("video", "image", "audio"))
            same_host = p.netloc.lower() == self.host
            allowed_host = same_host or p.netloc.lower() in self.allowed_external_media_hosts
            found.append({
                "url": url,
                "page_url": page_url,
                "media_type": media_type or "embedded_reference",
                "role": role,
                "alt": alt,
                "download_candidate": bool(direct and allowed_host)
            })

        for tag in soup.select("video[src], audio[src], source[src]"):
            add(tag.get("src"), tag.name)
        for tag in soup.select("video source[src], audio source[src]"):
            add(tag.get("src"), "source")
        for tag in soup.select("img[src], img[data-src], img[data-original]"):
            src = tag.get("src") or tag.get("data-src") or tag.get("data-original")
            add(src, "image", tag.get("alt"))
            srcset = tag.get("srcset")
            if srcset:
                for candidate in srcset.split(","):
                    item = candidate.strip().split(" ")[0]
                    add(item, "image_srcset", tag.get("alt"))
        for tag in soup.select("iframe[src]"):
            add(tag.get("src"), "iframe")
        for tag in soup.select("a[href]"):
            href = tag.get("href", "")
            p = urlparse(urljoin(page_url, href))
            if Path(p.path).suffix.lower() in MEDIA_EXTENSIONS:
                add(href, "media_link")
            elif any(h in p.netloc.lower() for h in ("vimeo.com", "youtube.com")):
                add(href, "video_platform_link")
        return found

    def extract_record(self, soup, url, kind):
        h1 = soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(" ", strip=True) if soup.title else "")
        text = soup.get_text(" ", strip=True)
        # Preserve link text + URL to support manual verification and provenance.
        links = []
        for a in soup.select("a[href]"):
            href = canonical_url(urljoin(url, a.get("href", "")))
            label = a.get_text(" ", strip=True)
            if label and href:
                links.append({"text": label, "url": href})

        categories, regions = [], []
        metadata_anchors = (
            soup.select('div.mb-4 > div.flex.flex-wrap.gap-2 > a[href]')
            if kind == "vocabulary" else soup.select("a[href]")
        )
        for a in metadata_anchors:
            label = a.get_text(" ", strip=True)
            href = urljoin(url, a.get("href", ""))
            path = urlparse(href).path.rstrip("/")
            if path.startswith("/vung-mien/") and label and label not in regions:
                regions.append(label)
            if self.relevant_kind(href) == "category" and label:
                categories.append(label)

        description = ""
        # Site pages commonly label the signing instructions this way.
        for heading in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            label = heading.get_text(" ", strip=True).lower()
            if "cách làm ký hiệu" in label or "mô tả" in label:
                parent_text = heading.parent.get_text(" ", strip=True) if heading.parent else ""
                description = parent_text[:2000]
                break

        return {
            "source": self.base,
            "page_url": url,
            "page_type": kind,
            "title": title,
            "gloss_candidate": title,
            "categories": sorted(set(categories)),
            "regions": regions,
            "signing_description_candidate": description,
            "page_text": text[:12000],
            "links": links,
            "crawled_at": utc_now()
        }

    def process_page(self, row):
        url, kind = row["url"], row["kind"]
        if not self.can_fetch(url):
            self.store.page_status(url, "blocked_robots", "Disallowed by robots.txt")
            self.log.warning("robots.txt disallowed %s", url)
            return

        try:
            response = self.request(url)
            if response.status_code in (401, 403, 429):
                self.store.page_status(url, "restricted", f"HTTP {response.status_code}")
                self.log.warning("Restricted page %s: HTTP %s", url, response.status_code)
                return
            response.raise_for_status()
            ctype = response.headers.get("Content-Type", "").lower()
            if "html" not in ctype:
                self.store.page_status(url, "skipped_non_html", ctype)
                return

            soup = BeautifulSoup(response.text, "html.parser")
            record = self.extract_record(soup, url, kind)
            media = self.extract_media(soup, url)
            record["media"] = media
            self.store.save_record(url, record)
            for item in media:
                self.store.add_media(item)

            discovered = 0
            for a in soup.select("a[href]"):
                target = canonical_url(urljoin(url, a.get("href", "")))
                if not self.in_scope(target):
                    continue
                target_kind = self.relevant_kind(target)
                if target_kind:
                    self.store.add_page(target, target_kind, url)
                    discovered += 1

            self.store.page_status(url, "done")
            self.log.info("Processed %s (%s), media=%d, discovered=%d",
                          url, kind, len(media), discovered)

        except Exception as exc:
            self.store.page_status(url, "error", str(exc))
            self.log.exception("Failed page %s", url)

    def crawl(self):
        self.check_robots()
        if self.args.retry_errors:
            self.store.conn.execute(
                "UPDATE pages SET status='queued', last_error=NULL WHERE status='error'"
            )
            self.store.conn.execute(
                "UPDATE media SET status='queued', last_error=NULL WHERE status='error'"
            )
            self.store.conn.commit()
            self.log.info("Re-queued failed pages and media for retry.")
        self.seed()
        while True:
            row = self.store.next_page()
            if row is None:
                break
            self.process_page(row)
            if self.args.max_pages and self.store.conn.execute(
                "SELECT COUNT(*) FROM pages WHERE status IN ('done','error','restricted','blocked_robots','skipped_non_html')"
            ).fetchone()[0] >= self.args.max_pages:
                self.log.warning("Reached max-pages limit.")
                break
        self.store.export(self.output)

    def download_media(self):
        if not self.args.download_media:
            return
        if not self.args.confirm_media_rights:
            raise RuntimeError(
                "Media download requires --confirm-media-rights after you have verified permission/licensing."
            )

        for row in tqdm(self.store.queued_media(), desc="Downloading direct media"):
            url = row["media_url"]
            p = urlparse(url)
            if not (p.netloc.lower() == self.host or p.netloc.lower() in self.allowed_external_media_hosts):
                self.store.media_update(url, "skipped_external_host", error="Host not allowlisted")
                continue
            if not self.can_fetch(url):
                self.store.media_update(url, "blocked_robots", error="Disallowed by robots.txt")
                continue

            try:
                response = self.request(url, stream=True)
                if response.status_code in (401, 403, 429):
                    self.store.media_update(url, "restricted", error=f"HTTP {response.status_code}")
                    continue
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";")[0].lower()
                if not content_type.startswith(("video/", "image/", "audio/")):
                    self.store.media_update(url, "not_direct_media", error=f"Content-Type: {content_type}")
                    continue

                ext = Path(p.path).suffix.lower()
                if ext not in MEDIA_EXTENSIONS:
                    ext = mimetypes.guess_extension(content_type) or ".bin"
                tmp = self.media_dir / "tmp"
                tmp.mkdir(parents=True, exist_ok=True)
                temp_path = tmp / (hashlib.sha256(url.encode()).hexdigest() + ".part")
                h = hashlib.sha256()
                total = 0
                max_bytes = self.args.max_media_mb * 1024 * 1024

                with temp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > max_bytes:
                            raise ValueError(f"Media exceeds max size ({self.args.max_media_mb} MB)")
                        h.update(chunk)
                        f.write(chunk)

                digest = h.hexdigest()
                kind = row["media_type"]
                folder = self.media_dir / kind
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{digest[:20]}{ext}"
                if target.exists():
                    temp_path.unlink(missing_ok=True)
                else:
                    temp_path.replace(target)
                self.store.media_update(url, "downloaded", str(target.relative_to(self.output)), digest)
                self.log.info("Downloaded %s -> %s", url, target)
            except Exception as exc:
                self.store.media_update(url, "error", error=str(exc))
                self.log.exception("Media download failed: %s", url)

        self.store.export(self.output)

def parse_args():
    p = argparse.ArgumentParser(description="Crawl public VSL dictionary metadata and permitted direct media.")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--output", default="VSL_Dataset")
    p.add_argument("--delay", type=float, default=2.0, help="Minimum delay between requests (seconds).")
    p.add_argument("--timeout", type=int, default=45)
    p.add_argument("--max-pages", type=int, default=0, help="0 = no page cap; use a small value for testing.")
    p.add_argument("--retry-errors", action="store_true",
                   help="Re-queue pages/media previously marked error and retry them.")
    p.add_argument("--user-agent", default=f"{BOT_NAME}/1.0 (academic research; contact site owner before bulk collection)")
    p.add_argument("--download-media", action="store_true", help="Download direct media files only.")
    p.add_argument("--confirm-media-rights", action="store_true",
                   help="Confirm you have permission/licensing to download and use the media.")
    p.add_argument("--allowed-media-host", action="append", default=[],
                   help="Additional media host explicitly approved for direct downloads; repeatable.")
    p.add_argument("--max-media-mb", type=int, default=250)
    return p.parse_args()

def main():
    args = parse_args()
    crawler = VslCrawler(args)
    try:
        crawler.crawl()
        crawler.download_media()
        print(f"\nFinished. Output: {crawler.output}")
        print("Resume by running the same command again; completed pages are retained in SQLite.")
        print("Review crawler.log, crawl_summary.json and media_manifest.csv.")
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved in SQLite; rerun the same command to resume.")
        crawler.store.export(crawler.output)
        sys.exit(130)
    except Exception as exc:
        logging.exception("Fatal crawler error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        print("No progress is discarded. Fix the issue and rerun to resume.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
