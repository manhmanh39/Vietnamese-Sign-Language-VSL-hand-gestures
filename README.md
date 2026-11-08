# VSL Dictionary Crawler

Crawler research starter for `https://tudienngonngukyhieu.com/`.

## What it does

- Starts from the homepage and discovers linked category/index pages.
- Crawls `/phan-loai/` category pages and follows pagination links discovered in HTML.
- Discovers vocabulary detail pages (`/tu-ngu/`) and other relevant internal routes.
- Extracts page title, candidate gloss, category/region labels, text, links, images, video/audio references, and embedded-player URLs.
- Stores progress in SQLite (`crawler.sqlite3`) so interrupted crawls can resume.
- Exports JSONL, CSV, media manifest, summary and logs.
- Deduplicates media by SHA-256 when direct media download is enabled.
- Downloads only direct media files from the same host or explicitly allowlisted hosts.

## Important permission and safety notes

The website's public pages are not, by themselves, proof that bulk downloading or ML training use is licensed.
Before downloading or using media in a model, review the website's terms and copyright notices and obtain permission from the site owner/rightsholders where needed. Respect `robots.txt`, rate limits, and third-party platform terms. This crawler does not bypass login, CAPTCHA, paywalls, hotlink protection, or access controls.

The script checks `robots.txt` and stops if it cannot verify the file (except an HTTP 404). It uses a conservative 2-second delay by default. Do not increase request rate without permission.

Vimeo/YouTube embeds are saved as references; they are not downloaded or extracted from the player. Use official download/export mechanisms only where you have permission.

## Install

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## First run: small test

```bash
python crawl_vsl_dictionary.py --output VSL_Dataset --max-pages 20
```

Check:
- `VSL_Dataset/crawler.log`
- `VSL_Dataset/crawl_summary.json`
- `VSL_Dataset/vsl_metadata.csv`
- `VSL_Dataset/media_manifest.csv`

## Full metadata crawl

After verifying the test output and confirming the website permits the activity:

```bash
python crawl_vsl_dictionary.py --output VSL_Dataset
```

## Download permitted direct media

Only after confirming rights/licensing:

```bash
python crawl_vsl_dictionary.py --output VSL_Dataset \
  --download-media --confirm-media-rights
```

If a separate media host is explicitly authorized, add it:

```bash
python crawl_vsl_dictionary.py --output VSL_Dataset \
  --download-media --confirm-media-rights \
  --allowed-media-host cdn.example.org
```

Do not add Vimeo/YouTube player domains expecting the script to download their videos; player embeds are references, not direct downloadable media.

## Resume

Run exactly the same command again. Pages already marked `done` are not fetched again. Failed pages remain marked `error` for inspection. To automatically re-queue failed pages and failed media, add `--retry-errors`:

```bash
python crawl_vsl_dictionary.py --output VSL_Dataset --retry-errors
``` 

Alternatively, reset only selected records with SQL:

To retry failed media:

```sql
UPDATE media SET status='queued', last_error=NULL WHERE status='error';
```

You can run these statements using a SQLite GUI or:

```bash
python -c "import sqlite3; c=sqlite3.connect('VSL_Dataset/crawler.sqlite3'); c.execute(\"UPDATE pages SET status='queued',last_error=NULL WHERE status='error'\"); c.execute(\"UPDATE media SET status='queued',last_error=NULL WHERE status='error'\"); c.commit()"
```

## Output layout

```text
VSL_Dataset/
├── crawler.sqlite3
├── crawler.log
├── vsl_metadata.jsonl
├── vsl_metadata.csv
├── media_manifest.csv
├── crawl_summary.json
└── media/
    ├── image/
    ├── video/
    ├── audio/
    └── tmp/
```

## Research-quality checks before training

1. Manually verify that every media item matches its gloss. A detail page may contain an incorrectly titled or mismatched embed.
2. Preserve regional variants as separate records; do not merge by Vietnamese word alone.
3. Distinguish a word-level gloss from phrases, songs, and alphabet/fingerspelling pages.
4. Track signer identity only when the source explicitly provides it; never infer signer identity from appearance.
5. Deduplicate exact files by SHA-256, but do not assume visually similar videos are duplicates.
6. Split by signer/source/session where possible. Randomly splitting frames from one video creates severe leakage.
7. Keep the original source URL and license/permission evidence for every sample.
8. The crawler's `gloss_candidate` is only a candidate label and requires quality control.

## Known limitation

The website may render some content dynamically or use third-party embedded players. This script extracts what is present in the returned HTML. If a page loads content only through JavaScript, inspect the page's public network/API documentation or ask the site owner for an authorized data export; do not reverse-engineer or bypass access controls.
