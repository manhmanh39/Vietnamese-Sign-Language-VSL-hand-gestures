import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(".")
DATA = ROOT / "VSL_Dataset"
INPUT = DATA / "master_manifest_normalized.csv"
OUTPUT = DATA / "master_manifest_downloaded.csv"
LOG = DATA / "video_download_log.csv"

VIDEO_DIR = DATA / "videos"
EXISTING_DIRS = [
    DATA / "top30_downloaded_videos",
    DATA / "test_download",
]

LOG_FIELDS = [
    "sample_id", "source_id", "status", "local_path",
    "sha256", "duration_sec", "error", "updated_at"
]


def now():
    return datetime.now().isoformat(timespec="seconds")


def safe_name(value):
    return str(value).strip().replace(":", "_").replace("/", "_")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("FFprobe validation failed")

    duration = float(result.stdout.strip().splitlines()[0])
    if duration <= 0:
        raise RuntimeError("Invalid video duration")

    return duration


def existing_candidates(row):
    source = row["source_id"]
    video_id = str(row.get("video_id", "")).strip()
    source_id = str(row.get("source_sample_id", "")).strip()
    sample_id = str(row["sample_id"]).strip()

    names = {f"{safe_name(sample_id)}.mp4"}

    if source == "TUDIEN":
        if video_id:
            names.update({
                f"VSL-{video_id}.mp4",
                f"{video_id}.mp4",
                f"test_{video_id}.mp4",
            })

    elif source == "QIPEDC":
        qid = source_id
        if qid.startswith("QIPEDC:"):
            qid = qid.split(":", 1)[1]

        if qid:
            names.update({
                f"QIPEDC_{qid}.mp4",
                f"{qid}.mp4",
                f"QIPEDC:{qid}.mp4",
            })

    # Also check the official output location.
    output_path = (
        VIDEO_DIR / source / f"{safe_name(sample_id)}.mp4"
    )
    if output_path.exists():
        yield output_path

    for folder in EXISTING_DIRS:
        for name in names:
            candidate = folder / name
            if candidate.exists():
                yield candidate


def download_qipedc(url, output_path):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise RuntimeError(
                f"Expected video but received {content_type}"
            )

        with open(output_path, "wb") as f:
            shutil.copyfileobj(response, f)


def download_tudien(page_url, output_path):
    with tempfile.TemporaryDirectory(
        prefix="vsl_yt_",
        dir=str(output_path.parent)
    ) as temp:
        temp_dir = Path(temp)

        command = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--retries", "3",
            "--fragment-retries", "3",
            "--file-access-retries", "3",
            "--socket-timeout", "30",
            "--format",
            "bestvideo[height<=720]/bestvideo/best",
            "--merge-output-format", "mp4",
            "--output", str(temp_dir / "%(id)s.%(ext)s"),
            page_url,
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode != 0:
            message = (result.stderr or result.stdout)[-1500:]
            raise RuntimeError(f"yt-dlp failed: {message}")

        candidates = [
            p for p in temp_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            and not p.name.endswith(".part")
        ]

        if not candidates:
            raise RuntimeError("yt-dlp completed but no video file was found")

        # Prefer the largest output if multiple files were produced.
        downloaded = max(candidates, key=lambda p: p.stat().st_size)

        if downloaded.suffix.lower() == ".mp4":
            shutil.move(str(downloaded), str(output_path))
        else:
            remux = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(downloaded),
                    "-c", "copy", "-movflags", "+faststart",
                    str(output_path)
                ],
                capture_output=True,
                text=True,
                timeout=300
            )
            if remux.returncode != 0:
                raise RuntimeError("FFmpeg MP4 remux failed")


def process_row(row):
    sample_id = str(row["sample_id"]).strip()
    source = str(row["source_id"]).strip()

    target_dir = VIDEO_DIR / source
    target_dir.mkdir(parents=True, exist_ok=True)

    target = target_dir / f"{safe_name(sample_id)}.mp4"
    temp_target = target.with_suffix(".download.mp4")

    try:
        # Reuse existing valid video before downloading.
        for candidate in existing_candidates(row):
            try:
                duration = probe_video(candidate)
                digest = sha256_file(candidate)

                return {
                    "sample_id": sample_id,
                    "source_id": source,
                    "status": "reused_existing",
                    "local_path": str(candidate),
                    "sha256": digest,
                    "duration_sec": duration,
                    "error": "",
                    "updated_at": now(),
                }
            except Exception:
                continue

        if source == "TUDIEN":
            # Vimeo embeds require the source page as the entry point.
            page_url = str(row.get("source_page_url", "")).strip()
            if not page_url:
                raise RuntimeError("Missing TUDIEN source_page_url")

            download_tudien(page_url, temp_target)

        elif source == "QIPEDC":
            url = str(row.get("video_url", "")).strip()
            if not url:
                raise RuntimeError("Missing QIPEDC video_url")

            download_qipedc(url, temp_target)

        else:
            raise RuntimeError(f"Unknown source: {source}")

        duration = probe_video(temp_target)
        digest = sha256_file(temp_target)

        os.replace(temp_target, target)

        return {
            "sample_id": sample_id,
            "source_id": source,
            "status": "downloaded",
            "local_path": str(target),
            "sha256": digest,
            "duration_sec": duration,
            "error": "",
            "updated_at": now(),
        }

    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()

        return {
            "sample_id": sample_id,
            "source_id": source,
            "status": "failed",
            "local_path": "",
            "sha256": "",
            "duration_sec": "",
            "error": str(e)[:2000],
            "updated_at": now(),
        }


def atomic_save(df, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, path)


def append_log(record):
    exists = LOG.exists()

    with open(LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in LOG_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT, dtype=str).fillna("")

    required = {
        "sample_id", "source_id", "video_url",
        "video_id", "source_sample_id",
        "source_page_url", "training_eligibility",
        "duplicate_status"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    for col in [
        "media_local_path", "media_sha256",
        "media_download_status", "media_last_error",
        "media_updated_at", "media_duration_sec"
    ]:
        if col not in df.columns:
            df[col] = ""

    # Do not download records explicitly excluded as duplicates.
    duplicate_mask = (
        df["training_eligibility"].eq("excluded_duplicate")
        | df["duplicate_status"].eq("duplicate_candidate")
    )

    df.loc[duplicate_mask, "media_download_status"] = "skipped_duplicate"

    # Normalize legacy/empty statuses before processing.
    # Existing valid files will subsequently become reused_existing.
    df.loc[~duplicate_mask, "media_download_status"] = "pending"

    # Existing valid files and completed downloads are reused.
    tasks = []
    reused = 0
    skipped = int(duplicate_mask.sum())

    for idx, row in df.iterrows():
        if duplicate_mask.loc[idx]:
            continue

        result = None

        # Check manifest path first.
        current_path = str(row.get("media_local_path", "")).strip()
        if current_path and Path(current_path).is_file():
            try:
                duration = probe_video(current_path)
                result = {
                    "sample_id": row["sample_id"],
                    "source_id": row["source_id"],
                    "status": "reused_existing",
                    "local_path": current_path,
                    "sha256": sha256_file(current_path),
                    "duration_sec": duration,
                    "error": "",
                    "updated_at": now(),
                }
            except Exception:
                result = None

        if result is None:
            # Check known folders and final output folder.
            for candidate in existing_candidates(row):
                try:
                    duration = probe_video(candidate)
                    result = {
                        "sample_id": row["sample_id"],
                        "source_id": row["source_id"],
                        "status": "reused_existing",
                        "local_path": str(candidate),
                        "sha256": sha256_file(candidate),
                        "duration_sec": duration,
                        "error": "",
                        "updated_at": now(),
                    }
                    break
                except Exception:
                    continue

        if result:
            df.at[idx, "media_local_path"] = result["local_path"]
            df.at[idx, "media_sha256"] = result["sha256"]
            df.at[idx, "media_download_status"] = result["status"]
            df.at[idx, "media_last_error"] = ""
            df.at[idx, "media_updated_at"] = result["updated_at"]
            df.at[idx, "media_duration_sec"] = str(result["duration_sec"])
            reused += 1
        else:
            tasks.append((idx, row.to_dict()))

    if args.limit > 0:
        tasks = tasks[:args.limit]

    atomic_save(df, OUTPUT)

    print("=" * 70)
    print("VSL MEDIA DOWNLOADER")
    print("=" * 70)
    print("Manifest records:", len(df))
    print("Duplicate exclusions:", skipped)
    print("Existing videos reused:", reused)
    print("Videos to download:", len(tasks))
    print("Workers:", args.workers)
    print("Output:", OUTPUT)
    print("Log:", LOG)
    print("=" * 70)

    completed = 0
    downloaded = 0
    failed = 0

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(process_row, row): idx
                for idx, row in tasks
            }

            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()

                df.at[idx, "media_local_path"] = result["local_path"]
                df.at[idx, "media_sha256"] = result["sha256"]
                df.at[idx, "media_download_status"] = result["status"]
                df.at[idx, "media_last_error"] = result["error"]
                df.at[idx, "media_updated_at"] = result["updated_at"]
                df.at[idx, "media_duration_sec"] = str(
                    result["duration_sec"]
                )

                append_log(result)

                completed += 1
                if result["status"] == "failed":
                    failed += 1
                else:
                    downloaded += 1

                if completed % 10 == 0:
                    atomic_save(df, OUTPUT)
                    print(
                        f"[{completed}/{len(tasks)}] "
                        f"success={downloaded} failed={failed}"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted. Saving completed results...")

    finally:
        atomic_save(df, OUTPUT)

    print("\n" + "=" * 70)
    print("DOWNLOAD SESSION SUMMARY")
    print("=" * 70)
    print("Completed this session:", completed)
    print("Downloaded/reused:", downloaded)
    print("Failed:", failed)
    print("Reused before session:", reused)
    print("Excluded duplicates:", skipped)

    print("\nCurrent status distribution:")
    print(df["media_download_status"].value_counts(dropna=False).to_string())

    print("\nSaved manifest:", OUTPUT.resolve())
    print("Saved log:", LOG.resolve())


if __name__ == "__main__":
    main()
