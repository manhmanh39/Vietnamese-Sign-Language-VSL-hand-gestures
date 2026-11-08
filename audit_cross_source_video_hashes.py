
import cv2
import numpy as np
import pandas as pd
import hashlib
import subprocess
import sys
import re
import json
from pathlib import Path
from itertools import product

# ============================================================
# CONFIG
# ============================================================

INPUT = Path("VSL_Dataset/top30_gloss_video_audit.csv")
VIDEO_DIR = Path("VSL_Dataset/top30_downloaded_videos")
HASH_OUT = Path("VSL_Dataset/top30_video_hashes.csv")
PAIR_OUT = Path("VSL_Dataset/cross_source_video_pairs.csv")

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

RETRIES = 2
DOWNLOAD_TIMEOUT = 240

POSITIONS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]

df = pd.read_csv(INPUT, dtype=str).fillna("")

required = {
    "sample_id",
    "source_id",
    "source_sample_id",
    "gloss_original",
    "video_url",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns in input: {sorted(missing)}")

if "source_page_url" not in df.columns:
    df["source_page_url"] = ""

# ============================================================
# HELPERS
# ============================================================

def clean_id(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def phash(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32))

    dct = cv2.dct(np.float32(gray))
    block = dct[:8, :8].flatten()

    median = np.median(block[1:])
    bits = block > median

    value = 0

    for bit in bits:
        value = (value << 1) | int(bit)

    return value


def hamming(a, b):
    return (int(a) ^ int(b)).bit_count()


def validate_video(path):
    """Check whether OpenCV can actually decode the video."""

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        cap.release()
        return False

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0 or fps <= 0:
        cap.release()
        return False

    ret, frame = cap.read()
    cap.release()

    return bool(ret and frame is not None)


def extract_hashes(path):
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        return [], []

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if frame_count <= 0 or fps <= 0:
        cap.release()
        return [], []

    duration = frame_count / fps

    full_hashes = []
    center_hashes = []

    for pos in POSITIONS:
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            duration * pos * 1000
        )

        ok, frame = cap.read()

        if not ok:
            continue

        h, w = frame.shape[:2]

        full_hashes.append(phash(frame))

        crop = frame[
            int(h * 0.05):int(h * 0.95),
            int(w * 0.15):int(w * 0.85)
        ]

        if crop.size > 0:
            center_hashes.append(phash(crop))

    cap.release()

    return full_hashes, center_hashes


def serialize_hashes(values):
    return ";".join(f"{v:x}" for v in values)


def parse_hashes(value):
    if not isinstance(value, str) or not value.strip():
        return []

    return [
        int(x, 16)
        for x in value.split(";")
        if x.strip()
    ]


def temporal_distance(a, b):
    if not a or not b:
        return None

    d1 = [min(hamming(x, y) for y in b) for x in a]
    d2 = [min(hamming(y, x) for x in a) for y in b]

    return float(np.mean(d1 + d2))


def get_download_url(row):
    """
    TUDIEN:
        Use embedding page URL.

    QIPEDC:
        Use direct video URL.
    """

    source = row["source_id"]

    if source == "TUDIEN":
        return row.get("source_page_url", "").strip()

    if source == "QIPEDC":
        return row["video_url"].strip()

    return row["video_url"].strip()


def find_existing_video(filename):
    candidates = [
        p for p in VIDEO_DIR.glob(filename + ".*")
        if p.is_file()
        and not p.name.endswith(".part")
        and p.suffix.lower() in {
            ".mp4", ".mkv", ".webm", ".mov"
        }
    ]

    for path in candidates:
        if validate_video(path):
            return path

    return None


def download_video(row, filename):
    source = row["source_id"]
    download_url = get_download_url(row)

    if not download_url:
        return None, "Missing download URL"

    last_error = ""

    for attempt in range(1, RETRIES + 1):

        print(f"    Download attempt {attempt}/{RETRIES}")

        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",

            "--no-warnings",
            "--no-progress",
            "--no-playlist",

            "--retries", "2",
            "--fragment-retries", "2",
            "--file-access-retries", "2",
            "--socket-timeout", "30",

            "-f", "bestvideo[height<=720]/bestvideo/best",

            "-o",
            str(VIDEO_DIR / (filename + ".%(ext)s")),

            download_url
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=DOWNLOAD_TIMEOUT
            )

            video_path = find_existing_video(filename)

            if video_path:
                return video_path, ""

            last_error = (
                result.stderr[-3000:]
                if result.stderr
                else f"yt-dlp return code: {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            last_error = "Download timeout"

        except Exception as e:
            last_error = str(e)

    return None, last_error


def save_hash_records(records):
    pd.DataFrame(records).to_csv(
        HASH_OUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# LOAD PREVIOUS RESULTS FOR RESUME
# ============================================================

cached = {}

if HASH_OUT.exists():

    old = pd.read_csv(
        HASH_OUT,
        dtype=str
    ).fillna("")

    if "sample_id" in old.columns:

        for _, row in old.iterrows():

            if (
                row.get("hash_status") == "ok"
                and row.get("local_path")
                and row.get("sha256")
                and Path(row["local_path"]).is_file()
            ):
                cached[row["sample_id"]] = row.to_dict()

print("Previous completed records:", len(cached))


# ============================================================
# DOWNLOAD + HASH
# ============================================================

records = []

for i, (_, row) in enumerate(df.iterrows(), start=1):

    sample_id = row["sample_id"]
    source = row["source_id"]
    gloss = row["gloss_original"]

    filename = clean_id(sample_id)

    print(
        f"\n[{i}/{len(df)}] "
        f"{source} | {gloss} | {sample_id}"
    )

    # --------------------------------------------------------
    # Reuse previously completed hash result
    # --------------------------------------------------------

    if sample_id in cached:

        print("    Reusing completed hash result")

        item = cached[sample_id]
        records.append(item)

        continue

    # --------------------------------------------------------
    # Find existing valid video
    # --------------------------------------------------------

    video_path = find_existing_video(filename)
    download_error = ""

    if video_path:

        print("    Existing valid video found")

    else:

        print("    Video not found. Downloading...")

        video_path, download_error = download_video(
            row,
            filename
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    item = {
        "sample_id": sample_id,
        "source_id": source,
        "source_sample_id": row["source_sample_id"],
        "gloss_original": gloss,

        "source_page_url": row.get("source_page_url", ""),
        "video_url": row["video_url"],

        "download_url": get_download_url(row),

        "local_path": str(video_path) if video_path else "",

        "download_status": (
            "downloaded_or_reused"
            if video_path else "failed"
        ),

        "download_error": download_error,

        "sha256": "",
        "full_phashes": "",
        "center_phashes": "",

        "hash_status": "pending"
    }

    # --------------------------------------------------------
    # Hash
    # --------------------------------------------------------

    if video_path:

        try:

            item["sha256"] = sha256_file(video_path)

            full, center = extract_hashes(video_path)

            if not full or not center:
                raise ValueError(
                    "Could not extract sufficient video frames"
                )

            item["full_phashes"] = serialize_hashes(full)
            item["center_phashes"] = serialize_hashes(center)

            item["hash_status"] = "ok"

            print("    SHA256:", item["sha256"][:16] + "...")
            print("    Frames hashed:", len(full))

        except Exception as e:

            item["hash_status"] = "failed"
            item["download_error"] = str(e)

            print("    Hash error:", e)

    else:

        item["hash_status"] = "failed"
        print("    Download failed")

    records.append(item)

    # Save after every video for interruption recovery
    save_hash_records(records)


# ============================================================
# CROSS-SOURCE COMPARISON
# ============================================================

hash_df = pd.DataFrame(records)

pairs = []

for gloss, group in hash_df.groupby(
    hash_df["gloss_original"].str.strip().str.lower()
):

    tudien = group[group["source_id"] == "TUDIEN"]
    qipedc = group[group["source_id"] == "QIPEDC"]

    for (_, a), (_, b) in product(
        tudien.iterrows(),
        qipedc.iterrows()
    ):

        ha = parse_hashes(a["full_phashes"])
        hb = parse_hashes(b["full_phashes"])

        ca = parse_hashes(a["center_phashes"])
        cb = parse_hashes(b["center_phashes"])

        exact = (
            bool(a["sha256"])
            and bool(b["sha256"])
            and a["sha256"] == b["sha256"]
        )

        pairs.append({

            "gloss": gloss,

            "TUDIEN_sample_id": a["sample_id"],
            "QIPEDC_sample_id": b["sample_id"],

            "TUDIEN_hash_status": a["hash_status"],
            "QIPEDC_hash_status": b["hash_status"],

            "exact_sha256_match": exact,

            "full_phash_mean_distance":
                temporal_distance(ha, hb),

            "center_phash_mean_distance":
                temporal_distance(ca, cb)
        })


pair_df = pd.DataFrame(pairs)

if not pair_df.empty:

    pair_df = pair_df.sort_values(
        [
            "exact_sha256_match",
            "center_phash_mean_distance"
        ],
        ascending=[False, True],
        na_position="last"
    )

pair_df.to_csv(
    PAIR_OUT,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========== SUMMARY ==========")

print("Video records:", len(hash_df))

print(
    "Downloaded/reused:",
    (hash_df["download_status"] == "downloaded_or_reused").sum()
)

print(
    "Failed:",
    (hash_df["download_status"] == "failed").sum()
)

print(
    "SHA256 computed:",
    hash_df["sha256"].ne("").sum()
)

print(
    "Hash completed:",
    (hash_df["hash_status"] == "ok").sum()
)

print("Cross-source pairs:", len(pair_df))

print(
    "Exact SHA256 matches:",
    pair_df["exact_sha256_match"].sum()
    if len(pair_df) else 0
)

print("\nBy source:")

print(
    hash_df.groupby(
        ["source_id", "download_status"]
    ).size()
)

print("\nSaved:", HASH_OUT)
print("Saved:", PAIR_OUT)