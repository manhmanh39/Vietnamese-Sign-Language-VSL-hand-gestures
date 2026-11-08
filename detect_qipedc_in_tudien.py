#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Detect QIPEDC watermark / region overlays in downloaded TUDIEN videos.

Inputs:
    VSL_Dataset/cross_source_video_pairs.csv
    VSL_Dataset/top30_downloaded_videos/
    QIPEDC_Dataset/qipedc_overlay_audit_normalized.csv

Outputs:
    VSL_Dataset/tudien_qipedc_ocr_audit.csv
    VSL_Dataset/cross_source_reuse_candidates.csv
    VSL_Dataset/tudien_qipedc_ocr_crops/

Usage:
    python detect_qipedc_in_tudien.py
    python detect_qipedc_in_tudien.py --limit 10
    python detect_qipedc_in_tudien.py --force
"""

import argparse
import re
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PAIRS_CSV = BASE_DIR / "VSL_Dataset" / "cross_source_video_pairs.csv"
VIDEO_DIR = BASE_DIR / "VSL_Dataset" / "top30_downloaded_videos"

QIPEDC_AUDIT_CSV = (
    BASE_DIR
    / "QIPEDC_Dataset"
    / "qipedc_overlay_audit_normalized.csv"
)

VIDEO_AUDIT_CSV = BASE_DIR / "VSL_Dataset" / "tudien_qipedc_ocr_audit.csv"
PAIR_OUTPUT_CSV = BASE_DIR / "VSL_Dataset" / "cross_source_reuse_candidates.csv"

CROP_DIR = BASE_DIR / "VSL_Dataset" / "tudien_qipedc_ocr_crops"


# ============================================================
# OCR CONFIGURATION
# ============================================================

OCR_LANG = "vie+eng"

# Sample frames at these percentages of video duration.
FRAME_POSITIONS = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]

# Regions of interest:
# Left: QIPEDC logo / watermark.
# Bottom-right: "Ký hiệu Miền Bắc/Nam/Trung".
#
# Coordinates are relative to frame width/height.
LOGO_ROI = (0.00, 0.00, 0.45, 1.00)
REGION_ROI = (0.52, 0.62, 1.00, 1.00)


# ============================================================
# NORMALIZATION
# ============================================================

def remove_accents(text):
    """Remove Vietnamese diacritics for robust OCR matching."""
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def normalize_text(text):
    text = remove_accents(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_region(text):
    """
    Return one of:
        Miền Bắc
        Miền Trung
        Miền Nam
        unknown
    """
    text = normalize_text(text)

    # Prefer specific region labels.
    if re.search(r"mien\s*(bac|b[a0]c)", text):
        return "Miền Bắc"

    if re.search(r"mien\s*trung", text):
        return "Miền Trung"

    if re.search(r"mien\s*nam", text):
        return "Miền Nam"

    return "unknown"


def normalize_qipedc_id(value):
    value = str(value).strip()

    if value.startswith("QIPEDC:"):
        return value.split(":", 1)[1]

    if value.startswith("QIPEDC_"):
        return value.split("_", 1)[1]

    return value


# ============================================================
# VIDEO / OCR HELPERS
# ============================================================

def get_frame_at_percentage(cap, percentage, total_frames):
    if total_frames <= 0:
        return None

    frame_index = int((total_frames - 1) * percentage)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()

    if not success or frame is None:
        return None

    return frame


def crop_roi(frame, roi):
    height, width = frame.shape[:2]

    x1 = int(roi[0] * width)
    y1 = int(roi[1] * height)
    x2 = int(roi[2] * width)
    y2 = int(roi[3] * height)

    return frame[y1:y2, x1:x2]


def preprocess_for_ocr(image):
    """
    Generate an enlarged grayscale image and a thresholded image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale to help Tesseract read small overlay text.
    gray = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Improve local contrast.
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return gray, threshold


def run_ocr(image):
    """
    Run Tesseract with two page segmentation modes.
    Return combined OCR text.
    """
    gray, threshold = preprocess_for_ocr(image)

    texts = []

    for processed in [gray, threshold]:
        for psm in [6, 11]:
            config = f"--oem 1 --psm {psm}"

            try:
                text = pytesseract.image_to_string(
                    processed,
                    lang=OCR_LANG,
                    config=config
                )
                if text:
                    texts.append(text)
            except Exception as exc:
                print(f"    OCR warning: {exc}")

    return "\n".join(texts)


def detect_qipedc_marker(text):
    """
    Detect OCR-readable QIPEDC text.
    This does not detect a purely graphical logo.
    """
    normalized = normalize_text(text)
    compact = re.sub(r"\s+", "", normalized)

    patterns = [
        "qipedc",
        "qip edc",
        "qipdec",
        "qiped",
    ]

    return any(
        re.sub(r"\s+", "", pattern) in compact
        for pattern in patterns
    )


def detect_video_overlays(video_path, sample_id, save_crops=True):
    """
    OCR one TUDIEN video.

    Returns a dictionary with:
        qipedc_marker_detected
        detected_region
        OCR text
        frame-level evidence
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "video_status": "cannot_open",
            "qipedc_marker_detected": False,
            "detected_region": "unknown",
            "logo_ocr_text": "",
            "region_ocr_text": "",
            "frames_processed": 0,
            "marker_frames": 0,
            "region_frames": 0,
            "evidence_frames": "",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return {
            "sample_id": sample_id,
            "video_path": str(video_path),
            "video_status": "invalid_frame_count",
            "qipedc_marker_detected": False,
            "detected_region": "unknown",
            "logo_ocr_text": "",
            "region_ocr_text": "",
            "frames_processed": 0,
            "marker_frames": 0,
            "region_frames": 0,
            "evidence_frames": "",
        }

    all_logo_text = []
    all_region_text = []

    marker_positions = []
    region_positions = []
    region_votes = []

    frames_processed = 0

    for percentage in FRAME_POSITIONS:
        frame = get_frame_at_percentage(
            cap,
            percentage,
            total_frames
        )

        if frame is None:
            continue

        frames_processed += 1

        logo_crop = crop_roi(frame, LOGO_ROI)
        region_crop = crop_roi(frame, REGION_ROI)

        logo_text = run_ocr(logo_crop)
        region_text = run_ocr(region_crop)

        all_logo_text.append(logo_text)
        all_region_text.append(region_text)

        marker_found = detect_qipedc_marker(logo_text)

        if marker_found:
            marker_positions.append(int(percentage * 100))

        detected_region = normalize_region(region_text)

        if detected_region != "unknown":
            region_positions.append(int(percentage * 100))
            region_votes.append(detected_region)

        # Save evidence crops for manual inspection.
        if save_crops and (
            marker_found or detected_region != "unknown"
        ):
            CROP_DIR.mkdir(parents=True, exist_ok=True)

            tag = f"{sample_id}_{int(percentage * 100)}"

            if marker_found:
                cv2.imwrite(
                    str(CROP_DIR / f"{tag}_logo.png"),
                    logo_crop
                )

            if detected_region != "unknown":
                cv2.imwrite(
                    str(CROP_DIR / f"{tag}_region.png"),
                    region_crop
                )

    cap.release()

    # Majority vote for detected region.
    if region_votes:
        region_counts = pd.Series(region_votes).value_counts()
        detected_region = region_counts.index[0]
    else:
        detected_region = "unknown"

    marker_detected = len(marker_positions) > 0

    return {
        "sample_id": sample_id,
        "video_path": str(video_path),
        "video_status": "processed",
        "qipedc_marker_detected": marker_detected,
        "detected_region": detected_region,
        "logo_ocr_text": " | ".join(all_logo_text),
        "region_ocr_text": " | ".join(all_region_text),
        "frames_processed": frames_processed,
        "marker_frames": len(marker_positions),
        "region_frames": len(region_positions),
        "evidence_frames": ",".join(
            map(str, sorted(set(marker_positions + region_positions)))
        ),
    }


# ============================================================
# LOAD INPUTS
# ============================================================

def load_pairs():
    if not PAIRS_CSV.exists():
        raise FileNotFoundError(f"Missing file: {PAIRS_CSV}")

    df = pd.read_csv(PAIRS_CSV)

    required = [
        "gloss",
        "TUDIEN_sample_id",
        "QIPEDC_sample_id",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing columns in cross_source_video_pairs.csv: {missing}"
        )

    return df


def load_qipedc_audit():
    if not QIPEDC_AUDIT_CSV.exists():
        raise FileNotFoundError(
            f"Missing file: {QIPEDC_AUDIT_CSV}"
        )

    df = pd.read_csv(QIPEDC_AUDIT_CSV)

    id_col = None

    for candidate in [
        "source_sample_id",
        "sample_id",
        "video_id",
        "_id",
    ]:
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col is None:
        raise ValueError(
            "Cannot find QIPEDC ID column in normalized OCR audit."
        )

    region_col = None

    for candidate in [
        "overlay_region_clean",
        "overlay_region",
        "region_clean",
    ]:
        if candidate in df.columns:
            region_col = candidate
            break

    if region_col is None:
        raise ValueError(
            "Cannot find normalized region column in QIPEDC OCR audit."
        )

    df["qipedc_id_normalized"] = (
        df[id_col].astype(str).apply(normalize_qipedc_id)
    )

    df["qipedc_region_normalized"] = (
        df[region_col].fillna("unknown").apply(normalize_region)
    )

    # The source audit may already contain normalized labels.
    # normalize_region handles "Miền Bắc/Nam/Trung" values.
    return df.drop_duplicates("qipedc_id_normalized")


# ============================================================
# PAIR-LEVEL COMPARISON
# ============================================================

def build_pair_results(pairs, video_audit, qipedc_audit):
    video_lookup = (
        video_audit
        .drop_duplicates("sample_id")
        .set_index("sample_id")
        .to_dict("index")
    )

    qipedc_lookup = (
        qipedc_audit
        .drop_duplicates("qipedc_id_normalized")
        .set_index("qipedc_id_normalized")
        .to_dict("index")
    )

    results = []

    for _, pair in pairs.iterrows():
        tudien_id = str(pair["TUDIEN_sample_id"])
        qipedc_id_raw = str(pair["QIPEDC_sample_id"])
        qipedc_id = normalize_qipedc_id(qipedc_id_raw)

        video_info = video_lookup.get(tudien_id, {})
        qipedc_info = qipedc_lookup.get(qipedc_id, {})

        tudien_marker = bool(
            video_info.get("qipedc_marker_detected", False)
        )

        tudien_region = video_info.get(
            "detected_region",
            "unknown"
        )

        qipedc_region = qipedc_info.get(
            "qipedc_region_normalized",
            "unknown"
        )

        if pd.isna(tudien_region):
            tudien_region = "unknown"

        if pd.isna(qipedc_region):
            qipedc_region = "unknown"

        region_match = (
            tudien_region == qipedc_region
            and tudien_region != "unknown"
        )

        video_status = video_info.get(
            "video_status", "not_processed"
        )

        if video_status != "processed":
            status = "not_processed"

        elif tudien_marker and region_match:
            status = "duplicate_candidate"

        elif tudien_marker or region_match:
            status = "review"

        else:
            status = "no_qipedc_evidence"

        results.append({
            "gloss": pair["gloss"],
            "TUDIEN_sample_id": tudien_id,
            "QIPEDC_sample_id": qipedc_id_raw,
            "TUDIEN_video_status": video_info.get(
                "video_status", "not_processed"
            ),
            "QIPEDC_marker_detected": tudien_marker,
            "TUDIEN_region": tudien_region,
            "QIPEDC_region": qipedc_region,
            "region_match": region_match,
            "TUDIEN_marker_frames": video_info.get(
                "marker_frames", 0
            ),
            "TUDIEN_region_frames": video_info.get(
                "region_frames", 0
            ),
            "full_phash_mean_distance": pair.get(
                "full_phash_mean_distance", np.nan
            ),
            "center_phash_mean_distance": pair.get(
                "center_phash_mean_distance", np.nan
            ),
            "exact_sha256_match": pair.get(
                "exact_sha256_match", False
            ),
            "reuse_status": status,
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of unique TUDIEN videos. 0 = all."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess videos even if they exist in previous audit CSV."
    )

    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Do not save OCR evidence crops."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("TUDIEN -> QIPEDC OVERLAY DETECTION")
    print("=" * 70)

    pairs = load_pairs()
    qipedc_audit = load_qipedc_audit()

    print(f"Cross-source pairs: {len(pairs)}")
    print(f"QIPEDC OCR records: {len(qipedc_audit)}")

    # One OCR pass per unique TUDIEN video.
    tudien_ids = (
        pairs["TUDIEN_sample_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print(f"Unique TUDIEN videos to inspect: {len(tudien_ids)}")

    if args.limit > 0:
        tudien_ids = tudien_ids[:args.limit]

    # Resume from existing audit unless --force is specified.
    previous = pd.DataFrame()

    if VIDEO_AUDIT_CSV.exists() and not args.force:
        previous = pd.read_csv(VIDEO_AUDIT_CSV)

        if "sample_id" in previous.columns:
            previous = previous.drop_duplicates(
                "sample_id",
                keep="last"
            )

    previous_lookup = {}

    if not previous.empty:
        previous_lookup = previous.set_index(
            "sample_id"
        ).to_dict("index")

    results = []
    processed_count = 0
    skipped_count = 0
    missing_count = 0

    for index, sample_id in enumerate(tudien_ids, start=1):
        print(f"\n[{index}/{len(tudien_ids)}] {sample_id}")

        # Resume only successfully processed videos.
        if (
            not args.force
            and sample_id in previous_lookup
            and previous_lookup[sample_id].get("video_status") == "processed"
        ):
            print("    Already processed; reusing previous OCR result.")
            results.append(previous_lookup[sample_id])
            skipped_count += 1
            continue

        video_path = VIDEO_DIR / f"{sample_id}.mp4"

        if not video_path.exists():
            print(f"    Missing video: {video_path}")
            missing_count += 1

            results.append({
                "sample_id": sample_id,
                "video_path": str(video_path),
                "video_status": "missing_file",
                "qipedc_marker_detected": False,
                "detected_region": "unknown",
                "logo_ocr_text": "",
                "region_ocr_text": "",
                "frames_processed": 0,
                "marker_frames": 0,
                "region_frames": 0,
                "evidence_frames": "",
            })

        else:
            result = detect_video_overlays(
                video_path=video_path,
                sample_id=sample_id,
                save_crops=not args.no_save_crops,
            )

            results.append(result)
            processed_count += 1

            print(
                f"    QIPEDC marker: "
                f"{result['qipedc_marker_detected']}"
            )
            print(
                f"    Region: {result['detected_region']}"
            )
            print(
                f"    Frames processed: {result['frames_processed']}"
            )

        # Save incrementally after each video.
        current = pd.DataFrame(results)

        if not previous.empty:
            completed_ids = set(current["sample_id"].astype(str))
            remaining_previous = previous[
                ~previous["sample_id"].astype(str).isin(completed_ids)
            ]

            current = pd.concat(
                [remaining_previous, current],
                ignore_index=True
            )

        current.to_csv(
            VIDEO_AUDIT_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    # Load the full audit file so previously processed rows are retained.
    video_audit = pd.read_csv(VIDEO_AUDIT_CSV)

    # Pair-level comparison.
    pair_results = build_pair_results(
        pairs=pairs,
        video_audit=video_audit,
        qipedc_audit=qipedc_audit,
    )

    pair_results.to_csv(
        PAIR_OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Unique TUDIEN videos in current run: {len(tudien_ids)}")
    print(f"Newly processed: {processed_count}")
    print(f"Reused previous results: {skipped_count}")
    print(f"Missing video files: {missing_count}")

    print("\nTUDIEN OCR results:")
    print(
        video_audit["video_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nQIPEDC marker detection:")
    print(
        video_audit["qipedc_marker_detected"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nDetected TUDIEN regions:")
    print(
        video_audit["detected_region"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nPair-level reuse status:")
    print(
        pair_results["reuse_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nRegion comparison:")
    print(
        pd.crosstab(
            pair_results["TUDIEN_region"],
            pair_results["QIPEDC_region"],
            margins=True
        ).to_string()
    )

    print("\nSaved files:")
    print(f"  Video OCR: {VIDEO_AUDIT_CSV}")
    print(f"  Pair audit: {PAIR_OUTPUT_CSV}")
    print(f"  Evidence crops: {CROP_DIR}")

    print("\nDone.")


if __name__ == "__main__":
    main()
