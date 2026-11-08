
import argparse
import csv
import os
import re
import tempfile
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import requests


INPUT_CSV = Path("QIPEDC_Dataset/qipedc_metadata.csv")
OUTPUT_CSV = Path("QIPEDC_Dataset/qipedc_overlay_audit.csv")
REVIEW_DIR = Path("QIPEDC_Dataset/overlay_review")

BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://qipedc.moet.gov.vn/dictionary",
}


def normalize_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", text)


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray, None, fx=3, fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, threshold = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold


def ocr_crop(crop):
    image = preprocess_crop(crop)

    try:
        text = pytesseract.image_to_string(
            image,
            lang="vie+eng",
            config="--psm 6"
        )
    except pytesseract.TesseractError:
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6"
        )

    return text.strip()


def extract_region(text):
    normalized = normalize_text(text)

    regions = {
        "mien bac": "Miền Bắc",
        "mien nam": "Miền Nam",
        "mien trung": "Miền Trung",
        "toan quoc": "Toàn quốc",
        "ca nuoc": "Toàn quốc",
    }

    for keyword, region in regions.items():
        if keyword in normalized:
            return region

    return "unknown"


def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0 or fps <= 0:
        cap.release()
        return None

    duration = total_frames / fps

    # Sample two frames to reduce the chance of
    # missing an overlay due to compression or transitions.
    timestamps = [duration * 0.5, duration * 0.8]

    results = []

    for timestamp in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        success, frame = cap.read()

        if not success:
            continue

        height, width = frame.shape[:2]

        # Top-right: gloss
        gloss_crop = frame[
            int(height * 0.02):int(height * 0.25),
            int(width * 0.60):int(width * 0.99)
        ]

        # Bottom-right: region label
        region_crop = frame[
            int(height * 0.78):int(height * 0.98),
            int(width * 0.55):int(width * 0.99)
        ]

        results.append({
            "gloss": ocr_crop(gloss_crop),
            "region": ocr_crop(region_crop),
            "gloss_crop": gloss_crop,
            "region_crop": region_crop,
        })

    cap.release()
    return results


def download_video(url, output_path):
    response = requests.get(
        url,
        headers=BASE_HEADERS,
        stream=True,
        timeout=60
    )

    if response.status_code != 200:
        response.close()
        return response.status_code

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)

    response.close()
    return 200


def process_record(row, index):
    video_url = row.get("video_url", "")
    sample_id = row.get("source_sample_id", str(index))

    result = {
        **row,
        "overlay_gloss": "",
        "overlay_region": "",
        "region_claimed": "unknown",
        "gloss_match": "unknown",
        "overlay_status": "failed",
        "video_http_status": "",
    }

    if not video_url:
        result["overlay_status"] = "missing_video_url"
        return result

    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / "sample.mp4"

        try:
            status = download_video(video_url, video_path)
            result["video_http_status"] = status

            if status != 200:
                result["overlay_status"] = "download_failed"
                return result

            frames = extract_frames(video_path)

            if not frames:
                result["overlay_status"] = "video_unreadable"
                return result

            gloss_text = " ".join(
                frame["gloss"] for frame in frames
            )
            region_text = " ".join(
                frame["region"] for frame in frames
            )

            result["overlay_gloss"] = gloss_text
            result["overlay_region"] = region_text

            region = extract_region(region_text)
            result["region_claimed"] = region

            expected = normalize_text(
                row.get("gloss_original", "")
            )
            observed = normalize_text(gloss_text)

            if expected and expected in observed:
                result["gloss_match"] = "yes"
            elif observed:
                result["gloss_match"] = "review"
            else:
                result["gloss_match"] = "unreadable"

            if region != "unknown":
                result["overlay_status"] = "region_detected"
            else:
                result["overlay_status"] = "region_not_detected"

            # Save crops for manual review.
            if region == "unknown" or result["gloss_match"] != "yes":
                REVIEW_DIR.mkdir(parents=True, exist_ok=True)

                for frame_index, frame in enumerate(frames):
                    cv2.imwrite(
                        str(REVIEW_DIR / f"{sample_id}_{frame_index}_gloss.jpg"),
                        frame["gloss_crop"]
                    )
                    cv2.imwrite(
                        str(REVIEW_DIR / f"{sample_id}_{frame_index}_region.jpg"),
                        frame["region_crop"]
                    )

        except Exception as error:
            result["overlay_status"] = (
                f"error: {type(error).__name__}: {error}"
            )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of videos to process; 0 means all"
    )
    args = parser.parse_args()

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    with open(INPUT_CSV, encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))

    if args.limit > 0:
        records = records[:args.limit]

    print(f"Videos to process: {len(records)}")

    output_rows = []

    for index, row in enumerate(records, 1):
        sample_id = row.get("source_sample_id", index)

        print(
            f"[{index}/{len(records)}] "
            f"{sample_id} | {row.get('gloss_original', '')}"
        )

        result = process_record(row, index)
        output_rows.append(result)

        print(
            f"  Region: {result['region_claimed']} | "
            f"Gloss match: {result['gloss_match']} | "
            f"Status: {result['overlay_status']}"
        )

        time.sleep(0.2)

    if output_rows:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(output_rows[0].keys())

        with open(
            OUTPUT_CSV,
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )
            writer.writeheader()
            writer.writerows(output_rows)

    print("\n[DONE]")
    print("Output:", OUTPUT_CSV)
    print("Review crops:", REVIEW_DIR)


if __name__ == "__main__":
    main()
