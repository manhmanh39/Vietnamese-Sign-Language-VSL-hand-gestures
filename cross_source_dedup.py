import argparse
import csv
from curses import raw
import hashlib
import json
import math
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PAIR_FIELDS = [
    "sample_id_a", "source_id_a", "gloss_a", "video_url_a",
    "sample_id_b", "source_id_b", "gloss_b", "video_url_b",
    "match_type", "confidence", "sha256_a", "sha256_b",
    "phash_frames_matched", "phash_mean_hamming", "metadata_evidence",
    "duplicate_review_status",
]
GROUP_FIELDS = [
    "duplicate_group_id", "sample_id", "source_id", "gloss_candidate",
    "video_id", "video_url", "local_path", "video_sha256",
    "duplicate_type", "duplicate_review_status",
]
FRAME_FRACTIONS = (0.25, 0.5, 0.75)
P_HASH_SIZE = 32
P_HASH_LOW_FREQ = 8


def text_value(row, *names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def number_value(value):
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def int_value(value):
    number = number_value(value)
    return int(number) if number is not None and number > 0 else None


def media_url_key(url):
    if not url:
        return ""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    path_segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if host == "vimeo.com" or host.endswith(".vimeo.com"):
        numeric = next((segment for segment in reversed(path_segments) if segment.isdigit()), "")
        if numeric:
            return f"vimeo:{numeric}"
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if video_id:
            return f"youtube:{video_id}"
    if host in {"youtu.be", "www.youtu.be"} and path_segments:
        return f"youtube:{path_segments[0]}"
    return f"url:{host}{parsed.path}?{parsed.query}"


def read_media_locations(path):
    locations = {}
    if not path:
        return locations
    media_manifest = Path(path).resolve()
    with media_manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            media_url = text_value(row, "media_url", "video_url", "url")
            local_path = text_value(row, "local_path", "file_path", "video_path")
            if not media_url or not local_path:
                continue
            candidate = Path(local_path).expanduser()
            if not candidate.is_absolute():
                candidate = media_manifest.parent / candidate
            locations[media_url_key(media_url)] = candidate.resolve()
    return locations


def resolve_local_path(row, manifest_path, media_locations, video_root):
    raw_path = text_value(row, "local_path", "video_path", "file_path", "download_path")
    if not raw_path:
        raw_path = str(media_locations.get(media_url_key(text_value(row, "video_url", "video", "media_url")), ""))
    if not raw_path:
        return ""
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.is_file() else ""
    roots = [Path(video_root).expanduser()] if video_root else []
    roots.append(Path(manifest_path).resolve().parent)
    for root in roots:
        resolved = (root / candidate).resolve()
        if resolved.is_file():
            return str(resolved)
    return ""


def load_manifest(path, media_locations, video_root, used_sample_ids):
    manifest_path = Path(path).resolve()
    records = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for index, raw in enumerate(csv.DictReader(stream), 1):
            source_id = text_value(raw, "source_id", "source", "dataset") or manifest_path.stem
            video_url = text_value(raw, "video_url", "video", "media_url")
            video_id = text_value(
                raw, "video_id", "source_sample_id",
                "qipedc_media_id", "qipedc_id"
            )

            sample_id = text_value(
                raw, "sample_id", "record_key",
                "qipedc_id", "qipedc_media_id", "source_sample_id"
            )
            if not sample_id:
                token = video_url or text_value(raw, "source_page_url", "detail_url", "url") or str(index)
                sample_id = hashlib.sha256(f"{source_id}|{token}".encode("utf-8")).hexdigest()[:16]
            if sample_id in used_sample_ids:
                sample_id = f"{source_id}:{sample_id}"
            used_sample_ids.add(sample_id)

            local_path = resolve_local_path(raw, manifest_path, media_locations, video_root)
            duration = number_value(text_value(raw, "duration_seconds", "duration", "video_duration"))
            width = int_value(text_value(raw, "width", "video_width"))
            height = int_value(text_value(raw, "height", "video_height"))
            resolution = text_value(raw, "resolution", "video_resolution")
            if resolution and (not width or not height):
                match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", resolution)
                if match:
                    width, height = int(match.group(1)), int(match.group(2))
            file_size = int_value(text_value(raw, "file_size_bytes", "file_size", "size_bytes"))
            if local_path:
                try:
                    file_size = Path(local_path).stat().st_size
                except OSError:
                    local_path = ""
            records.append({
                "sample_id": sample_id,
                "source_id": source_id,
                "gloss": text_value(raw, "canonical_gloss_id", "gloss_candidate", "gloss_original", "gloss", "label"),
                "video_id": video_id,
                "video_url": video_url,
                "source_page_url": text_value(raw, "source_page_url", "detail_url", "page_url", "url"),
                "local_path": local_path,
                "duration": duration,
                "width": width,
                "height": height,
                "file_size": file_size,
                "sha256": "",
                "phash": [],
            })
    return records


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path, ffprobe):
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, text=True, timeout=30)
        data = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None, None, None
    stream = next(iter(data.get("streams", [])), {})
    duration = number_value(data.get("format", {}).get("duration"))
    return duration, int_value(stream.get("width")), int_value(stream.get("height"))


def extract_gray_frame(path, seconds, ffmpeg):
    command = [
        ffmpeg, "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={P_HASH_SIZE}:{P_HASH_SIZE}:flags=area",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    expected = P_HASH_SIZE * P_HASH_SIZE
    return result.stdout if len(result.stdout) == expected else None


def phash_from_gray(frame):
    size = P_HASH_SIZE
    low = P_HASH_LOW_FREQ
    cosines = [
        [math.cos((2 * position + 1) * frequency * math.pi / (2 * size)) for position in range(size)]
        for frequency in range(low)
    ]
    alpha = [math.sqrt(1 / size) if frequency == 0 else math.sqrt(2 / size) for frequency in range(low)]
    rows = [list(frame[start:start + size]) for start in range(0, len(frame), size)]
    horizontal = [
        [sum(rows[y][x] * cosines[u][x] for x in range(size)) * alpha[u] for u in range(low)]
        for y in range(size)
    ]
    coefficients = [
        sum(horizontal[y][u] * cosines[v][y] for y in range(size)) * alpha[v]
        for v in range(low) for u in range(low)
    ]
    values = coefficients[1:]
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]
    result = 0
    for coefficient in values:
        result = (result << 1) | int(coefficient >= median)
    return result


def hash_video_frames(record, ffmpeg, ffprobe):
    if not record["local_path"] or not ffmpeg:
        return []
    duration = record["duration"]
    if not duration and ffprobe:
        duration, width, height = probe_video(record["local_path"], ffprobe)
        record["duration"] = duration
        record["width"] = record["width"] or width
        record["height"] = record["height"] or height
    if not duration or duration <= 0:
        return []
    hashes = []
    for fraction in FRAME_FRACTIONS:
        frame = extract_gray_frame(record["local_path"], duration * fraction, ffmpeg)
        hashes.append(phash_from_gray(frame) if frame else None)
    return hashes


def metadata_evidence(first, second):
    evidence = {}
    duration_a, duration_b = first["duration"], second["duration"]
    if duration_a is not None and duration_b is not None:
        difference = abs(duration_a - duration_b)
        tolerance = max(0.25, max(duration_a, duration_b) * 0.005)
        evidence["duration_difference_seconds"] = round(difference, 3)
        evidence["duration_close"] = difference <= tolerance
    if first["width"] and first["height"] and second["width"] and second["height"]:
        evidence["same_resolution"] = (first["width"], first["height"]) == (second["width"], second["height"])
    size_a, size_b = first["file_size"], second["file_size"]
    if size_a and size_b:
        difference_ratio = abs(size_a - size_b) / max(size_a, size_b)
        evidence["file_size_difference_ratio"] = round(difference_ratio, 5)
        evidence["file_size_close"] = difference_ratio <= 0.01
    has_supporting_metadata = evidence.get("same_resolution") or evidence.get("file_size_close")
    if evidence.get("duration_close") and has_supporting_metadata:
        return evidence
    return {}


def compare_phashes(first, second, threshold):
    left, right = first["phash"], second["phash"]
    distances = [
        (a ^ b).bit_count()
        for a, b in zip(left, right)
        if a is not None and b is not None
    ]
    matched = sum(distance <= threshold for distance in distances)
    if matched >= 2:
        return matched, round(sum(distances) / len(distances), 2)
    return 0, round(sum(distances) / len(distances), 2) if distances else ""


def pair_key(first, second):
    return tuple(sorted((first["sample_id"], second["sample_id"])))


def add_evidence(pair_map, first, second, match_type, confidence, details):
    if first["source_id"] == second["source_id"]:
        return
    key = pair_key(first, second)
    pair = pair_map.setdefault(key, {"a": first, "b": second, "evidence": []})
    pair["evidence"].append({
        "match_type": match_type,
        "confidence": confidence,
        "details": details,
    })


def candidate_priority(match_type):
    return {"sha256_exact": 0, "same_media_url": 1, "visual_candidate": 2, "metadata_candidate": 3}.get(match_type, 9)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Find cross-source duplicate video candidates without downloading media.")
    parser.add_argument("--manifest", action="append", required=True, help="CSV manifest; repeat once per source.")
    parser.add_argument("--media-manifest", action="append", default=[], help="Optional crawler media_manifest.csv; repeatable.")
    parser.add_argument("--video-root", type=Path, help="Root for relative local_path values.")
    parser.add_argument("--output-dir", type=Path, default=Path("VSL_Dataset"))
    parser.add_argument("--phash-distance", type=int, default=8)
    parser.add_argument("--max-phash-pairs", type=int, default=2_000_000)
    args = parser.parse_args()

    media_locations = {}
    for path in args.media_manifest:
        media_locations.update(read_media_locations(path))
    used_ids = set()
    records = []
    records_by_manifest = {}
    for path in args.manifest:
        loaded = load_manifest(path, media_locations, args.video_root, used_ids)
        records.extend(loaded)
        records_by_manifest[Path(path).name] = len(loaded)

    for record in records:
        if record["local_path"]:
            record["sha256"] = sha256_file(record["local_path"])
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    local_records = [record for record in records if record["local_path"]]
    phash_skipped_reason = ""
    if local_records and ffmpeg:
        for record in local_records:
            record["phash"] = hash_video_frames(record, ffmpeg, ffprobe)
    elif local_records:
        phash_skipped_reason = "ffmpeg is not installed; perceptual hashes were not computed."
    else:
        phash_skipped_reason = "No local video files were mapped from the input manifests."

    pair_map = {}
    by_sha = defaultdict(list)
    by_url = defaultdict(list)
    by_source = defaultdict(list)
    for record in records:
        if record["sha256"]:
            by_sha[record["sha256"]].append(record)
        url_key = media_url_key(record["video_url"])
        if url_key:
            by_url[url_key].append(record)
        if record["video_url"] or record["local_path"]:
            by_source[record["source_id"]].append(record)

    for digest, group in by_sha.items():
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                add_evidence(pair_map, first, second, "sha256_exact", 1.0, {"sha256": digest})
    for url_key, group in by_url.items():
        for index, first in enumerate(group):
            for second in group[index + 1:]:
                add_evidence(pair_map, first, second, "same_media_url", 0.98, {"canonical_media_key": url_key})

    has_metadata = any(
        r["duration"] is not None
        or r["width"] is not None
        or r["height"] is not None
        or r["file_size"] is not None
        for r in records
    )

    source_ids = sorted(by_source) if (local_records or has_metadata) else []
    metadata_pairs_checked = 0
    phash_pairs_checked = 0
    phash_pairs_skipped_cap = False
    phash_pair_total = sum(
        len(by_source[first_source]) * len(by_source[second_source])
        for index, first_source in enumerate(source_ids)
        for second_source in source_ids[index + 1:]
    )
    check_phash = bool(local_records and ffmpeg and phash_pair_total <= args.max_phash_pairs)
    phash_pairs_skipped_cap = bool(local_records and ffmpeg and phash_pair_total > args.max_phash_pairs)

    for index, first_source in enumerate(source_ids):
        for second_source in source_ids[index + 1:]:
            for first in by_source[first_source]:
                for second in by_source[second_source]:
                    metadata = metadata_evidence(first, second)
                    if metadata:
                        metadata_pairs_checked += 1
                        add_evidence(pair_map, first, second, "metadata_candidate", 0.35, metadata)
                    if check_phash and first["phash"] and second["phash"]:
                        phash_pairs_checked += 1
                        matched, mean_distance = compare_phashes(first, second, args.phash_distance)
                        if matched:
                            add_evidence(pair_map, first, second, "visual_candidate", 0.8, {
                                "matched_sampled_frames": matched,
                                "mean_hamming_distance": mean_distance,
                                "threshold": args.phash_distance,
                            })

    candidate_rows = []
    for pair in pair_map.values():
        evidence = sorted(pair["evidence"], key=lambda item: candidate_priority(item["match_type"]))
        strongest = evidence[0]
        first, second = pair["a"], pair["b"]
        phash_evidence = next((item["details"] for item in evidence if item["match_type"] == "visual_candidate"), {})
        candidate_rows.append({
            "sample_id_a": first["sample_id"],
            "source_id_a": first["source_id"],
            "gloss_a": first["gloss"],
            "video_url_a": first["video_url"],
            "sample_id_b": second["sample_id"],
            "source_id_b": second["source_id"],
            "gloss_b": second["gloss"],
            "video_url_b": second["video_url"],
            "match_type": strongest["match_type"],
            "confidence": strongest["confidence"],
            "sha256_a": first["sha256"],
            "sha256_b": second["sha256"],
            "phash_frames_matched": phash_evidence.get("matched_sampled_frames", ""),
            "phash_mean_hamming": phash_evidence.get("mean_hamming_distance", ""),
            "metadata_evidence": json.dumps([item["details"] for item in evidence if item["match_type"] == "metadata_candidate"], ensure_ascii=False),
            "duplicate_review_status": "confirmed_exact" if strongest["match_type"] == "sha256_exact" else "pending_manual_review",
        })
    candidate_rows.sort(key=lambda row: (candidate_priority(row["match_type"]), row["source_id_a"], row["source_id_b"], row["sample_id_a"], row["sample_id_b"]))

    exact_groups = defaultdict(list)
    for digest, group in by_sha.items():
        if len({record["source_id"] for record in group}) > 1:
            exact_groups[digest].extend(group)
    group_rows = []
    for digest, group in sorted(exact_groups.items()):
        group_id = f"DUP-SHA-{digest[:12].upper()}"
        for record in group:
            group_rows.append({
                "duplicate_group_id": group_id,
                "sample_id": record["sample_id"],
                "source_id": record["source_id"],
                "gloss_candidate": record["gloss"],
                "video_id": record["video_id"],
                "video_url": record["video_url"],
                "local_path": record["local_path"],
                "video_sha256": digest,
                "duplicate_type": "exact_file",
                "duplicate_review_status": "confirmed_exact",
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cross_source_duplicate_candidates.csv", PAIR_FIELDS, candidate_rows)
    write_csv(args.output_dir / "duplicate_groups.csv", GROUP_FIELDS, group_rows)
    source_counts = defaultdict(lambda: {"records": 0, "video_references": 0, "local_files": 0})
    for record in records:
        stats = source_counts[record["source_id"]]
        stats["records"] += 1
        stats["video_references"] += bool(record["video_url"])
        stats["local_files"] += bool(record["local_path"])
    summary = {
        "input_manifests": records_by_manifest,
        "records_by_source": dict(sorted(source_counts.items())),
        "records_total": len(records),
        "local_video_files": len(local_records),
        "exact_hashes_computed": sum(bool(record["sha256"]) for record in records),
        "phash_records_computed": sum(bool(record["phash"]) for record in records),
        "ffmpeg_available": bool(ffmpeg),
        "ffprobe_available": bool(ffprobe),
        "phash_skipped_reason": phash_skipped_reason,
        "cross_source_metadata_pairs_with_matching_evidence": metadata_pairs_checked,
        "cross_source_phash_pairs_checked": phash_pairs_checked,
        "phash_pairs_skipped_pair_limit": phash_pairs_skipped_cap,
        "duplicate_candidate_pairs": len(candidate_rows),
        "candidate_pairs_by_method": {
            method: sum(row["match_type"] == method for row in candidate_rows)
            for method in ("sha256_exact", "same_media_url", "visual_candidate", "metadata_candidate")
        },
        "confirmed_cross_source_exact_duplicate_groups": len(exact_groups),
        "same_gloss_alone_used_as_duplicate_evidence": False,
        "semantic_duplicate_detection": "manual review required; no semantic or pose-based auto-merge is performed",
        "limitations": [
            "Metadata similarity and pHash only create candidates; they do not prove duplicate content.",
            "Exact SHA-256 groups are confirmed byte-identical files.",
            "Use only local video files that the project is authorized to access and process.",
            "Candidate pairs do not modify or delete source records. Merge confirmed duplicate_group_id values into the master manifest before rerunning the split planner.",
        ],
    }
    (args.output_dir / "dedup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()