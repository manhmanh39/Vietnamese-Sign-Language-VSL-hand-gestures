import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_fps(value):
    try:
        a, b = value.split("/")
        return round(float(a) / float(b), 3) if float(b) else None
    except Exception:
        return None


def run_cmd(cmd, timeout):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )


def audit_one(row, root, decode_timeout):
    sample_id = str(row["sample_id"])
    source_id = str(row["source_id"])
    raw_path = row.get("media_local_path", "")

    result = {
        "sample_id": sample_id,
        "source_id": source_id,
        "media_path": raw_path,
        "integrity_status": "unknown",
        "file_exists": False,
        "file_size_mb": None,
        "sha256_actual": None,
        "sha256_expected": row.get("media_sha256", ""),
        "sha256_match": None,
        "duration_sec": None,
        "expected_duration_sec": row.get("media_duration_sec", None),
        "width": None,
        "height": None,
        "fps": None,
        "video_codec": None,
        "audio_codec": None,
        "format_name": None,
        "ffprobe_status": "not_run",
        "decode_status": "not_run",
        "error": "",
    }

    if not raw_path or pd.isna(raw_path):
        result["integrity_status"] = "missing_path"
        result["error"] = "media_local_path is empty"
        return result

    path = Path(str(raw_path))
    if not path.is_absolute():
        path = root / path

    path = path.resolve()

    if not path.is_file():
        result["integrity_status"] = "missing_file"
        result["error"] = f"File not found: {path}"
        return result

    result["file_exists"] = True
    result["file_size_mb"] = round(path.stat().st_size / (1024 ** 2), 3)

    if path.stat().st_size == 0:
        result["integrity_status"] = "empty_file"
        result["error"] = "File size is zero"
        return result

    # 1. Probe metadata
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration,size,format_name:"
        "stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate",
        "-of", "json",
        str(path)
    ]

    try:
        probe = run_cmd(probe_cmd, timeout=60)

        if probe.returncode != 0:
            result["ffprobe_status"] = "failed"
            result["integrity_status"] = "probe_failed"
            result["error"] = probe.stderr[-2000:]
            return result

        info = json.loads(probe.stdout)
        fmt = info.get("format", {})
        streams = info.get("streams", [])

        result["ffprobe_status"] = "ok"
        result["format_name"] = fmt.get("format_name")

        try:
            result["duration_sec"] = round(float(fmt["duration"]), 3)
        except (KeyError, TypeError, ValueError):
            pass

        video_streams = [
            s for s in streams if s.get("codec_type") == "video"
        ]
        audio_streams = [
            s for s in streams if s.get("codec_type") == "audio"
        ]

        if not video_streams:
            result["integrity_status"] = "no_video_stream"
            result["error"] = "No video stream found"
            return result

        video = video_streams[0]
        result["video_codec"] = video.get("codec_name")
        result["width"] = video.get("width")
        result["height"] = video.get("height")
        result["fps"] = parse_fps(
            video.get("avg_frame_rate", "0/0")
        ) or parse_fps(video.get("r_frame_rate", "0/0"))

        if audio_streams:
            result["audio_codec"] = audio_streams[0].get("codec_name")

    except subprocess.TimeoutExpired:
        result["ffprobe_status"] = "timeout"
        result["integrity_status"] = "probe_timeout"
        result["error"] = "ffprobe timed out"
        return result
    except Exception as e:
        result["integrity_status"] = "probe_error"
        result["error"] = str(e)
        return result

    # 2. Calculate SHA256
    try:
        actual_hash = sha256_file(path)
        result["sha256_actual"] = actual_hash

        expected_hash = result["sha256_expected"]
        if pd.notna(expected_hash) and str(expected_hash).strip():
            result["sha256_match"] = (
                actual_hash.lower() == str(expected_hash).strip().lower()
            )
    except Exception as e:
        result["integrity_status"] = "hash_error"
        result["error"] = str(e)
        return result

    # 3. Decode the entire video
    decode_cmd = [
        "ffmpeg", "-v", "error", "-xerror",
        "-i", str(path),
        "-map", "0:v:0",
        "-f", "null", "-"
    ]

    try:
        decode = run_cmd(decode_cmd, timeout=decode_timeout)

        if decode.returncode != 0:
            result["decode_status"] = "failed"
            result["integrity_status"] = "decode_failed"
            result["error"] = decode.stderr[-2000:]
            return result

        result["decode_status"] = "ok"

    except subprocess.TimeoutExpired:
        result["decode_status"] = "timeout"
        result["integrity_status"] = "decode_timeout"
        result["error"] = f"Full decode exceeded {decode_timeout}s"
        return result
    except Exception as e:
        result["decode_status"] = "error"
        result["integrity_status"] = "decode_error"
        result["error"] = str(e)
        return result

    # 4. Hash comparison
    if result["sha256_match"] is False:
        result["integrity_status"] = "hash_mismatch"
    else:
        result["integrity_status"] = "valid"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="VSL_Dataset/master_manifest_downloaded.csv"
    )
    parser.add_argument(
        "--output",
        default="VSL_Dataset/media_integrity_audit.csv"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--decode-timeout", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    manifest_path = root / args.manifest
    output_path = root / args.output

    df = pd.read_csv(manifest_path, low_memory=False)

    # Exclude duplicate records
    eligible = df[
        ~df["media_download_status"].eq("skipped_duplicate")
    ].copy()

    print("Total manifest records:", len(df))
    print("Records to audit:", len(eligible))
    print("Excluded duplicates:", len(df) - len(eligible))

    # Resume from existing audit
    previous = {}
    if output_path.exists() and not args.force:
        old = pd.read_csv(output_path, low_memory=False)
        if "sample_id" in old.columns:
            previous = {
                str(r["sample_id"]): r.to_dict()
                for _, r in old.iterrows()
            }

    pending = []
    results = list(previous.values()) if previous else []

    for _, row in eligible.iterrows():
        sid = str(row["sample_id"])
        if sid in previous:
            continue
        pending.append(row.to_dict())

    print("Already audited:", len(previous))
    print("Remaining:", len(pending))
    print("Workers:", args.workers)

    completed = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_one, row, root, args.decode_timeout
            ): row["sample_id"]
            for row in pending
        }

        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                results.append({
                    "sample_id": str(futures[future]),
                    "integrity_status": "script_error",
                    "error": str(e)
                })

            completed += 1

            if completed % 50 == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)
                elapsed = time.time() - start
                print(
                    f"[{completed}/{len(pending)}] "
                    f"elapsed={elapsed/60:.1f} min"
                )

    audit = pd.DataFrame(results)
    audit.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("MEDIA INTEGRITY AUDIT SUMMARY")
    print("=" * 60)
    print("Audited:", len(audit))
    print("\nIntegrity status:")
    print(audit["integrity_status"].value_counts(dropna=False))

    if "source_id" in audit.columns:
        print("\nBy source:")
        print(pd.crosstab(
            audit["source_id"],
            audit["integrity_status"],
            margins=True
        ))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
