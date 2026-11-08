"""
prepare_training_data.py
Prepares and partitions video dataset for VSL sign recognition model training.
Verifies video integrity, extracts video metadata (frames, fps, duration),
and generates stratified Train / Validation / Test splits with class mappings.
"""

import argparse
import json
import random
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm


def probe_video(video_path: str):
    """Probes video using OpenCV to get frame count, fps, resolution and duration."""
    p = Path(video_path)
    if not p.exists() or p.stat().st_size == 0:
        return None

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        return None

    duration = round(frame_count / max(fps, 1.0), 3)
    return {
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def stratified_split(df: pd.DataFrame, label_col: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Splits dataset into stratified train, val, and test partitions."""
    rng = random.Random(seed)
    train_rows, val_rows, test_rows = [], [], []

    for label, group in df.groupby(label_col):
        items = group.to_dict("records")
        rng.shuffle(items)
        n = len(items)

        if n == 1:
            train_rows.extend(items)
        elif n == 2:
            train_rows.append(items[0])
            test_rows.append(items[1])
        elif n == 3:
            train_rows.append(items[0])
            val_rows.append(items[1])
            test_rows.append(items[2])
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            # ensure at least 1 in test
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            n_test = n - n_train - n_val

            train_rows.extend(items[:n_train])
            val_rows.extend(items[n_train:n_train + n_val])
            test_rows.extend(items[n_train + n_val:])

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare VSL training data manifests.")
    parser.add_argument("--mode", choices=["pilot", "top_classes", "full"], default="pilot",
                        help="Dataset mode: 'pilot' uses verified pilot benchmark, 'top_classes' selects top-K glosses.")
    parser.add_argument("--manifest", type=str, default="VSL_Dataset/standardized_glosses.csv",
                        help="Path to standardized glosses manifest.")
    parser.add_argument("--pilot-manifest", type=str, default="VSL_Dataset/pilot_sample_manifest.csv",
                        help="Path to pilot manifest.")
    parser.add_argument("--output-dir", type=str, default="VSL_Dataset/training_data",
                        help="Output directory for training data.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of classes if using top_classes mode.")
    parser.add_argument("--min-samples", type=int, default=6, help="Minimum samples per class.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Preparing VSL Training Data (Mode: {args.mode}) ===")

    records = []
    if args.mode == "pilot" and Path(args.pilot_manifest).exists():
        print(f"Loading pilot manifest from {args.pilot_manifest}...")
        df = pd.read_csv(args.pilot_manifest, dtype=str, low_memory=False).fillna("")
        gloss_col = "gloss" if "gloss" in df.columns else "normalized_gloss"
    else:
        print(f"Loading standardized glosses from {args.manifest}...")
        df = pd.read_csv(args.manifest, dtype=str, low_memory=False).fillna("")
        gloss_col = "normalized_gloss"

    # Filter to only records where video exists
    print("Verifying video file existence and probing video containers...")
    valid_samples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Probing videos"):
        video_path = row.get("media_local_path", "") or row.get("video_path", "")
        if not video_path:
            continue
        p = Path(video_path)
        if not p.exists():
            continue

        probe = probe_video(str(p))
        if probe is None:
            continue

        gloss = row.get(gloss_col, "").strip()
        if not gloss:
            continue

        sample_dict = {
            "sample_id": row.get("sample_id", p.stem),
            "source_id": row.get("source_id", "VSL"),
            "gloss": gloss,
            "region": row.get("region_claimed", row.get("region", "unknown")),
            "video_path": str(p),
            "frame_count": probe["frame_count"],
            "fps": probe["fps"],
            "duration_sec": probe["duration_sec"],
            "width": probe["width"],
            "height": probe["height"],
        }
        valid_samples.append(sample_dict)

    df_valid = pd.DataFrame(valid_samples)
    print(f"Total valid, readable video samples: {len(df_valid)}")

    if args.mode == "top_classes":
        # Select top-K glosses with at least min_samples
        class_counts = df_valid["gloss"].value_counts()
        eligible_classes = class_counts[class_counts >= args.min_samples].head(args.top_k).index.tolist()
        df_valid = df_valid[df_valid["gloss"].isin(eligible_classes)].copy()
        print(f"Filtered to top {len(eligible_classes)} classes with >= {args.min_samples} samples. Total: {len(df_valid)}")

    # Class mappings
    classes = sorted(df_valid["gloss"].unique().tolist())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}

    df_valid["label_idx"] = df_valid["gloss"].map(class_to_idx)

    # Stratified Train / Val / Test Split
    train_df, val_df, test_df = stratified_split(df_valid, label_col="gloss", train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    # Save outputs
    full_path = out_dir / "full_dataset_manifest.csv"
    train_path = out_dir / "train_manifest.csv"
    val_path = out_dir / "val_manifest.csv"
    test_path = out_dir / "test_manifest.csv"
    class_map_path = out_dir / "class_to_idx.json"
    idx_map_path = out_dir / "idx_to_class.json"
    summary_path = out_dir / "dataset_summary.json"

    df_valid.to_csv(full_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    with open(idx_map_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    summary = {
        "mode": args.mode,
        "num_classes": len(classes),
        "total_samples": len(df_valid),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "classes": classes,
        "class_sample_distribution": df_valid["gloss"].value_counts().to_dict(),
        "train_split": len(train_df) / len(df_valid),
        "val_split": len(val_df) / len(df_valid),
        "test_split": len(test_df) / len(df_valid),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n--- Data Preparation Complete ---")
    print(f"Num Classes:   {len(classes)}")
    print(f"Total Samples: {len(df_valid)}")
    print(f"Train Samples: {len(train_df)} ({len(train_df)/len(df_valid)*100:.1f}%)")
    print(f"Val Samples:   {len(val_df)} ({len(val_df)/len(df_valid)*100:.1f}%)")
    print(f"Test Samples:  {len(test_df)} ({len(test_df)/len(df_valid)*100:.1f}%)")
    print(f"Manifests saved to: {out_dir}")


if __name__ == "__main__":
    main()
