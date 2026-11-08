import cv2
import pandas as pd
from pathlib import Path
import math

BASE = Path("VSL_Dataset")
MANIFEST = BASE / "pilot_sample_manifest.csv"
OUT = BASE / "pilot_visual_audit"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST).fillna("")

def get_path(row):
    for col in ["local_path", "media_local_path", "file_path"]:
        if col in row and str(row[col]).strip():
            p = Path(str(row[col]))
            if p.exists():
                return p
    return None

def make_tile(path, gloss, source, sample_id):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for ratio in [0.2, 0.5, 0.8]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * ratio)))
        ok, frame = cap.read()
        if ok:
            frame = cv2.resize(frame, (320, 180))
            frames.append(frame)

    cap.release()

    if not frames:
        return None

    strip = cv2.hconcat(frames)
    cv2.putText(
        strip, f"{gloss} | {source} | {sample_id}",
        (8, 170), cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 255, 0), 1, cv2.LINE_AA
    )
    return strip

audit_rows = []

for gloss, group in df.groupby("gloss"):
    tiles = []

    for _, row in group.iterrows():
        path = get_path(row)

        if path is None:
            audit_rows.append({
                "sample_id": row.get("sample_id", ""),
                "gloss": gloss,
                "source_id": row.get("source_id", ""),
                "visual_status": "missing_path",
            })
            continue

        tile = make_tile(
            path,
            gloss,
            row.get("source_id", ""),
            row.get("sample_id", "")
        )

        status = "ok" if tile is not None else "decode_failed"

        audit_rows.append({
            "sample_id": row.get("sample_id", ""),
            "gloss": gloss,
            "source_id": row.get("source_id", ""),
            "visual_status": status,
            "local_path": str(path),
            "visual_quality": "",
            "label_correct": "",
            "review_notes": "",
        })

        if tile is not None:
            tiles.append(tile)

    if tiles:
        cols = 2
        rows = math.ceil(len(tiles) / cols)
        blank = 255 * (tiles[0].shape[0], tiles[0].shape[1], 3)
        canvas = cv2.copyMakeBorder(
            cv2.vconcat([
                cv2.hconcat(
                    tiles[i:i+cols] +
                    [blank] * (cols - len(tiles[i:i+cols]))
                )
                for i in range(0, len(tiles), cols)
            ]),
            10, 10, 10, 10,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )

        safe = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in gloss
        )
        cv2.imwrite(str(OUT / f"{safe}.jpg"), canvas)

audit = pd.DataFrame(audit_rows)
audit.to_csv(
    OUT / "pilot_visual_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Glosses:", df["gloss"].nunique())
print("Videos:", len(df))
print("Visual status:")
print(audit["visual_status"].value_counts())
print("Saved to:", OUT)
