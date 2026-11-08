
import pandas as pd
import re
import unicodedata
from pathlib import Path

INPUT = Path("QIPEDC_Dataset/qipedc_overlay_audit.csv")
OUTPUT = Path("QIPEDC_Dataset/qipedc_overlay_audit_normalized.csv")

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        c for c in value
        if unicodedata.category(c) != "Mn"
    )
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def detect_region(value):
    text = normalize_text(value)

    if not text:
        return "unknown"

    # Bắc
    if re.search(r"mien\s+bac", text):
        return "Miền Bắc"

    # Nam: xử lý một số lỗi OCR phổ biến
    if re.search(r"mien\s+(nam|nem|nom|no[mn])", text):
        return "Miền Nam"

    # Trung: xử lý trường hợp OCR tách chữ
    if re.search(r"mien\s+(trung|t\s*rung)", text):
        return "Miền Trung"

    # Toàn quốc
    if re.search(r"toan\s+quoc|ca\s+nuoc", text):
        return "Toàn quốc"

    return "unknown"


df = pd.read_csv(INPUT)

# Giữ nguyên OCR gốc, không ghi đè dữ liệu cũ
df["overlay_region_raw"] = df["overlay_region"]
df["overlay_region_clean"] = df["overlay_region"].apply(detect_region)

df["region_ocr_status"] = df["overlay_region_clean"].apply(
    lambda x: "detected" if x != "unknown" else "unknown"
)

df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("Total:", len(df))
print("\nNormalized region distribution:")
print(df["overlay_region_clean"].value_counts().to_string())

print("\nDetection rate:")
print(
    f"{(df['overlay_region_clean'] != 'unknown').mean() * 100:.2f}%"
)

print("\nSaved:", OUTPUT)
