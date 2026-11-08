"""
create_sentence_corpus.py
Constructs a comprehensive Vietnamese Sign Language (VSL) sentence corpus
from crawled VSL phrase metadata, QIPEDC dictionary definitions, and verified
VSL gloss-sequence-to-Vietnamese sentence alignments.
"""

import json
import re
from pathlib import Path
import pandas as pd

VSL_METADATA_PATH = Path("VSL_Dataset/vsl_metadata.csv")
QIPEDC_METADATA_PATH = Path("QIPEDC_Dataset/qipedc_metadata.csv")
OUTPUT_CSV = Path("VSL_Dataset/sentence_corpus.csv")
OUTPUT_JSON = Path("VSL_Dataset/sentence_corpus.json")

# Verified VSL gloss sequence to natural Vietnamese sentence alignments
# Spanning VSL grammar patterns: SVO, Topic-Comment (OSV), Negation, Questions, Greetings, Aspect
CORE_VSL_SENTENCE_PAIRS = [
    # Greetings & Social Expressions
    {
        "gloss_sequence": ["chúc mừng", "buổi sáng", "bà"],
        "vietnamese_sentence": "Chúc mừng buổi sáng bà!",
        "pattern": "greeting",
        "category": "Giao tiếp hàng ngày"
    },
    {
        "gloss_sequence": ["chúc mừng", "năm mới", "gia đình"],
        "vietnamese_sentence": "Chúc mừng năm mới cả gia đình!",
        "pattern": "greeting",
        "category": "Lễ hội"
    },
    {
        "gloss_sequence": ["cảm ơn", "bà", "giúp đỡ"],
        "vietnamese_sentence": "Cảm ơn bà đã giúp đỡ tôi.",
        "pattern": "social_formula",
        "category": "Giao tiếp hàng ngày"
    },
    # Declarative SVO & Activities
    {
        "gloss_sequence": ["bà", "đi", "cửa hàng", "mua", "bánh mì"],
        "vietnamese_sentence": "Bà đi cửa hàng mua bánh mì.",
        "pattern": "svo",
        "category": "Đời sống"
    },
    {
        "gloss_sequence": ["em trai", "học giỏi", "thường xuyên"],
        "vietnamese_sentence": "Em trai thường xuyên học giỏi.",
        "pattern": "adverb_modifier",
        "category": "Giáo dục"
    },
    {
        "gloss_sequence": ["mẹ", "dùng", "dầu ăn", "nấu ăn"],
        "vietnamese_sentence": "Mẹ dùng dầu ăn để nấu ăn.",
        "pattern": "instrumental",
        "category": "Ẩm thực"
    },
    {
        "gloss_sequence": ["chúng tôi", "họp", "buổi sáng"],
        "vietnamese_sentence": "Chúng tôi họp vào buổi sáng.",
        "pattern": "time_aspect",
        "category": "Công việc"
    },
    {
        "gloss_sequence": ["bà", "kể", "kinh nghiệm", "quý báu"],
        "vietnamese_sentence": "Bà kể lại những kinh nghiệm quý báu.",
        "pattern": "svo",
        "category": "Gia đình"
    },
    {
        "gloss_sequence": ["em trai", "tò mò", "hỏi", "bà"],
        "vietnamese_sentence": "Em trai tò mò hỏi bà.",
        "pattern": "svo",
        "category": "Gia đình"
    },
    # Topic-Comment (OSV in VSL -> SVO in Vietnamese)
    {
        "gloss_sequence": ["bánh mì", "này", "tôi", "thích", "ăn"],
        "vietnamese_sentence": "Tôi thích ăn ổ bánh mì này.",
        "pattern": "topic_comment",
        "category": "Ẩm thực"
    },
    {
        "gloss_sequence": ["dây điện", "đó", "nguy hiểm", "đừng", "chạm"],
        "vietnamese_sentence": "Dây điện đó rất nguy hiểm, đừng chạm vào.",
        "pattern": "topic_comment",
        "category": "An toàn"
    },
    # Negation (VSL: Verb + Không -> Vietnamese: Không + Verb)
    {
        "gloss_sequence": ["tôi", "hiểu", "không"],
        "vietnamese_sentence": "Tôi không hiểu.",
        "pattern": "negation",
        "category": "Giao tiếp cơ bản"
    },
    {
        "gloss_sequence": ["em trai", "hối hận", "không"],
        "vietnamese_sentence": "Em trai không hối hận.",
        "pattern": "negation",
        "category": "Tâm lý"
    },
    {
        "gloss_sequence": ["con chuột", "yếu", "không"],
        "vietnamese_sentence": "Con chuột không hề yếu.",
        "pattern": "negation",
        "category": "Động vật"
    },
    # Questions (VSL: Question word/marker at end)
    {
        "gloss_sequence": ["bà", "xem", "gì"],
        "vietnamese_sentence": "Bà đang xem gì thế?",
        "pattern": "wh_question",
        "category": "Câu hỏi"
    },
    {
        "gloss_sequence": ["cửa hàng", "ở đâu"],
        "vietnamese_sentence": "Cửa hàng ở đâu vậy?",
        "pattern": "wh_question",
        "category": "Câu hỏi"
    },
    {
        "gloss_sequence": ["ai", "giúp đỡ", "em trai"],
        "vietnamese_sentence": "Ai đã giúp đỡ em trai?",
        "pattern": "wh_question",
        "category": "Câu hỏi"
    },
    {
        "gloss_sequence": ["bạn", "hiểu", "phải không"],
        "vietnamese_sentence": "Bạn có hiểu không?",
        "pattern": "yes_no_question",
        "category": "Câu hỏi"
    },
    {
        "gloss_sequence": ["con cọp", "yếu", "phải không"],
        "vietnamese_sentence": "Con cọp có yếu không?",
        "pattern": "yes_no_question",
        "category": "Câu hỏi"
    },
    # Descriptive & Adjectival (VSL: Noun + Adj -> Vietnamese: Noun + [là/rất] + Adj)
    {
        "gloss_sequence": ["màn", "màu trắng", "rất", "đẹp"],
        "vietnamese_sentence": "Tấm màn màu trắng rất đẹp.",
        "pattern": "adjectival_predicate",
        "category": "Đồ vật"
    },
    {
        "gloss_sequence": ["mùa xuân", "hoa", "nở", "nhiều"],
        "vietnamese_sentence": "Vào mùa xuân, hoa nở rất nhiều.",
        "pattern": "time_aspect",
        "category": "Thiên nhiên"
    },
    {
        "gloss_sequence": ["con cọp", "mạnh mẽ", "con chuột", "yếu"],
        "vietnamese_sentence": "Con cọp rất mạnh mẽ, còn con chuột thì yếu.",
        "pattern": "comparative",
        "category": "Động vật"
    },
    {
        "gloss_sequence": ["con muỗi", "truyền", "bệnh", "nguy hiểm"],
        "vietnamese_sentence": "Con muỗi truyền bệnh rất nguy hiểm.",
        "pattern": "svo",
        "category": "Sức khỏe"
    },
    {
        "gloss_sequence": ["con nuôi", "yêu quý", "cha mẹ"],
        "vietnamese_sentence": "Người con nuôi rất yêu quý cha mẹ.",
        "pattern": "svo",
        "category": "Gia đình"
    },
]


def extract_phrases_from_vsl(metadata_path: Path):
    """Extracts phrases crawled under /cau/ in VSL metadata."""
    if not metadata_path.exists():
        return []

    print(f"Reading VSL metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path, dtype=str, low_memory=False).fillna("")
    phrase_rows = df[df["page_type"] == "phrase"]

    extracted = []
    seen = set()

    for _, r in phrase_rows.iterrows():
        title = r.get("title", "")
        # Remove site branding
        m = re.search(r"Hướng dẫn thực hiện ký hiệu cho câu:\s*(.+)$", title)
        sentence = m.group(1).strip() if m else ""

        if not sentence:
            gc = r.get("gloss_candidate", "")
            m2 = re.search(r"Hướng dẫn thực hiện ký hiệu cho câu:\s*(.+)$", gc)
            sentence = m2.group(1).strip() if m2 else gc.strip()

        # Clean sentence
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if sentence and sentence not in seen and len(sentence) > 3:
            seen.add(sentence)
            # Derive approximate gloss sequence (tokenized content words)
            tokens = [w.lower() for w in re.findall(r"\b\w+\b", sentence)]
            extracted.append({
                "gloss_sequence": tokens,
                "vietnamese_sentence": sentence,
                "pattern": "extracted_phrase",
                "category": r.get("categories", "Câu thông dụng"),
                "source": "tudienngonngukyhieu"
            })

    # Also extract popular sentences from page texts
    homepage = df[df["page_type"] == "homepage"]
    if not homepage.empty:
        text = homepage.iloc[0].get("page_text", "")
        m_pop = re.search(r"Câu phổ biến\s*(.+?)(?:Designed and coded|$)", text, re.DOTALL)
        if m_pop:
            pop_text = m_pop.group(1)
            pop_lines = [line.strip("- *• \t\r") for line in pop_text.split("\n") if line.strip()]
            for pl in pop_lines:
                if len(pl) > 8 and pl not in seen:
                    seen.add(pl)
                    tokens = [w.lower() for w in re.findall(r"\b\w+\b", pl)]
                    extracted.append({
                        "gloss_sequence": tokens,
                        "vietnamese_sentence": pl,
                        "pattern": "extracted_popular_sentence",
                        "category": "Câu phổ biến",
                        "source": "tudienngonngukyhieu"
                    })

    print(f"Extracted {len(extracted)} phrase sentences from VSL metadata.")
    return extracted


def extract_definitions_from_qipedc(qipedc_path: Path):
    """Extracts concise illustrative definition sentences from QIPEDC."""
    if not qipedc_path.exists():
        return []

    print(f"Reading QIPEDC metadata from {qipedc_path}...")
    df = pd.read_csv(qipedc_path, dtype=str, low_memory=False).fillna("")
    extracted = []
    seen = set()

    for _, r in df.iterrows():
        gloss = r.get("gloss_original", "").strip().lower()
        desc = r.get("description", "").strip()
        if not desc or len(desc) < 10 or desc in seen:
            continue

        # Split into short sentences
        sub_sentences = re.split(r"[;\.\n]+", desc)
        for s in sub_sentences:
            s_clean = s.strip()
            if 10 <= len(s_clean) <= 120 and s_clean not in seen:
                seen.add(s_clean)
                tokens = [w.lower() for w in re.findall(r"\b\w+\b", s_clean)]
                extracted.append({
                    "gloss_sequence": [gloss] + tokens[:4],
                    "vietnamese_sentence": s_clean.capitalize() + ("." if not s_clean.endswith((".", "?", "!")) else ""),
                    "pattern": "definition",
                    "category": "Từ điển định nghĩa",
                    "source": "QIPEDC"
                })
                if len(extracted) >= 200:  # Cap at high quality sample
                    break

    print(f"Extracted {len(extracted)} illustrative definitions from QIPEDC.")
    return extracted


def main():
    print("=== Building Sentence Corpus for VSL Grammar Engine ===")

    all_records = []

    # 1. Add core VSL sentence pairs (gold standard for grammar engine)
    for item in CORE_VSL_SENTENCE_PAIRS:
        record = dict(item)
        record["source"] = "VSL_GOLD_ALIGNMENT"
        all_records.append(record)

    # 2. Add extracted phrase sentences
    vsl_phrases = extract_phrases_from_vsl(VSL_METADATA_PATH)
    all_records.extend(vsl_phrases)

    # 3. Add concise QIPEDC illustrative definitions
    qipedc_defs = extract_definitions_from_qipedc(QIPEDC_METADATA_PATH)
    all_records.extend(qipedc_defs)

    print(f"Total corpus records: {len(all_records)}")

    # Format dataframe
    df_corpus = pd.DataFrame(all_records)
    # Convert gloss_sequence list to comma-separated string for CSV
    df_corpus["gloss_sequence_str"] = df_corpus["gloss_sequence"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else str(g)
    )

    df_corpus.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved CSV corpus to {OUTPUT_CSV}")

    # Save full JSON with structured lists
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON corpus to {OUTPUT_JSON}")

    print("\nSample Corpus Alignments:")
    for sample in all_records[:5]:
        print(f"  Gloss Seq : {sample['gloss_sequence']}")
        print(f"  Vietnamese: {sample['vietnamese_sentence']}")
        print(f"  Pattern   : {sample.get('pattern', '')}\n")


if __name__ == "__main__":
    main()
