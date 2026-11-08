"""
create_lexicon_and_grammar.py
Constructs a comprehensive Part-of-Speech tagged Lexicon and VSL Grammar Ruleset
for converting predicted VSL sign gloss sequences into natural Vietnamese sentences.
"""

import json
from pathlib import Path
import pandas as pd

GLOSSES_PATH = Path("VSL_Dataset/standardized_glosses.csv")
VOCAB_SUMMARY_PATH = Path("VSL_Dataset/gloss_vocabulary_summary.csv")
LEXICON_CSV = Path("VSL_Dataset/lexicon.csv")
LEXICON_JSON = Path("VSL_Dataset/lexicon.json")
GRAMMAR_RULES_TXT = Path("VSL_Dataset/grammar_rules.txt")
GRAMMAR_RULES_JSON = Path("VSL_Dataset/grammar_rules.json")

# Curated high-precision lexical attributes for VSL signs
EXPLICIT_LEXICON = {
    # Pronouns & Kinship / Honorifics
    "bà": {"pos": "PRON", "subcat": "honorific_kinship", "classifier": "", "gender": "female", "gloss_en": "grandmother/lady"},
    "ông": {"pos": "PRON", "subcat": "honorific_kinship", "classifier": "", "gender": "male", "gloss_en": "grandfather/gentleman"},
    "mẹ": {"pos": "PRON", "subcat": "kinship", "classifier": "", "gender": "female", "gloss_en": "mother"},
    "ba": {"pos": "PRON", "subcat": "kinship", "classifier": "", "gender": "male", "gloss_en": "father"},
    "em trai": {"pos": "PRON", "subcat": "kinship", "classifier": "", "gender": "male", "gloss_en": "younger_brother"},
    "chị gái": {"pos": "PRON", "subcat": "kinship", "classifier": "", "gender": "female", "gloss_en": "older_sister"},
    "tôi": {"pos": "PRON", "subcat": "first_person", "classifier": "", "gender": "neutral", "gloss_en": "I/me"},
    "bạn": {"pos": "PRON", "subcat": "second_person", "classifier": "", "gender": "neutral", "gloss_en": "you"},
    "chúng tôi": {"pos": "PRON", "subcat": "first_person_plural", "classifier": "", "gender": "neutral", "gloss_en": "we"},
    "con nuôi": {"pos": "NOUN", "subcat": "kinship", "classifier": "người", "gender": "neutral", "gloss_en": "adopted_child"},
    "cha mẹ": {"pos": "NOUN", "subcat": "kinship", "classifier": "", "gender": "neutral", "gloss_en": "parents"},

    # Nouns & Concrete Entities
    "bánh mì": {"pos": "NOUN", "subcat": "food", "classifier": "ổ", "gloss_en": "bread"},
    "dầu ăn": {"pos": "NOUN", "subcat": "liquid/food", "classifier": "chai", "gloss_en": "cooking_oil"},
    "gạo": {"pos": "NOUN", "subcat": "food", "classifier": "hạt", "gloss_en": "rice"},
    "bánh cuốn": {"pos": "NOUN", "subcat": "food", "classifier": "đĩa", "gloss_en": "steamed_rice_rolls"},
    "hoa": {"pos": "NOUN", "subcat": "plant", "classifier": "bông", "gloss_en": "flower"},
    "lá cây": {"pos": "NOUN", "subcat": "plant", "classifier": "chiếc", "gloss_en": "tree_leaf"},
    "dây điện": {"pos": "NOUN", "subcat": "object", "classifier": "sợi", "gloss_en": "electric_wire"},
    "màn": {"pos": "NOUN", "subcat": "object", "classifier": "tấm", "gloss_en": "curtain/mosquito_net"},
    "bảng đen": {"pos": "NOUN", "subcat": "object", "classifier": "cái", "gloss_en": "blackboard"},
    "cửa hàng": {"pos": "NOUN", "subcat": "location", "classifier": "", "gloss_en": "store/shop"},
    "kinh nghiệm": {"pos": "NOUN", "subcat": "abstract", "classifier": "", "gloss_en": "experience"},
    "con chuột": {"pos": "NOUN", "subcat": "animal", "classifier": "con", "gloss_en": "mouse"},
    "con cọp": {"pos": "NOUN", "subcat": "animal", "classifier": "con", "gloss_en": "tiger"},
    "con muỗi": {"pos": "NOUN", "subcat": "animal", "classifier": "con", "gloss_en": "mosquito"},
    "con ếch": {"pos": "NOUN", "subcat": "animal", "classifier": "con", "gloss_en": "frog"},
    "con người": {"pos": "NOUN", "subcat": "human", "classifier": "", "gloss_en": "human_being"},
    "bệnh": {"pos": "NOUN", "subcat": "abstract", "classifier": "căn", "gloss_en": "illness"},

    # Verbs & Actions
    "chúc mừng": {"pos": "VERB", "subcat": "speech_act", "transitive": True, "gloss_en": "congratulate/greet"},
    "cảm ơn": {"pos": "VERB", "subcat": "speech_act", "transitive": True, "gloss_en": "thank"},
    "học giỏi": {"pos": "VERB", "subcat": "predicate", "transitive": False, "gloss_en": "study_well"},
    "giúp đỡ": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "help/assist"},
    "hiểu": {"pos": "VERB", "subcat": "cognition", "transitive": True, "gloss_en": "understand"},
    "hỏi": {"pos": "VERB", "subcat": "communication", "transitive": True, "gloss_en": "ask"},
    "hối hận": {"pos": "VERB", "subcat": "emotion", "transitive": False, "gloss_en": "regret"},
    "họp": {"pos": "VERB", "subcat": "action", "transitive": False, "gloss_en": "meet/attend_meeting"},
    "xem": {"pos": "VERB", "subcat": "perception", "transitive": True, "gloss_en": "watch/look"},
    "kiểm tra": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "check/inspect"},
    "đi": {"pos": "VERB", "subcat": "motion", "transitive": False, "gloss_en": "go"},
    "mua": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "buy"},
    "ăn": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "eat"},
    "nở": {"pos": "VERB", "subcat": "process", "transitive": False, "gloss_en": "bloom"},
    "truyền": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "transmit"},
    "yêu quý": {"pos": "VERB", "subcat": "emotion", "transitive": True, "gloss_en": "cherish/love"},
    "dùng": {"pos": "VERB", "subcat": "action", "transitive": True, "gloss_en": "use"},

    # Adjectives & Descriptors
    "tò mò": {"pos": "ADJ", "subcat": "personality", "gloss_en": "curious"},
    "yếu": {"pos": "ADJ", "subcat": "state", "gloss_en": "weak"},
    "mạnh mẽ": {"pos": "ADJ", "subcat": "state", "gloss_en": "strong"},
    "màu trắng": {"pos": "ADJ", "subcat": "color", "gloss_en": "white"},
    "màu nâu": {"pos": "ADJ", "subcat": "color", "gloss_en": "brown"},
    "kém": {"pos": "ADJ", "subcat": "quality", "gloss_en": "poor/subpar"},
    "đẹp": {"pos": "ADJ", "subcat": "quality", "gloss_en": "beautiful"},
    "nguy hiểm": {"pos": "ADJ", "subcat": "state", "gloss_en": "dangerous"},
    "quý báu": {"pos": "ADJ", "subcat": "quality", "gloss_en": "valuable"},

    # Time & Temporal Adverbials
    "buổi sáng": {"pos": "TIME", "subcat": "time_of_day", "prep": "vào", "gloss_en": "morning"},
    "buổi trưa": {"pos": "TIME", "subcat": "time_of_day", "prep": "vào", "gloss_en": "noon"},
    "buổi tối": {"pos": "TIME", "subcat": "time_of_day", "prep": "vào", "gloss_en": "evening"},
    "mùa xuân": {"pos": "TIME", "subcat": "season", "prep": "vào", "gloss_en": "spring"},
    "hôm qua": {"pos": "TIME", "subcat": "past", "prep": "", "gloss_en": "yesterday"},
    "hôm nay": {"pos": "TIME", "subcat": "present", "prep": "", "gloss_en": "today"},
    "ngày mai": {"pos": "TIME", "subcat": "future", "prep": "vào", "gloss_en": "tomorrow"},

    # Adverbs & Intensifiers
    "thường xuyên": {"pos": "ADV", "subcat": "frequency", "gloss_en": "frequently/often"},
    "rất": {"pos": "ADV", "subcat": "degree", "gloss_en": "very"},
    "nhiều": {"pos": "ADV", "subcat": "quantity", "gloss_en": "much/many"},
    "đừng": {"pos": "ADV", "subcat": "imperative_negation", "gloss_en": "do_not"},

    # Negation Markers
    "không": {"pos": "NEG", "subcat": "general_negation", "gloss_en": "not/no"},
    "chưa": {"pos": "NEG", "subcat": "aspect_negation", "gloss_en": "not_yet"},

    # Interrogatives & Question Markers
    "gì": {"pos": "Q_MARKER", "subcat": "what", "gloss_en": "what"},
    "ở đâu": {"pos": "Q_MARKER", "subcat": "where", "gloss_en": "where"},
    "khi nào": {"pos": "Q_MARKER", "subcat": "when", "gloss_en": "when"},
    "ai": {"pos": "Q_MARKER", "subcat": "who", "gloss_en": "who"},
    "phải không": {"pos": "Q_MARKER", "subcat": "yes_no", "gloss_en": "is_it_true"},
    "bao nhiêu": {"pos": "Q_MARKER", "subcat": "how_many", "gloss_en": "how_many"},
}


def infer_pos(gloss: str, category_raw: str = "") -> dict:
    """Infers POS and grammatical subcategory for glosses not in explicit dict."""
    g = gloss.lower().strip()
    cat = (category_raw or "").lower()

    if g in EXPLICIT_LEXICON:
        return dict(EXPLICIT_LEXICON[g])

    # Rule-based inference
    if "động từ" in cat or any(g.startswith(v) for v in ["chạy", "ăn", "uống", "nói", "làm", "đi", "đến"]):
        return {"pos": "VERB", "subcat": "general", "transitive": True}
    if "tính từ" in cat or any(g.startswith(a) for a in ["tốt", "xấu", "cao", "thấp", "nhanh", "chậm", "màu"]):
        return {"pos": "ADJ", "subcat": "general"}
    if "thời gian" in cat or any(g.startswith(t) for t in ["buổi", "tháng", "năm", "ngày", "tuần"]):
        return {"pos": "TIME", "subcat": "time_general", "prep": "vào"}
    if "câu hỏi" in cat or g in ["ai", "gì", "đâu", "sao", "thế nào"]:
        return {"pos": "Q_MARKER", "subcat": "wh_question"}
    if "con vật" in cat or g.startswith("con "):
        return {"pos": "NOUN", "subcat": "animal", "classifier": "con"}
    if "thực vật" in cat or g.startswith("cây ") or g.startswith("hoa "):
        return {"pos": "NOUN", "subcat": "plant", "classifier": "cây"}
    if "đồ vật" in cat or g.startswith("cái ") or g.startswith("chiếc "):
        return {"pos": "NOUN", "subcat": "object", "classifier": "cái"}

    return {"pos": "NOUN", "subcat": "general", "classifier": ""}


def build_grammar_rules():
    """Defines formal VSL grammar rules and transformations."""
    rules_cfg = [
        "# ===================================================================",
        "# VIETNAMESE SIGN LANGUAGE (VSL) TO SPOKEN VIETNAMESE GRAMMAR RULES",
        "# ===================================================================",
        "# VSL Clause Types:",
        "S -> GREETING_CLAUSE | DECLARATIVE_CLAUSE | NEGATIVE_CLAUSE | INTERROGATIVE_CLAUSE | DESCRIPTIVE_CLAUSE",
        "",
        "# Rule 1: Greetings & Social Formulas",
        "# VSL: [CHÚC MỪNG] [BUỔI SÁNG] [HONORIFIC] -> 'Chúc mừng buổi sáng [Honorific]!'",
        "GREETING_CLAUSE -> GREETING TIME_EXPR [PRON_HONORIFIC]",
        "GREETING -> 'chúc mừng' | 'xin chào'",
        "",
        "# Rule 2: Basic SVO Declarative & Temporal Fronting",
        "# VSL: [TIME]? [SUBJECT] [VERB] [OBJECT]? -> [Vào TIME], [SUBJECT] [VERB] [OBJECT].",
        "DECLARATIVE_CLAUSE -> [TIME_PHRASE] NP VP [NP]",
        "",
        "# Rule 3: Negation Inversion (Critical VSL Grammar Feature)",
        "# VSL Post-verbal negation: [SUBJECT] [VERB] [KHÔNG/CHƯA] -> [SUBJECT] [không/chưa] [VERB]",
        "NEGATIVE_CLAUSE -> NP NEG VP [NP] | VP NEG [NP]",
        "NEG -> 'không' | 'chưa'",
        "",
        "# Rule 4: Topic-Comment (OSV -> SVO)",
        "# VSL fronted topic: [TOPIC_NP] [SUBJECT] [TRANSITIVE_VERB] -> [SUBJECT] [TRANSITIVE_VERB] [TOPIC_NP]",
        "TOPIC_CLAUSE -> NP_TOPIC NP_SUBJ VP_TRANS",
        "",
        "# Rule 5: Interrogative Question Formation",
        "# VSL end-sentence question marker: [SUBJECT] [VERB] [GÌ/Ở ĐÂU/PHẢI KHÔNG] -> Question with '?'",
        "INTERROGATIVE_CLAUSE -> NP VP Q_MARKER | NP Q_MARKER",
        "Q_MARKER -> 'gì' | 'ở đâu' | 'khi nào' | 'ai' | 'phải không' | 'bao nhiêu'",
        "",
        "# Rule 6: Adjectival Predication with Intensifier Insertion",
        "# VSL: [NOUN] [ADJ] -> [NOUN] [rất/là] [ADJ].",
        "DESCRIPTIVE_CLAUSE -> NP [ADV_DEGREE] ADJ_PHRASE",
        "ADV_DEGREE -> 'rất' | 'khá' | 'hơi'",
        "",
        "# Lexical Categories:",
        "NP -> [DET] [CLASSIFIER] NOUN [ADJ] | PRON",
        "VP -> VERB [ADV_FREQ] | [ADV_FREQ] VERB",
        "TIME_PHRASE -> [PREP_TIME] TIME",
        "PREP_TIME -> 'vào' | 'trong'",
    ]

    structured_rules = {
        "negation_transformation": {
            "name": "Post-Verbal Negation Shift",
            "description": "In VSL, negation particles (không, chưa) typically follow the verb. The engine shifts negation to precede the verb.",
            "pattern": ["VERB", "NEG"],
            "target": ["NEG", "VERB"]
        },
        "topic_comment_transformation": {
            "name": "Topic-Comment OSV to SVO",
            "description": "Fronted topics in VSL are reordered to standard Vietnamese Subject-Verb-Object.",
            "pattern": ["NOUN_OBJECT", "PRON_SUBJECT", "VERB_TRANSITIVE"],
            "target": ["PRON_SUBJECT", "VERB_TRANSITIVE", "NOUN_OBJECT"]
        },
        "interrogative_formatting": {
            "name": "Question Marker & Punctuation",
            "description": "Identifies terminal question markers (gì, ở đâu, ai, phải không) and produces natural question phrasing with '?'",
            "markers": ["gì", "ở đâu", "khi nào", "ai", "phải không", "bao nhiêu"]
        },
        "copula_insertion": {
            "name": "Adjectival Copula & Degree Insertion",
            "description": "Inserts appropriate degree adverb (rất) or copula (là) when a bare noun is followed by a stative adjective in VSL.",
            "rule": "NOUN + ADJ -> NOUN + 'rất' + ADJ"
        },
        "greeting_formatting": {
            "name": "Greeting Formula Reordering",
            "description": "Formats VSL greeting sequence [CHÚC MỪNG] [BUỔI SÁNG] [BÀ] into fluent Vietnamese with exclamation point.",
            "rule": "CHÚC_MỪNG + BUỔI_SÁNG + PRON -> 'Chúc mừng buổi sáng' + PRON + '!'"
        }
    }

    return "\n".join(rules_cfg), structured_rules


def main():
    print("=== Creating VSL Lexicon & Grammar Rules ===")

    lexicon_records = []

    # 1. Add all explicitly curated entries
    for gloss, meta in EXPLICIT_LEXICON.items():
        rec = {"word": gloss, "gloss": gloss}
        rec.update(meta)
        lexicon_records.append(rec)

    # 2. Add remaining entries from standardized glosses if available
    if VOCAB_SUMMARY_PATH.exists():
        vocab_df = pd.read_csv(VOCAB_SUMMARY_PATH, dtype=str, low_memory=False).fillna("")
        for _, row in vocab_df.iterrows():
            g = row["normalized_gloss"].strip()
            if g and g not in EXPLICIT_LEXICON:
                inferred = infer_pos(g, row.get("categories", ""))
                rec = {"word": g, "gloss": g}
                rec.update(inferred)
                lexicon_records.append(rec)

    df_lex = pd.DataFrame(lexicon_records)
    df_lex.to_csv(LEXICON_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved Lexicon CSV to {LEXICON_CSV} ({len(df_lex)} words)")

    # Save Lexicon JSON for quick lookup
    lex_dict = {r["word"]: r for r in lexicon_records}
    with open(LEXICON_JSON, "w", encoding="utf-8") as f:
        json.dump(lex_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved Lexicon JSON to {LEXICON_JSON}")

    # Build and save Grammar Rules
    rules_text, rules_json = build_grammar_rules()

    with open(GRAMMAR_RULES_TXT, "w", encoding="utf-8") as f:
        f.write(rules_text)
    print(f"Saved CFG Grammar Rules to {GRAMMAR_RULES_TXT}")

    with open(GRAMMAR_RULES_JSON, "w", encoding="utf-8") as f:
        json.dump(rules_json, f, ensure_ascii=False, indent=2)
    print(f"Saved Structured Grammar Rules to {GRAMMAR_RULES_JSON}")


if __name__ == "__main__":
    main()
