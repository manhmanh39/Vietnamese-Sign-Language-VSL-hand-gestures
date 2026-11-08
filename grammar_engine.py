"""
grammar_engine.py
Vietnamese Sign Language (VSL) Grammar Engine.
Converts predicted VSL gloss sequences into natural, grammatically correct Vietnamese sentences.

Rule Taxonomy Classification:
- literature_supported: Grounded in established VSL linguistic research (e.g., post-verbal negation inversion, terminal WH-question markers).
- dictionary_observed: Empirically validated against authentic VSL phrases and dictionary examples (e.g., greeting formulas, frequency adverb positioning).
- heuristic: Practical heuristics for surface naturalness, calibrated to avoid over-generalization.
- unvalidated: Experimental rules awaiting empirical linguistic verification.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

LEXICON_JSON_PATH = Path("VSL_Dataset/lexicon.json")
GRAMMAR_RULES_PATH = Path("VSL_Dataset/grammar_rules.json")


class VSLGrammarEngine:
    def __init__(self, lexicon_path: Path = LEXICON_JSON_PATH, rules_path: Path = GRAMMAR_RULES_PATH):
        self.lexicon: Dict[str, dict] = {}
        self.rules: dict = {}

        if lexicon_path.exists():
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)

        if rules_path.exists():
            with open(rules_path, "r", encoding="utf-8") as f:
                self.rules = json.load(f)

    def lookup(self, gloss: str) -> dict:
        """Looks up POS and linguistic features of a gloss."""
        g = gloss.strip().lower()
        if g in self.lexicon:
            return dict(self.lexicon[g])

        # Heuristic fallback
        if g in ["không", "chưa", "chẳng"]:
            return {"pos": "NEG", "word": g, "subcat": "negation"}
        if g in ["gì", "ở đâu", "ai", "khi nào", "sao", "phải không", "bao nhiêu"]:
            return {"pos": "Q_MARKER", "word": g, "subcat": "question"}
        if g in ["rất", "quá", "lắm", "hơi", "thường xuyên"]:
            return {"pos": "ADV", "word": g, "subcat": "adverb"}
        if g in ["bà", "ông", "mẹ", "ba", "anh", "chị", "em", "em trai", "tôi", "bạn", "chúng tôi"]:
            return {"pos": "PRON", "word": g, "subcat": "pronoun"}

        return {"pos": "NOUN", "word": g, "subcat": "general"}

    def transform_detailed(self, gloss_sequence: List[str]) -> Dict:
        """Transforms a VSL gloss sequence into natural Vietnamese text,
        returning the generated sentence along with an audit trail of applied rules
        and their taxonomy classification.
        """
        if not gloss_sequence:
            return {"text": "", "applied_rules": [], "tokens": [], "is_question": False}

        tokens = [g.strip().lower() for g in gloss_sequence if g and g.strip()]
        if not tokens:
            return {"text": "", "applied_rules": [], "tokens": [], "is_question": False}

        tagged = [self.lookup(t) for t in tokens]
        n = len(tagged)
        applied_rules = []

        # ----------------------------------------------------------------------
        # Rule 1: Greeting & Social Formulas [dictionary_observed]
        # ----------------------------------------------------------------------
        is_greeting = any(t["word"] in ["chúc mừng", "xin chào"] for t in tagged)
        if is_greeting:
            greeting_word = next((t["word"] for t in tagged if t["word"] in ["chúc mừng", "xin chào"]), "Chúc mừng")
            other_tokens = [t for t in tagged if t["word"] not in ["chúc mừng", "xin chào"]]

            time_tokens = [t["word"] for t in other_tokens if t.get("pos") == "TIME"]
            pron_tokens = [t["word"] for t in other_tokens if t.get("pos") == "PRON"]
            noun_tokens = [t["word"] for t in other_tokens if t.get("pos") not in ["TIME", "PRON"]]

            parts = [greeting_word.capitalize()]
            if time_tokens:
                parts.extend(time_tokens)
            if pron_tokens:
                parts.extend(pron_tokens)
            if noun_tokens:
                parts.extend(noun_tokens)

            sentence = " ".join(parts).strip() + "!"
            applied_rules.append({
                "rule_id": "R_GREETING_FORMULA",
                "taxonomy": "dictionary_observed",
                "description": "Greeting formula reordering (chúc mừng + time + recipient -> exclamation)."
            })
            return {"text": sentence, "applied_rules": applied_rules, "tokens": tagged, "is_question": False}

        # Social formula: CẢM ƠN [dictionary_observed]
        if tagged[0]["word"] == "cảm ơn":
            parts = ["Cảm ơn"]
            recipients = [t["word"] for t in tagged[1:] if t.get("pos") in ["PRON", "NOUN"]]
            actions = [t["word"] for t in tagged[1:] if t.get("pos") == "VERB"]
            if recipients:
                parts.extend(recipients)
            if actions:
                parts.append("đã")
                parts.extend(actions)
            sentence = " ".join(parts).strip() + "."
            applied_rules.append({
                "rule_id": "R_SOCIAL_FORMULA",
                "taxonomy": "dictionary_observed",
                "description": "Thanking formula structure (cảm ơn + recipient + đã + action)."
            })
            return {"text": sentence, "applied_rules": applied_rules, "tokens": tagged, "is_question": False}

        # ----------------------------------------------------------------------
        # Rule 2: Post-Verbal Negation Inversion [literature_supported]
        # In VSL, negation particles (không, chưa) occur post-verbally.
        # Spoken Vietnamese places negation before the main verb.
        # ----------------------------------------------------------------------
        has_negation = any(t.get("pos") == "NEG" for t in tagged)
        if has_negation:
            neg_idx = next(i for i, t in enumerate(tagged) if t.get("pos") == "NEG")
            if neg_idx > 0:
                prev_token = tagged[neg_idx - 1]
                if prev_token.get("pos") in ["VERB", "ADJ"]:
                    # Shift negation before the verb / predicate
                    tagged = tagged[:neg_idx - 1] + [tagged[neg_idx], prev_token] + tagged[neg_idx + 1:]
                    applied_rules.append({
                        "rule_id": "R_NEG_INVERSION",
                        "taxonomy": "literature_supported",
                        "description": "Post-verbal negation shifted to pre-verbal position (V + không -> không + V)."
                    })

        # ----------------------------------------------------------------------
        # Rule 3: Frequency Adverb Pre-verbal Placement [dictionary_observed]
        # ----------------------------------------------------------------------
        for i in range(len(tagged) - 1):
            if tagged[i].get("pos") == "VERB" and tagged[i + 1]["word"] in ["thường xuyên", "luôn luôn"]:
                tagged[i], tagged[i + 1] = tagged[i + 1], tagged[i]
                applied_rules.append({
                    "rule_id": "R_FREQ_ADVERB",
                    "taxonomy": "dictionary_observed",
                    "description": "Frequency adverb placed before verbal predicate."
                })
                break

        # ----------------------------------------------------------------------
        # Rule 4: Topic-Comment Handling [heuristic - Softened]
        # Topic-fronted sentences are valid in spoken Vietnamese (e.g. 'Bánh mì này tôi thích ăn').
        # Only normalize if ambiguity exists or when specifically configured.
        # ----------------------------------------------------------------------
        # No aggressive forced reordering of OSV to SVO, preserving natural topic-comment.

        # ----------------------------------------------------------------------
        # Rule 5: Adjectival Predication [heuristic - Softened]
        # Calibrated: DO NOT force "rất" into adjectival predicates.
        # Bare predicates like 'Con chuột yếu.' or 'Màn màu trắng.' are grammatically valid.
        # ----------------------------------------------------------------------
        # Bare adjectives are retained as natural predicates without inserting unprompted degree adverbs.

        # ----------------------------------------------------------------------
        # Rule 6: Terminal Interrogative Markers & Punctuation [literature_supported]
        # ----------------------------------------------------------------------
        is_question = any(t.get("pos") == "Q_MARKER" for t in tagged) or any(
            t["word"] in ["gì", "ở đâu", "ai", "khi nào", "sao", "phải không", "bao nhiêu"] for t in tagged
        )
        if is_question:
            applied_rules.append({
                "rule_id": "R_WH_QUESTION",
                "taxonomy": "literature_supported",
                "description": "Interrogative terminal marker detected, formatted with '?'."
            })

        # ----------------------------------------------------------------------
        # Assemble Surface String
        # ----------------------------------------------------------------------
        words = []
        for i, item in enumerate(tagged):
            w = item["word"]
            if item.get("pos") == "TIME" and i > 0 and tagged[i - 1].get("pos") == "VERB":
                words.append("vào")
                applied_rules.append({
                    "rule_id": "R_TIME_PREPOSITION",
                    "taxonomy": "dictionary_observed",
                    "description": "Temporal preposition 'vào' inserted before post-verbal time expression."
                })
            words.append(w)

        raw_sentence = " ".join(words).strip()
        clean_sentence = re.sub(r"\s+", " ", raw_sentence)

        if clean_sentence:
            clean_sentence = clean_sentence[0].upper() + clean_sentence[1:]

        if is_question:
            if not clean_sentence.endswith("?"):
                clean_sentence = clean_sentence.rstrip(".!;, ") + "?"
        else:
            if not clean_sentence.endswith((".", "!", "?")):
                clean_sentence = clean_sentence.rstrip(";, ") + "."

        return {
            "text": clean_sentence,
            "applied_rules": applied_rules,
            "tokens": tagged,
            "is_question": is_question
        }

    def transform(self, gloss_sequence: List[str]) -> str:
        """Convenience method returning just the surface sentence string."""
        return self.transform_detailed(gloss_sequence)["text"]


_engine_singleton = None

def glosses_to_vietnamese(gloss_sequence: List[str]) -> str:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = VSLGrammarEngine()
    return _engine_singleton.transform(gloss_sequence)


if __name__ == "__main__":
    engine = VSLGrammarEngine()

    test_cases = [
        ["chúc mừng", "buổi sáng", "bà"],
        ["tôi", "hiểu", "không"],
        ["em trai", "học giỏi", "thường xuyên"],
        ["bà", "xem", "gì"],
        ["cửa hàng", "ở đâu"],
        ["con chuột", "yếu"],
        ["bánh mì", "tôi", "thích", "ăn"],
        ["cảm ơn", "bà", "giúp đỡ"],
    ]

    print("=== Testing Calibrated VSL Grammar Engine ===")
    for seq in test_cases:
        res = engine.transform_detailed(seq)
        print(f"Gloss Seq : {seq}")
        print(f"Vietnamese: {res['text']}")
        print(f"Rules     : {[r['rule_id'] + ' (' + r['taxonomy'] + ')' for r in res['applied_rules']]}\n")
