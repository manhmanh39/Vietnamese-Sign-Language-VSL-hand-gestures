"""
test_lexicon_and_grammar.py
Unit tests verifying the VSL Lexicon, Grammar Rules, Sentence Corpus,
and the calibrated VSL Grammar Engine translation quality.
"""

import unittest
import json
from pathlib import Path
import pandas as pd

from grammar_engine import VSLGrammarEngine

LEXICON_CSV = Path("VSL_Dataset/lexicon.csv")
LEXICON_JSON = Path("VSL_Dataset/lexicon.json")
GRAMMAR_RULES_TXT = Path("VSL_Dataset/grammar_rules.txt")
GRAMMAR_RULES_JSON = Path("VSL_Dataset/grammar_rules.json")
SENTENCE_CORPUS_CSV = Path("VSL_Dataset/sentence_corpus.csv")


class TestVSLLexiconAndGrammar(unittest.TestCase):

    def setUp(self):
        self.engine = VSLGrammarEngine(lexicon_path=LEXICON_JSON, rules_path=GRAMMAR_RULES_JSON)

        self.lexicon_df = pd.read_csv(LEXICON_CSV, dtype=str).fillna("")
        with open(LEXICON_JSON, "r", encoding="utf-8") as f:
            self.lexicon_dict = json.load(f)

        with open(GRAMMAR_RULES_TXT, "r", encoding="utf-8") as f:
            self.grammar_rules_txt = f.read().splitlines()

        with open(GRAMMAR_RULES_JSON, "r", encoding="utf-8") as f:
            self.grammar_rules_json = json.load(f)

    def test_lexicon_not_empty_and_valid(self):
        self.assertFalse(self.lexicon_df.empty, "Lexicon dataframe should not be empty.")
        self.assertGreater(len(self.lexicon_dict), 100, "Lexicon dictionary should contain over 100 entries.")
        self.assertIn("word", self.lexicon_df.columns)
        self.assertIn("pos", self.lexicon_df.columns)

        for expected_word in ["bà", "bánh mì", "buổi sáng", "chúc mừng", "không", "hiểu"]:
            self.assertIn(expected_word, self.lexicon_dict, f"Missing essential VSL word: {expected_word}")
            self.assertTrue(bool(self.lexicon_dict[expected_word].get("pos")), f"Word {expected_word} must have POS tag.")

    def test_grammar_rules_structured_and_taxonomy(self):
        self.assertGreater(len(self.grammar_rules_txt), 0, "CFG rules should not be empty.")
        self.assertIn("negation_transformation", self.grammar_rules_json)
        self.assertIn("topic_comment", self.grammar_rules_json)
        self.assertIn("interrogative_formatting", self.grammar_rules_json)
        self.assertIn("greeting_formatting", self.grammar_rules_json)

        # Verify taxonomy classification
        self.assertEqual(self.grammar_rules_json["negation_transformation"]["taxonomy"], "literature_supported")
        self.assertEqual(self.grammar_rules_json["interrogative_formatting"]["taxonomy"], "literature_supported")
        self.assertEqual(self.grammar_rules_json["greeting_formatting"]["taxonomy"], "dictionary_observed")
        self.assertEqual(self.grammar_rules_json["topic_comment"]["taxonomy"], "heuristic")

    def test_grammar_engine_negation_inversion(self):
        # In VSL: [Verb] + [không] -> [không] + [Verb]
        res = self.engine.transform_detailed(["tôi", "hiểu", "không"])
        self.assertEqual(res["text"], "Tôi không hiểu.")
        rule_ids = [r["rule_id"] for r in res["applied_rules"]]
        self.assertIn("R_NEG_INVERSION", rule_ids)

    def test_grammar_engine_greeting_transformation(self):
        # In VSL: [chúc mừng] [buổi sáng] [bà] -> "Chúc mừng buổi sáng bà!"
        res = self.engine.transform_detailed(["chúc mừng", "buổi sáng", "bà"])
        self.assertEqual(res["text"], "Chúc mừng buổi sáng bà!")
        rule_ids = [r["rule_id"] for r in res["applied_rules"]]
        self.assertIn("R_GREETING_FORMULA", rule_ids)

    def test_grammar_engine_interrogative_formatting(self):
        # In VSL: Question word at end -> question with '?'
        res_what = self.engine.transform_detailed(["bà", "xem", "gì"])
        self.assertTrue(res_what["text"].endswith("?"), "Interrogative sentence must end with '?'")
        self.assertEqual(res_what["text"], "Bà xem gì?")
        self.assertTrue(res_what["is_question"])

        res_where = self.engine.transform(["cửa hàng", "ở đâu"])
        self.assertEqual(res_where, "Cửa hàng ở đâu?")

    def test_grammar_engine_bare_adjective_predicate(self):
        # Calibrated: [con chuột] [yếu] -> "Con chuột yếu." (DO NOT force "rất")
        res = self.engine.transform(["con chuột", "yếu"])
        self.assertEqual(res, "Con chuột yếu.")

    def test_grammar_engine_declarative_svo(self):
        res = self.engine.transform(["bà", "đi", "cửa hàng", "mua", "bánh mì"])
        self.assertEqual(res, "Bà đi cửa hàng mua bánh mì.")

    def test_sentence_corpus_exists_and_populated(self):
        if SENTENCE_CORPUS_CSV.exists():
            df_corpus = pd.read_csv(SENTENCE_CORPUS_CSV)
            self.assertGreater(len(df_corpus), 50, "Sentence corpus should contain at least 50 samples.")
            self.assertIn("gloss_sequence", df_corpus.columns)
            self.assertIn("vietnamese_sentence", df_corpus.columns)


if __name__ == "__main__":
    unittest.main()
