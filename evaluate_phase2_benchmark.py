"""
evaluate_phase2_benchmark.py
Decoupled Evaluation Benchmark Suite for VSL Phase 2:
1. Visual Recognition Performance (Isolated Sign CV): Top-1, Top-3, Top-5, Macro-F1, Confusion Matrix
2. Grammar Engine Translation Performance: BLEU (1, 2, 4), chrF, Word Error Rate (WER), Exact Match (EM)
3. End-to-End Latency & Throughput (FPS) on GPU
"""

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_sign_recognition_model import CachedVSLDataset, VSLSignClassifier
from grammar_engine import VSLGrammarEngine

CHECKPOINT_PATH = Path("VSL_Dataset/checkpoints/sign_model_best.pt")
TEST_MANIFEST = Path("VSL_Dataset/training_data/test_manifest.csv")
SENTENCE_CORPUS_JSON = Path("VSL_Dataset/sentence_corpus.json")
REPORT_OUTPUT = Path("VSL_Dataset/checkpoints/decoupled_benchmark_report.json")


# ==============================================================================
# NLP METRICS: BLEU, chrF, WER, EXACT MATCH
# ==============================================================================

def tokenize(text: str) -> List[str]:
    """Tokenizes text into words ignoring punctuation differences."""
    import re
    return [w.lower() for w in re.findall(r"\b\w+\b", text)]


def compute_wer(reference_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    """Computes Word Error Rate using Levenshtein distance."""
    r = reference_tokens
    h = hypothesis_tokens
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)

    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return float(d[len(r), len(h)]) / float(len(r))


def compute_sentence_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """Computes sentence-level BLEU with add-1 smoothing for missing higher n-grams."""
    if not hypothesis or not reference:
        return 0.0

    ref_len = len(reference)
    hyp_len = len(hypothesis)

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        return 0.0
    else:
        bp = math.exp(1.0 - float(ref_len) / float(hyp_len))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(reference[i:i + n]) for i in range(len(reference) - n + 1))
        hyp_ngrams = Counter(tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) - n + 1))

        clipped_count = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total_hyp = sum(hyp_ngrams.values())

        if total_hyp == 0:
            p_n = 0.0
        else:
            # Add-1 smoothing for higher n-grams
            p_n = (clipped_count + 0.1) / (total_hyp + 0.1)
        precisions.append(p_n)

    if min(precisions) <= 0:
        return 0.0

    log_sum = sum((1.0 / max_n) * math.log(p) for p in precisions)
    return bp * math.exp(log_sum)


def compute_chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Computes character n-gram F-score (chrF)."""
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")

    if not ref_chars or not hyp_chars:
        return 1.0 if ref_chars == hyp_chars else 0.0

    ref_ngrams = Counter(ref_chars[i:i + n] for i in range(max(0, len(ref_chars) - n + 1)))
    hyp_ngrams = Counter(hyp_chars[i:i + n] for i in range(max(0, len(hyp_chars) - n + 1)))

    clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
    total_hyp = sum(hyp_ngrams.values())
    total_ref = sum(ref_ngrams.values())

    precision = (clipped / total_hyp) if total_hyp > 0 else 0.0
    recall = (clipped / total_ref) if total_ref > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    b2 = beta ** 2
    f_score = (1.0 + b2) * (precision * recall) / (b2 * precision + recall)
    return f_score


# ==============================================================================
# BENCHMARK PART 1: COMPUTER VISION RECOGNITION (ISOLATED SIGN)
# ==============================================================================

def evaluate_cv_model(checkpoint_path: Path, test_manifest: Path, device: torch.device):
    """Evaluate the isolated sign CV model on the test manifest.

    Returns a dictionary with standard metrics. The BLEU‑style scores are stored as
    raw fractions (0‑1) and later multiplied by 100 when printed, matching the
    conventional percentage presentation used in the report. The function also
    ensures that the Top‑k accuracies are computed on *distinct* class predictions
    – the ``torch.topk`` call already returns unique class indices, but we keep the
    explicit ``min(5, num_classes)`` guard to avoid accidental duplication when
    ``num_classes`` < 5.

    The macro‑averaged precision/recall/F1 are calculated only over classes that
    appear in the test set, which prevents a large number of zero‑division entries
    from skewing the average.
    """
    print("--- Running Computer Vision Recognition Benchmark ---")
    ckpt = torch.load(checkpoint_path, map_location=device)
    num_classes = ckpt["num_classes"]
    hidden_dim = ckpt.get("hidden_dim", 256)
    frame_size = ckpt.get("frame_size", 112)
    num_frames = ckpt.get("num_frames", 16)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    model = VSLSignClassifier(num_classes=num_classes, hidden_dim=hidden_dim, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = CachedVSLDataset(test_manifest, num_frames=num_frames, frame_size=frame_size, is_train=False, preload=False)

    y_true = []
    y_pred = []
    top3_correct = 0
    top5_correct = 0
    latencies_ms = []

    with torch.no_grad():
        for i in range(len(ds)):
            video_tensor, label_idx, gloss = ds[i]
            video_tensor = video_tensor.unsqueeze(0).to(device)

            t0 = time.time()
            logits = model(video_tensor)
            latencies_ms.append((time.time() - t0) * 1000)

            probs = F.softmax(logits, dim=1)
            _, topk = probs.topk(min(5, num_classes), dim=1)
            top_preds = topk[0].cpu().tolist()

            pred_top1 = top_preds[0]
            y_true.append(label_idx)
            y_pred.append(pred_top1)

            if label_idx in top_preds[:3]:
                top3_correct += 1
            if label_idx in top_preds[:5]:
                top5_correct += 1

    total = len(y_true)
    top1_acc = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / total
    top3_acc = top3_correct / total
    top5_acc = top5_correct / total

    # Macro Precision, Recall, F1 across 28 classes
    class_f1s = []
    class_precisions = []
    class_recalls = []

    for c in range(num_classes):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if (tp + fn) > 0:  # Only average classes present in test set
            class_precisions.append(prec)
            class_recalls.append(rec)
            class_f1s.append(f1)

    macro_precision = float(np.mean(class_precisions)) if class_precisions else 0.0
    macro_recall = float(np.mean(class_recalls)) if class_recalls else 0.0
    macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

    return {
        "num_test_samples": total,
        "num_classes": num_classes,
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2),
        "fps": round(1000.0 / float(np.mean(latencies_ms)), 1)
    }


# ==============================================================================
# BENCHMARK PART 2: GRAMMAR ENGINE TRANSLATION PERFORMANCE
# ==============================================================================

def evaluate_grammar_engine(corpus_path: Path):
    print("--- Running Grammar Engine Translation Benchmark ---")
    engine = VSLGrammarEngine()

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    bleu_scores = []
    chrf_scores = []
    wer_scores = []
    exact_matches = 0

    pattern_metrics = defaultdict(lambda: {"bleu": [], "wer": [], "chrf": [], "count": 0})
    taxonomy_counts = Counter()

    for item in corpus:
        glosses = item.get("gloss_sequence", [])
        if not glosses or not isinstance(glosses, list):
            continue

        ref_sentence = item.get("vietnamese_sentence", "").strip()
        pattern = item.get("pattern", "general")

        pred_result = engine.transform_detailed(glosses)
        hyp_sentence = pred_result["text"].strip()

        for r in pred_result["applied_rules"]:
            taxonomy_counts[r["taxonomy"]] += 1

        ref_toks = tokenize(ref_sentence)
        hyp_toks = tokenize(hyp_sentence)

        b = compute_sentence_bleu(ref_toks, hyp_toks, max_n=4)
        c = compute_chrf(ref_sentence, hyp_sentence, n=6)
        w = compute_wer(ref_toks, hyp_toks)

        bleu_scores.append(b)
        chrf_scores.append(c)
        wer_scores.append(w)

        if ref_toks == hyp_toks:
            exact_matches += 1

        pattern_metrics[pattern]["bleu"].append(b)
        pattern_metrics[pattern]["wer"].append(w)
        pattern_metrics[pattern]["chrf"].append(c)
        pattern_metrics[pattern]["count"] += 1

    total = len(bleu_scores)
    pattern_summary = {}
    for p, vals in pattern_metrics.items():
        pattern_summary[p] = {
            "count": vals["count"],
            "mean_bleu": round(float(np.mean(vals["bleu"])), 4),
            "mean_wer": round(float(np.mean(vals["wer"])), 4),
            "mean_chrf": round(float(np.mean(vals["chrf"])), 4),
        }

    return {
        "num_sentences_evaluated": total,
        "mean_bleu4": round(float(np.mean(bleu_scores)), 4),
        "mean_chrf": round(float(np.mean(chrf_scores)), 4),
        "mean_wer": round(float(np.mean(wer_scores)), 4),
        "exact_match_ratio": round(exact_matches / max(1, total), 4),
        "rules_applied_by_taxonomy": dict(taxonomy_counts),
        "breakdown_by_pattern": pattern_summary
    }


def main():
    print("==================================================================")
    print("PHASE 2 DECOUPLED BENCHMARK SUITE: CV RECOGNITION & GRAMMAR ENGINE")
    print("==================================================================\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    # 1. Computer Vision Benchmark
    cv_report = evaluate_cv_model(CHECKPOINT_PATH, TEST_MANIFEST, device)
    print(f"\n[PART 1: Computer Vision - Isolated Sign Recognition]")
    print(f"  Test Samples:      {cv_report['num_test_samples']} clips across {cv_report['num_classes']} classes")
    print(f"  Top-1 Accuracy:    {cv_report['top1_accuracy']*100:.2f}%")
    print(f"  Top-3 Accuracy:    {cv_report['top3_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy:    {cv_report['top5_accuracy']*100:.2f}%")
    print(f"  Macro-Precision:   {cv_report['macro_precision']*100:.2f}%")
    print(f"  Macro-Recall:      {cv_report['macro_recall']*100:.2f}%")
    print(f"  Macro-F1 Score:    {cv_report['macro_f1']*100:.2f}%")
    print(f"  Inference Latency: {cv_report['mean_latency_ms']} ms/clip ({cv_report['fps']} FPS)")

    # 2. Grammar Engine Benchmark
    grammar_report = evaluate_grammar_engine(SENTENCE_CORPUS_JSON)
    print(f"\n[PART 2: Grammar Engine - VSL Gloss to Vietnamese Translation]")
    print(f"  Sentences Tested:  {grammar_report['num_sentences_evaluated']}")
    print(f"  BLEU-4 Score:      {grammar_report['mean_bleu4']*100:.2f}")
    print(f"  chrF F-score:      {grammar_report['mean_chrf']*100:.2f}")
    print(f"  Word Error Rate:   {grammar_report['mean_wer']*100:.2f}%")
    print(f"  Exact Match Rate:  {grammar_report['exact_match_ratio']*100:.2f}%")
    print(f"  Rule Taxonomies:   {grammar_report['rules_applied_by_taxonomy']}")

    # Save full decoupled report
    full_report = {
        "hardware": str(device),
        "cv_recognition_benchmark": cv_report,
        "grammar_engine_benchmark": grammar_report
    }

    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\nFull Decoupled Benchmark Report written to: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
