"""
pipeline_video_to_text.py
Phase 2 — End-to-End Offline Video-to-Text Pipeline for Vietnamese Sign Language (VSL).
Steps:
1. Video Input: Loads and preprocesses sign gesture videos.
2. Sign Recognition: Predicts sign glosses with confidence scores using trained PyTorch model.
3. Gloss Sequence Decoding: Assembles, filters, and orders predicted signs.
4. Grammar Engine Integration: Converts VSL gloss sequence into natural, fluent Vietnamese text.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd

from video_to_gloss import VSLSignRecognizer
from grammar_engine import VSLGrammarEngine

CHECKPOINT_PATH = Path("VSL_Dataset/checkpoints/sign_model_best.pt")
CLASS_MAP_PATH = Path("VSL_Dataset/checkpoints/class_to_idx.json")
TEST_MANIFEST_PATH = Path("VSL_Dataset/training_data/test_manifest.csv")


class VSLVideoToTextPipeline:
    """Offline End-to-End VSL Video-to-Text Pipeline."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH, class_map_path: Path = CLASS_MAP_PATH, device: str = None):
        print(f"Initializing VSL Sign Recognizer from {checkpoint_path}...")
        self.recognizer = VSLSignRecognizer(checkpoint_path=checkpoint_path, class_map_path=class_map_path, device=device)
        print("Initializing VSL Grammar Engine...")
        self.grammar_engine = VSLGrammarEngine()

    def process_single_video(self, video_path: str, top_k: int = 3) -> Dict:
        """Processes a single video clip: recognizes gloss and generates Vietnamese equivalent."""
        pred = self.recognizer.predict_video(video_path, top_k=top_k)
        gloss = pred["top1_gloss"]

        # Run through grammar engine
        vietnamese_text = self.grammar_engine.transform([gloss])

        return {
            "video_path": str(video_path),
            "predicted_gloss": gloss,
            "confidence": pred["top1_confidence"],
            "candidates": pred["candidates"],
            "vietnamese_text": vietnamese_text
        }

    def process_video_sequence(self, video_paths: List[str], min_confidence: float = 0.0) -> Dict:
        """Processes an ordered sequence of video clips representing a multi-sign sentence."""
        seq_predictions = self.recognizer.predict_sequence(video_paths, min_confidence=min_confidence)

        gloss_sequence = [p["top1_gloss"] for p in seq_predictions]
        vietnamese_sentence = self.grammar_engine.transform(gloss_sequence)

        return {
            "num_clips": len(video_paths),
            "predicted_sequence": [
                {
                    "video": p["video_path"],
                    "gloss": p["top1_gloss"],
                    "confidence": p["top1_confidence"]
                }
                for p in seq_predictions
            ],
            "raw_gloss_sequence": gloss_sequence,
            "vietnamese_sentence": vietnamese_sentence
        }

    def run_benchmark_demo(self, num_samples: int = 5) -> List[Dict]:
        """Runs offline demonstration using samples from the test manifest."""
        if not TEST_MANIFEST_PATH.exists():
            print(f"Test manifest not found at {TEST_MANIFEST_PATH}")
            return []

        df_test = pd.read_csv(TEST_MANIFEST_PATH)
        print(f"\n=======================================================")
        print(f"RUNNING VSL OFFLINE VIDEO-TO-TEXT BENCHMARK DEMO")
        print(f"Total test samples available: {len(df_test)}")
        print(f"=======================================================\n")

        results = []
        samples = df_test.sample(min(num_samples, len(df_test)), random_state=42)

        for _, row in samples.iterrows():
            v_path = row["video_path"]
            true_label = row["gloss"]
            res = self.process_single_video(v_path)
            res["ground_truth_gloss"] = true_label
            res["match"] = (res["predicted_gloss"].strip().lower() == true_label.strip().lower())
            results.append(res)

            print(f"Video File:     {Path(v_path).name}")
            print(f"Ground Truth:   {true_label}")
            print(f"Predicted Sign: {res['predicted_gloss']} (Confidence: {res['confidence']*100:.1f}%)")
            print(f"Top-3:          {[c['gloss'] for c in res['candidates']]}")
            print(f"Vietnamese Text:{res['vietnamese_text']}")
            print(f"Match:          {'CORRECT' if res['match'] else 'DIFFERENT'}")
            print("-" * 55)

        return results


def main():
    parser = argparse.ArgumentParser(description="End-to-End VSL Video-to-Text Pipeline.")
    parser.add_argument("--video", type=str, help="Path to single input video.")
    parser.add_argument("--videos", nargs="+", help="Ordered list of input video paths.")
    parser.add_argument("--continuous", action="store_true", help="Run Continuous Sign Recognition (temporal sliding window) on input video.")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Model checkpoint path.")
    parser.add_argument("--demo", action="store_true", help="Run benchmark demo on test manifest.")
    parser.add_argument("--output", type=str, help="Optional output JSON path for results.")
    args = parser.parse_args()

    if args.continuous and args.video:
        from continuous_sign_decoder import ContinuousVSLDecoder
        decoder = ContinuousVSLDecoder(checkpoint_path=Path(args.checkpoint))
        res = decoder.decode_continuous_video(args.video)
        print("\n=== Continuous Video-to-Text Result ===")
        print(f"Video File:           {res['video_path']}")
        print(f"Video Duration:       {res.get('video_duration_sec', 0)}s ({res.get('total_windows_evaluated', 0)} windows)")
        print(f"Detected Segments:    {len(res.get('detected_segments', []))}")
        for s in res.get("detected_segments", []):
            print(f"  [{s['start_sec']}s - {s['end_sec']}s]: {s['gloss']} (Conf: {s['peak_confidence']*100:.1f}%)")
        print(f"Decoded Glosses:      {res.get('decoded_gloss_sequence', [])}")
        print(f"Vietnamese Sentence:  {res.get('vietnamese_sentence', '')}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        return

    pipeline = VSLVideoToTextPipeline(checkpoint_path=Path(args.checkpoint))

    if args.demo:
        results = pipeline.run_benchmark_demo(num_samples=8)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    elif args.video:
        res = pipeline.process_single_video(args.video)
        print("\n=== Single Video-to-Text Result ===")
        print(f"Input Video:     {res['video_path']}")
        print(f"Predicted Sign:  {res['predicted_gloss']} ({res['confidence']*100:.1f}%)")
        print(f"Candidates:      {res['candidates']}")
        print(f"Vietnamese Text: {res['vietnamese_text']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

    elif args.videos:
        res = pipeline.process_video_sequence(args.videos)
        print("\n=== Video Sequence-to-Text Result ===")
        print(f"Processed {res['num_clips']} sign clips.")
        print(f"Gloss Sequence:    {res['raw_gloss_sequence']}")
        print(f"Vietnamese Output: {res['vietnamese_sentence']}")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

    else:
        print("Please provide --video, --videos, or --demo flag. Use -h for help.")


if __name__ == "__main__":
    main()
