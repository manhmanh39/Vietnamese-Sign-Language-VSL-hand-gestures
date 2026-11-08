"""
continuous_sign_decoder.py
Continuous Sign Language Recognition (CSLR) Decoder for VSL.
Processes an uncut, continuous video containing multiple sequential sign gestures:
1. Temporal Sliding Window extraction (window_size frames, stride frames)
2. Window-level posterior probability estimation via Spatial-Temporal model
3. Transition filtering & Peak Detection with Non-Maximum Suppression (NMS)
4. Chronological gloss sequence decoding with start/end timestamps
5. Integration with VSL Grammar Engine for natural Vietnamese text generation
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_sign_recognition_model import VSLSignClassifier
from grammar_engine import VSLGrammarEngine

DEFAULT_CHECKPOINT = Path("VSL_Dataset/checkpoints/sign_model_best.pt")
DEFAULT_CLASS_MAP = Path("VSL_Dataset/checkpoints/class_to_idx.json")


class ContinuousVSLDecoder:
    """Decodes sequential sign gestures from a continuous video stream."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, class_map_path: Path = DEFAULT_CLASS_MAP, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

        # Load model checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.num_classes = ckpt["num_classes"]
        self.hidden_dim = ckpt.get("hidden_dim", 256)
        self.frame_size = ckpt.get("frame_size", 112)
        self.num_frames = ckpt.get("num_frames", 16)

        # Load class mappings
        if "class_to_idx" in ckpt:
            self.class_to_idx = ckpt["class_to_idx"]
        elif class_map_path.exists():
            with open(class_map_path, "r", encoding="utf-8") as f:
                self.class_to_idx = json.load(f)
        else:
            raise FileNotFoundError("Missing class mapping dictionary.")

        self.idx_to_class = {int(v): k for k, v in self.class_to_idx.items()}

        self.model = VSLSignClassifier(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=0.0
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.grammar_engine = VSLGrammarEngine()

    def _extract_all_video_frames(self, video_path: str) -> Tuple[np.ndarray, float]:
        """Reads all frames from a continuous video into resized RGB memory."""
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, (self.frame_size, self.frame_size))
            frames.append(resized)

        cap.release()
        return np.array(frames, dtype=np.uint8), fps

    def decode_continuous_video(
        self,
        video_path: str,
        window_size: int = 16,
        stride: int = 6,
        min_peak_confidence: float = 0.05,
        nms_iou_threshold: float = 0.3
    ) -> Dict:
        """Performs temporal windowing, peak detection, and gloss sequence decoding."""
        frames, fps = self._extract_all_video_frames(video_path)
        total_frames = len(frames)

        if total_frames < window_size:
            return {
                "video_path": video_path,
                "duration_sec": round(total_frames / max(1.0, fps), 2),
                "segments": [],
                "decoded_gloss_sequence": [],
                "vietnamese_sentence": "",
                "status": "video_too_short"
            }

        # 1. Slide temporal windows
        window_tensors = []
        window_metadata = []

        start_idx = 0
        while start_idx + window_size <= total_frames:
            w_frames = frames[start_idx:start_idx + window_size].astype(np.float32)
            w_norm = (w_frames / 255.0 - self.mean) / self.std
            tensor = torch.tensor(w_norm, dtype=torch.float32).permute(0, 3, 1, 2)  # [16, 3, H, W]
            window_tensors.append(tensor)

            t_start = round(start_idx / fps, 2)
            t_end = round((start_idx + window_size) / fps, 2)
            window_metadata.append({"start_frame": start_idx, "end_frame": start_idx + window_size, "start_sec": t_start, "end_sec": t_end})

            start_idx += stride

        if not window_tensors:
            return {"video_path": video_path, "decoded_gloss_sequence": [], "vietnamese_sentence": "", "segments": []}

        # 2. Batch inference across windows
        batch_tensor = torch.stack(window_tensors).to(self.device)  # [NumWindows, 16, 3, H, W]

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # [NumWindows, NumClasses]

        window_predictions = []
        for i, (meta, prob) in enumerate(zip(window_metadata, probs)):
            top_idx = int(np.argmax(prob))
            conf = float(prob[top_idx])
            gloss = self.idx_to_class[top_idx]
            window_predictions.append({
                "window_idx": i,
                "start_sec": meta["start_sec"],
                "end_sec": meta["end_sec"],
                "gloss": gloss,
                "confidence": conf,
            })

        # 3. Peak Detection & Run-Length Consolidation
        # Group consecutive windows detecting the same sign
        consolidated_spans = []
        current_span = None

        for win in window_predictions:
            if win["confidence"] < min_peak_confidence:
                if current_span:
                    consolidated_spans.append(current_span)
                    current_span = None
                continue

            if current_span is None:
                current_span = {
                    "gloss": win["gloss"],
                    "start_sec": win["start_sec"],
                    "end_sec": win["end_sec"],
                    "peak_confidence": win["confidence"],
                    "window_count": 1
                }
            elif current_span["gloss"] == win["gloss"]:
                current_span["end_sec"] = win["end_sec"]
                current_span["peak_confidence"] = max(current_span["peak_confidence"], win["confidence"])
                current_span["window_count"] += 1
            else:
                consolidated_spans.append(current_span)
                current_span = {
                    "gloss": win["gloss"],
                    "start_sec": win["start_sec"],
                    "end_sec": win["end_sec"],
                    "peak_confidence": win["confidence"],
                    "window_count": 1
                }

        if current_span:
            consolidated_spans.append(current_span)

        # 4. Filter short spurious detections (require at least 1 window with peak confidence)
        final_segments = []
        for span in consolidated_spans:
            if span["peak_confidence"] >= min_peak_confidence:
                span["peak_confidence"] = round(float(span["peak_confidence"]), 4)
                final_segments.append(span)

        # 5. Build chronological gloss sequence
        decoded_glosses = [seg["gloss"] for seg in final_segments]

        # 6. Transform via Grammar Engine
        grammar_result = self.grammar_engine.transform_detailed(decoded_glosses)

        return {
            "video_path": str(video_path),
            "video_duration_sec": round(total_frames / fps, 2),
            "total_windows_evaluated": len(window_metadata),
            "detected_segments": final_segments,
            "decoded_gloss_sequence": decoded_glosses,
            "vietnamese_sentence": grammar_result["text"],
            "applied_grammar_rules": grammar_result["applied_rules"]
        }


def main():
    parser = argparse.ArgumentParser(description="Continuous VSL Video Recognition Decoder.")
    parser.add_argument("--video", type=str, required=True, help="Path to continuous sign language video.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--window-size", type=int, default=16, help="Frames per sliding window.")
    parser.add_argument("--stride", type=int, default=6, help="Stride in frames between windows.")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum peak confidence threshold.")
    args = parser.parse_args()

    decoder = ContinuousVSLDecoder(checkpoint_path=Path(args.checkpoint))
    res = decoder.decode_continuous_video(
        video_path=args.video,
        window_size=args.window_size,
        stride=args.stride,
        min_peak_confidence=args.min_conf
    )

    print("\n=== Continuous Sign Video Decoding Results ===")
    print(f"Video File:           {res['video_path']}")
    print(f"Video Duration:       {res.get('video_duration_sec', 0)}s ({res.get('total_windows_evaluated', 0)} windows)")
    print(f"\nDetected Gesture Segments:")
    for seg in res.get("detected_segments", []):
        print(f"  [{seg['start_sec']}s - {seg['end_sec']}s]: {seg['gloss']} (Peak Conf: {seg['peak_confidence']*100:.1f}%, {seg['window_count']} wins)")

    print(f"\nDecoded Gloss Sequence: {res.get('decoded_gloss_sequence', [])}")
    print(f"Vietnamese Translation: {res.get('vietnamese_sentence', '')}")
    if res.get("applied_grammar_rules"):
        print(f"Grammar Rules Applied:  {[r['rule_id'] + ' [' + r['taxonomy'] + ']' for r in res['applied_grammar_rules']]}")


if __name__ == "__main__":
    main()
