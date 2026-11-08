"""
video_to_gloss.py
Inference module for recognizing sign gestures from VSL video clips
and converting model predictions into decoded gloss sequences.
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

DEFAULT_CHECKPOINT = Path("VSL_Dataset/checkpoints/sign_model_best.pt")
DEFAULT_CLASS_MAP = Path("VSL_Dataset/checkpoints/class_to_idx.json")


class VSLSignRecognizer:
    """Loads a trained VSL recognition model and predicts gloss labels from video files."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, class_map_path: Path = DEFAULT_CLASS_MAP, device: Optional[str] = None):
        if not checkpoint_path.exists():
            # Fallback to final model
            alt_ckpt = checkpoint_path.parent / "sign_model_final.pt"
            if alt_ckpt.exists():
                checkpoint_path = alt_ckpt
            else:
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
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

        # Instantiate model architecture
        self.model = VSLSignClassifier(
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=0.0
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def preprocess_video(self, video_path: str) -> torch.Tensor:
        """Extracts and normalizes uniform temporal frames from video."""
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(p))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0 or not cap.isOpened():
            cap.release()
            raise ValueError(f"Unable to read frames from video: {video_path}")

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frames.append(frame)

        cap.release()
        frames_arr = np.array(frames, dtype=np.float32)

        # Normalize
        frames_arr = (frames_arr / 255.0 - self.mean) / self.std

        # Convert to tensor: [1, T, C, H, W]
        tensor = torch.tensor(frames_arr, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0)
        return tensor

    def predict_video(self, video_path: str, top_k: int = 3) -> Dict:
        """Predicts top-1 and top-K sign glosses with confidence scores."""
        tensor = self.preprocess_video(video_path).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            confs, indices = probs.topk(min(top_k, self.num_classes))

        top1_idx = int(indices[0].item())
        top1_gloss = self.idx_to_class[top1_idx]
        top1_conf = float(confs[0].item())

        candidates = [
            {"gloss": self.idx_to_class[int(idx.item())], "confidence": round(float(conf.item()), 4)}
            for conf, idx in zip(confs, indices)
        ]

        return {
            "video_path": str(video_path),
            "top1_gloss": top1_gloss,
            "top1_confidence": round(top1_conf, 4),
            "candidates": candidates
        }

    def predict_sequence(self, video_paths: List[str], min_confidence: float = 0.0, deduplicate: bool = True) -> List[Dict]:
        """Predicts gloss sequence from an ordered sequence of video clips."""
        results = []
        last_gloss = None

        for path in video_paths:
            pred = self.predict_video(path)
            gloss = pred["top1_gloss"]
            conf = pred["top1_confidence"]

            if conf < min_confidence:
                continue

            if deduplicate and gloss == last_gloss:
                continue

            results.append(pred)
            last_gloss = gloss

        return results


def main():
    parser = argparse.ArgumentParser(description="Predict VSL Gloss from Video.")
    parser.add_argument("--video", type=str, help="Single video file path.")
    parser.add_argument("--videos", nargs="+", help="Ordered list of video file paths.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K candidate glosses.")
    args = parser.parse_args()

    recognizer = VSLSignRecognizer(checkpoint_path=Path(args.checkpoint))

    if args.video:
        res = recognizer.predict_video(args.video, top_k=args.top_k)
        print(f"\n--- Prediction for {args.video} ---")
        print(f"Top-1 Gloss:      {res['top1_gloss']} (Confidence: {res['top1_confidence']*100:.1f}%)")
        print(f"Top Candidates:   {res['candidates']}")

    elif args.videos:
        print(f"\n--- Predicting Sequence for {len(args.videos)} video clips ---")
        seq_results = recognizer.predict_sequence(args.videos)
        gloss_seq = [r["top1_gloss"] for r in seq_results]
        print(f"Decoded Gloss Sequence: {gloss_seq}")
        for r in seq_results:
            print(f"  {r['video_path']} -> {r['top1_gloss']} ({r['top1_confidence']*100:.1f}%)")


if __name__ == "__main__":
    main()
