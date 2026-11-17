"""
api_server.py
Lightweight HTTP Backend API for VSL (Vietnamese Sign Language) Recognition.
Uses Python standard library (http.server) - Zero external dependencies!

Endpoints:
  POST /api/v1/recognize-clip
    - Accepts: raw video bytes (or multipart) with query/header/body
    - Outputs: predicted gloss, confidence, top-3 candidates, status ("recognized", "uncertain", "error")
  
  POST /api/v1/decode-stream
    - Accepts: continuous video file
    - Outputs: chronological gloss sequence, timestamps, and translated natural Vietnamese sentence
  
  GET /api/v1/labels
    - Returns: list of 28 supported pilot sign classes
  
  GET /api/v1/health
    - Returns: backend service status and device info
"""

import cgi
import io
import json
import os
import tempfile
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from train_sign_recognition_model import TransferVSLClassifier, sample_video_frames
from continuous_sign_decoder import ContinuousVSLDecoder
from grammar_engine import VSLGrammarEngine

# Configuration
CHECKPOINT_PATH = Path("VSL_Dataset/checkpoints/sign_model_best.pt")
CLASS_MAP_PATH = Path("VSL_Dataset/checkpoints/class_to_idx.json")
CONFIDENCE_THRESHOLD = 0.40  # >= 0.40 is "recognized", < 0.40 is "uncertain"

# Global inference engine
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[API Server] Loading model on: {device}")

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
num_classes = ckpt["num_classes"]
hidden_dim = ckpt["hidden_dim"]
backbone = ckpt.get("backbone", "resnet50")
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {int(v): k for k, v in class_to_idx.items()}

model = TransferVSLClassifier(
    num_classes=num_classes,
    hidden_dim=hidden_dim,
    backbone=backbone
)
float32_state = {k: v.float() for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(float32_state)
model.to(device).eval()

grammar_engine = VSLGrammarEngine()
continuous_decoder = ContinuousVSLDecoder(checkpoint_path=CHECKPOINT_PATH, device=str(device))

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def predict_single_clip(video_bytes: bytes) -> Dict:
    """Predicts sign gloss for a single isolated clip."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        t0 = time.time()
        frames = sample_video_frames(tmp_path, num_frames=16, frame_size=224, is_train=False)
        if frames is None or len(frames) == 0 or np.all(frames == 0):
            return {
                "status": "error",
                "error_code": "INVALID_VIDEO",
                "message": "Không thể trích xuất khung hình từ video (tệp lỗi hoặc quá ngắn)"
            }

        frames_norm = (frames.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.tensor(frames_norm, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        topk_idx = probs.argsort()[-3:][::-1]
        top1_idx = int(topk_idx[0])
        top1_conf = round(float(probs[top1_idx]), 4)
        top1_gloss = idx_to_class[top1_idx]

        top3_list = [
            {"gloss": idx_to_class[int(i)], "confidence": round(float(probs[i]), 4)}
            for i in topk_idx
        ]

        latency_ms = round((time.time() - t0) * 1000, 2)
        status = "recognized" if top1_conf >= CONFIDENCE_THRESHOLD else "uncertain"

        # Translation
        trans = grammar_engine.transform_detailed([top1_gloss])

        return {
            "status": status,
            "gloss": top1_gloss,
            "confidence": top1_conf,
            "vietnamese": trans["text"],
            "top3": top3_list,
            "latency_ms": latency_ms,
            "threshold_used": CONFIDENCE_THRESHOLD
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


class VSLRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, data: Dict):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/v1/health":
            self._send_json(200, {
                "status": "healthy",
                "device": str(device),
                "model": "ResNet50-BiGRU-Attention",
                "num_classes": num_classes,
                "confidence_threshold": CONFIDENCE_THRESHOLD
            })
        elif self.path == "/api/v1/labels":
            labels = sorted(list(class_to_idx.keys()))
            self._send_json(200, {
                "total": len(labels),
                "labels": labels
            })
        else:
            self._send_json(404, {"status": "error", "message": "Not Found"})

    def do_POST(self):
        if self.path == "/api/v1/recognize-clip":
            content_type = self.headers.get("Content-Type", "")
            content_length = int(self.headers.get("Content-Length", 0))

            if content_length == 0:
                self._send_json(400, {
                    "status": "error",
                    "error_code": "EMPTY_PAYLOAD",
                    "message": "Không nhận được dữ liệu video"
                })
                return

            # Read payload
            payload = self.rfile.read(content_length)

            # Check if multipart form
            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].encode()
                parts = payload.split(b"--" + boundary)
                video_bytes = None
                for part in parts:
                    if b"filename=" in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            video_bytes = part[header_end + 4:].rstrip(b"\r\n")
                            break
                if not video_bytes:
                    video_bytes = payload
            else:
                video_bytes = payload

            res = predict_single_clip(video_bytes)
            code = 200 if res["status"] in ["recognized", "uncertain"] else 400
            self._send_json(code, res)

        elif self.path == "/api/v1/decode-stream":
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json(400, {"status": "error", "message": "Payload empty"})
                return

            payload = self.rfile.read(content_length)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(payload)
                tmp_path = tmp.name

            try:
                dec_res = continuous_decoder.decode_continuous_video(tmp_path)
                self._send_json(200, {
                    "status": "success",
                    "duration_sec": dec_res.get("video_duration_sec", 0),
                    "gloss_sequence": dec_res.get("decoded_gloss_sequence", []),
                    "vietnamese_translation": dec_res.get("vietnamese_sentence", ""),
                    "segments": dec_res.get("detected_segments", [])
                })
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            self._send_json(404, {"status": "error", "message": "Endpoint not found"})


def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, VSLRequestHandler)
    print(f"=======================================================")
    print(f"VSL Recognition Backend Server running on port {port}...")
    print(f"Health check: http://localhost:{port}/api/v1/health")
    print(f"Labels list : http://localhost:{port}/api/v1/labels")
    print(f"POST clip   : http://localhost:{port}/api/v1/recognize-clip")
    print(f"=======================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
