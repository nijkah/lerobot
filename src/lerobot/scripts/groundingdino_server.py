#!/usr/bin/env python

"""Minimal GroundingDINO HTTP inference server.

This script is intended to run on the GPU node that hosts the GroundingDINO weights.
It exposes a simple HTTP interface so that a remote robot controller (CPU node)
can POST camera frames and text prompts to receive detections.

Example:
    python -m lerobot.scripts.groundingdino_server \
        --host 0.0.0.0 --port 8080 \
        --grounding-config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounding-checkpoint /data/weights/groundingdino_swint_ogc.pth

Request format (POST /predict):
    {
        "caption": "blue mug on table",
        "image_b64": "<base64 JPEG or PNG bytes>",
        "box_threshold": 0.4,         # optional override
        "text_threshold": 0.25,       # optional
        "min_box_area": 1500.0        # optional
    }

Response format:
    {
        "detections": [
            {"bbox": [x1, y1, x2, y2], "confidence": 0.87, "label": "blue mug"}
        ],
        "best": {"bbox": [...], "confidence": 0.87, "label": "blue mug"},
        "latency_ms": 42.1
    }
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
GROUNDING_REPO = REPO_ROOT / "GroundingDINO"


@dataclass
class DetectionResult:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    label: str


class GroundingDinoDetector:
    """Thin wrapper around GroundingDINO's inference helper."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str):
        if not GROUNDING_REPO.exists():
            raise FileNotFoundError(
                f"GroundingDINO submodule not found at {GROUNDING_REPO}. "
                "Clone https://github.com/IDEA-Research/GroundingDINO into the repo root."
            )
        if str(GROUNDING_REPO) not in sys.path:
            sys.path.insert(0, str(GROUNDING_REPO))

        from groundingdino.util.inference import Model  # type: ignore[import-not-found]

        config_path = str(Path(config_path).expanduser())
        checkpoint_path = str(Path(checkpoint_path).expanduser())
        if not Path(config_path).exists():
            raise FileNotFoundError(f"GroundingDINO config not found: {config_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"GroundingDINO checkpoint not found: {checkpoint_path}")

        LOG.info("Loading GroundingDINO model from %s", config_path)
        self.model = Model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device=device,
        )

    def predict(
        self,
        frame_bgr: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
        top_k: int | None = None,
    ) -> list[DetectionResult]:
        detections, phrases = self.model.predict_with_caption(
            image=frame_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if detections.xyxy.size == 0:
            return []

        results: list[DetectionResult] = []
        for bbox, score, phrase in zip(detections.xyxy, detections.confidence, phrases, strict=False):
            x1, y1, x2, y2 = bbox.astype(float)
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if area < min_box_area:
                continue
            results.append(
                DetectionResult(
                    bbox=np.array([x1, y1, x2, y2], dtype=float),
                    confidence=float(score),
                    label=phrase,
                )
            )

        results.sort(key=lambda det: det.confidence, reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results


@dataclass
class ServerState:
    detector: GroundingDinoDetector
    default_box_threshold: float
    default_text_threshold: float
    default_min_box_area: float
    top_k: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class GroundingDINOServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, state: ServerState):
        super().__init__(server_address, handler_cls)
        self.state = state


class GroundingDINORequestHandler(BaseHTTPRequestHandler):
    server: GroundingDINOServer  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOG.info("%s - %s", self.address_string(), format % args)

    def do_GET(self):  # noqa: N802
        if self.path == "/healthz":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):  # noqa: N802
        if self.path != "/predict":
            self.send_error(404, "Not Found")
            return
        try:
            body = self._read_json()
        except ValueError as exc:
            self.send_error(400, str(exc))
            return

        caption = body.get("caption")
        if not caption or not isinstance(caption, str):
            self.send_error(400, "Field 'caption' is required.")
            return
        try:
            frame = self._decode_image(body)
        except ValueError as exc:
            self.send_error(400, str(exc))
            return

        box_threshold = float(body.get("box_threshold", self.server.state.default_box_threshold))
        text_threshold = float(body.get("text_threshold", self.server.state.default_text_threshold))
        min_box_area = float(body.get("min_box_area", self.server.state.default_min_box_area))
        top_k = self.server.state.top_k
        if "top_k" in body:
            val = body["top_k"]
            if val is not None:
                top_k = max(1, int(val))
            else:
                top_k = None

        start = time.perf_counter()
        with self.server.state.lock:
            detections = self.server.state.detector.predict(
                frame_bgr=frame,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                min_box_area=min_box_area,
                top_k=top_k,
            )
        latency_ms = (time.perf_counter() - start) * 1e3
        response = {
            "detections": [
                {
                    "bbox": det.bbox.tolist(),
                    "confidence": det.confidence,
                    "label": det.label,
                }
                for det in detections
            ],
            "best": (
                {
                    "bbox": detections[0].bbox.tolist(),
                    "confidence": detections[0].confidence,
                    "label": detections[0].label,
                }
                if detections
                else None
            ),
            "latency_ms": latency_ms,
        }
        self._send_json(response)

    def _send_json(self, payload: dict[str, Any], code: int = 200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("Missing request body.")
        data = self.rfile.read(length)
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise ValueError(f"Invalid JSON payload: {exc}") from exc

    @staticmethod
    def _decode_image(body: dict[str, Any]) -> np.ndarray:
        if "image_b64" in body:
            encoded = body["image_b64"]
            if not isinstance(encoded, str):
                raise ValueError("'image_b64' must be a base64 string.")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to decode base64 image: {exc}") from exc
            np_arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode failed for provided image.")
            return frame
        elif "image_path" in body:
            path = Path(body["image_path"]).expanduser()
            if not path.exists():
                raise ValueError(f"image_path not found: {path}")
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read image from {path}")
            return frame
        raise ValueError("Provide 'image_b64' (recommended) or 'image_path'.")


def _default_paths() -> tuple[str | None, str | None]:
    config_path = GROUNDING_REPO / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt_path = REPO_ROOT / "weights" / "groundingdino_swint_ogc.pth"
    return (
        str(config_path) if config_path.exists() else None,
        str(ckpt_path) if ckpt_path.exists() else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    parser.add_argument("--grounding-config", help="GroundingDINO config path.")
    parser.add_argument("--grounding-checkpoint", help="GroundingDINO checkpoint path.")
    parser.add_argument("--grounding-device", default="cuda", help="Torch device for GroundingDINO.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="Default box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Default text threshold.")
    parser.add_argument("--min-box-area", type=float, default=1_500.0, help="Default minimum box area.")
    parser.add_argument("--top-k", type=int, help="Max detections to return per request (default: unlimited).")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    config_path = args.grounding_config
    checkpoint_path = args.grounding_checkpoint
    if config_path is None or checkpoint_path is None:
        default_config, default_ckpt = _default_paths()
        config_path = config_path or default_config
        checkpoint_path = checkpoint_path or default_ckpt
    if config_path is None or checkpoint_path is None:
        raise ValueError("Provide --grounding-config and --grounding-checkpoint (or place defaults under weights/).")

    detector = GroundingDinoDetector(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=args.grounding_device,
    )
    state = ServerState(
        detector=detector,
        default_box_threshold=args.box_threshold,
        default_text_threshold=args.text_threshold,
        default_min_box_area=args.min_box_area,
        top_k=args.top_k,
    )

    server = GroundingDINOServer((args.host, args.port), GroundingDINORequestHandler, state)
    LOG.info("GroundingDINO server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
