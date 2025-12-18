#!/usr/bin/env python

"""HTTP inference server for the Grounded SAM 2 demo pipeline.

This server mirrors :mod:`lerobot.scripts.groundingdino_server` but runs the
Grounded-SAM-2 *local* demo (Grounding DINO + SAM 2 segmentation) so that the
same checkpoints used in ``grounded_sam2_local_demo.py`` can serve HTTP
requests.
It accepts POST requests with a text prompt and image payload, then returns
bounding boxes plus optional masks for each detected object.

Example usage::

    python -m lerobot.scripts.groundedsam2_server \
        --host 0.0.0.0 --port 8090 \
        --grounding-config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --grounding-checkpoint Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth \
        --sam2-model-config configs/sam2.1/sam2.1_hiera_l.yaml \
        --sam2-checkpoint Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt

The HTTP interface matches ``groundingdino_server`` so that existing clients
can be re-used. Masks are returned as base64-encoded PNGs under ``mask_png``.
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
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
GROUNDED_SAM2_REPO = REPO_ROOT / "Grounded-SAM-2"
SAM2_PACKAGE_DIR = GROUNDED_SAM2_REPO / "sam2"
GROUNDING_DINO_DIR = GROUNDED_SAM2_REPO / "grounding_dino"
LOG = logging.getLogger(__name__)


@dataclass
class GroundingDINODetection:
    bbox: np.ndarray
    confidence: float
    label: str
    mask: np.ndarray | None = None


class GroundedSAM2Pipeline:
    """Wrapper around the local Grounded SAM 2 demo pipeline."""

    def __init__(
        self,
        *,
        grounding_config_path: str,
        grounding_checkpoint_path: str,
        sam2_config_path: str,
        sam2_checkpoint_path: str,
        device: str,
        normalize_caption: bool = True,
        autocast: bool = True,
        multimask_output: bool = False,
    ) -> None:
        if not GROUNDED_SAM2_REPO.exists():
            raise FileNotFoundError(
                f"Grounded-SAM-2 repo not found at {GROUNDED_SAM2_REPO}."
                " Clone https://github.com/IDEA-Research/Grounded-SAM-2 into the repo root."
            )
        if str(GROUNDED_SAM2_REPO) not in sys.path:
            sys.path.insert(0, str(GROUNDED_SAM2_REPO))
        if str(GROUNDING_DINO_DIR) not in sys.path:
            sys.path.insert(0, str(GROUNDING_DINO_DIR))

        from sam2.build_sam import build_sam2  # type: ignore[import-not-found]
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore[import-not-found]
        from grounding_dino.groundingdino.util.inference import Model as GroundingDINOModel  # type: ignore[import-not-found]

        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but torch.cuda is unavailable.")
            if self.device.index is not None:
                torch.cuda.set_device(self.device.index)
                index = self.device.index
            else:
                index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        sam2_config_name = resolve_sam2_config_name(sam2_config_path)
        sam2_config_file = SAM2_PACKAGE_DIR / sam2_config_name
        sam2_checkpoint_path = str(Path(sam2_checkpoint_path).expanduser())
        grounding_config_path = str(Path(grounding_config_path).expanduser())
        grounding_checkpoint_path = str(Path(grounding_checkpoint_path).expanduser())
        for path, desc in (
            (sam2_checkpoint_path, "SAM2 checkpoint"),
            (grounding_config_path, "GroundingDINO config"),
            (grounding_checkpoint_path, "GroundingDINO checkpoint"),
        ):
            if not Path(path).exists():
                raise FileNotFoundError(f"{desc} not found: {path}")

        LOG.info("Loading SAM2 model from %s", sam2_config_file)
        sam2_model = build_sam2(sam2_config_name, sam2_checkpoint_path, device=str(self.device))
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        LOG.info("Loading GroundingDINO from %s", grounding_config_path)
        self.grounding_model = GroundingDINOModel(
            model_config_path=grounding_config_path,
            model_checkpoint_path=grounding_checkpoint_path,
            device=str(self.device),
        )

        self.normalize_caption = normalize_caption
        self.autocast_enabled = autocast and self.device.type == "cuda" and torch.cuda.is_available()
        self.multimask_output = bool(multimask_output)

    def predict(
        self,
        *,
        frame_bgr: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
        top_k: int | None,
    ) -> list[GroundingDINODetection]:
        if frame_bgr is None:
            raise ValueError("frame_bgr cannot be None")
        if not caption:
            return []
        prompt = self._normalize_caption(caption) if self.normalize_caption else caption

        self.sam2_predictor.set_image(frame_bgr)

        detections, phrases = self.grounding_model.predict_with_caption(
            image=frame_bgr,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if detections.xyxy.size == 0:
            return []
        boxes = detections.xyxy.astype(float)
        scores = detections.confidence.astype(float)
        labels = [str(label) for label in phrases]

        keep: list[int] = []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if area >= min_box_area:
                keep.append(idx)
        if not keep:
            return []

        boxes = boxes[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep]

        order = np.argsort(scores)[::-1]
        if top_k is not None:
            order = order[: top_k]
        boxes = boxes[order]
        scores = scores[order]
        labels = [labels[i] for i in order]

        predict_kwargs = {
            "point_coords": None,
            "point_labels": None,
            "box": boxes.astype(np.float32),
            "multimask_output": self.multimask_output,
        }
        if self.autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores_mask, _ = self.sam2_predictor.predict(**predict_kwargs)
        else:
            masks, scores_mask, _ = self.sam2_predictor.predict(**predict_kwargs)
        if self.multimask_output and masks.ndim == 4:
            best = np.argmax(scores_mask, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool)

        detections: list[GroundingDINODetection] = []
        for box, score, label, mask in zip(boxes, scores, labels, masks, strict=False):
            detections.append(
                GroundingDINODetection(
                    bbox=box.astype(float),
                    confidence=float(score),
                    label=label,
                    mask=mask,
                )
            )
        return detections

    def _normalize_caption(self, caption: str) -> str:
        normalized = caption.strip().lower()
        if normalized and not normalized.endswith("."):
            normalized = f"{normalized}."
        return normalized


def _relative_sam2_config(path: Path) -> str | None:
    config_path = SAM2_PACKAGE_DIR / path
    if config_path.exists():
        return path.as_posix()
    return None


def resolve_sam2_config_name(config: str) -> str:
    package_root = SAM2_PACKAGE_DIR.resolve()
    candidates = [
        Path(config).expanduser(),
        (REPO_ROOT / config).expanduser(),
        (GROUNDED_SAM2_REPO / config).expanduser(),
        (SAM2_PACKAGE_DIR / config).expanduser(),
    ]
    outside_candidate: Path | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(package_root)
        except ValueError:
            outside_candidate = resolved
            continue
        return relative.as_posix()
    if outside_candidate is not None:
        raise ValueError(f"SAM 2 config must live under {package_root}, got {outside_candidate}.")
    raise FileNotFoundError(f"SAM 2 config not found: {config}")


def default_groundedsam2_paths() -> tuple[str | None, str | None]:
    config_rel = Path("configs") / "sam2.1" / "sam2.1_hiera_l.yaml"
    config = _relative_sam2_config(config_rel)
    ckpt = GROUNDED_SAM2_REPO / "checkpoints" / "sam2.1_hiera_large.pt"
    return (
        config,
        str(ckpt) if ckpt.exists() else None,
    )


def default_grounding_paths() -> tuple[str | None, str | None]:
    config = GROUNDING_DINO_DIR / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt = GROUNDED_SAM2_REPO / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
    return (
        str(config) if config.exists() else None,
        str(ckpt) if ckpt.exists() else None,
    )

@dataclass
class ServerState:
    pipeline: GroundedSAM2Pipeline
    default_box_threshold: float
    default_text_threshold: float
    default_min_box_area: float
    top_k: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class GroundedSAM2Server(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, state: ServerState):
        super().__init__(server_address, handler_cls)
        self.state = state


class GroundedSAM2RequestHandler(BaseHTTPRequestHandler):
    server: GroundedSAM2Server  # type: ignore[assignment]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        LOG.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._send_json({"status": "ok"})
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:  # noqa: N802
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
            value = body["top_k"]
            if value is not None:
                top_k = max(1, int(value))
            else:
                top_k = None

        start = time.perf_counter()
        with self.server.state.lock:
            detections = self.server.state.pipeline.predict(
                frame_bgr=frame,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                min_box_area=min_box_area,
                top_k=top_k,
            )
        latency_ms = (time.perf_counter() - start) * 1e3

        response = {
            "detections": [self._serialize_detection(det) for det in detections],
            "best": self._serialize_detection(detections[0]) if detections else None,
            "latency_ms": latency_ms,
        }
        self._send_json(response)

    def _serialize_detection(self, det: GroundingDINODetection) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bbox": det.bbox.tolist(),
            "confidence": det.confidence,
            "label": det.label,
        }
        if det.mask is not None:
            payload["mask_png"] = self._encode_mask(det.mask)
        return payload

    def _encode_mask(self, mask: np.ndarray) -> str | None:
        mask_u8 = (mask.astype(np.uint8)) * 255
        ok, buffer = cv2.imencode(".png", mask_u8)
        if not ok:
            LOG.warning("cv2.imencode failed for mask")
            return None
        return base64.b64encode(buffer).decode("utf-8")

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
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
        if "image_path" in body:
            path = Path(body["image_path"]).expanduser()
            if not path.exists():
                raise ValueError(f"image_path not found: {path}")
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read image from {path}")
            return frame
        raise ValueError("Provide 'image_b64' (recommended) or 'image_path'.")


def parse_args() -> argparse.Namespace:
    default_sam2_config, default_sam2_ckpt = default_groundedsam2_paths()
    default_grounding_config, default_grounding_ckpt = default_grounding_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8090, help="Bind port (default: 8090).")
    parser.add_argument(
        "--grounding-config",
        default=default_grounding_config,
        help=(
            "Path to the GroundingDINO config file."
            " Defaults to Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py when available."
        ),
    )
    parser.add_argument(
        "--grounding-checkpoint",
        default=default_grounding_ckpt,
        help=(
            "Path to the GroundingDINO checkpoint (.pth)."
            " Defaults to Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth when available."
        ),
    )
    parser.add_argument(
        "--sam2-model-config",
        default=default_sam2_config,
        help=(
            "Path to the SAM 2 config YAML file."
            " Provide paths relative to the `sam2` Python package (e.g. configs/sam2.1/sam2.1_hiera_l.yaml,"
            " located under Grounded-SAM-2/sam2/)."
        ),
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=default_sam2_ckpt,
        help=(
            "Path to the SAM 2 checkpoint file."
            " Defaults to Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt when present."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Torch device for both models (default: cuda).")
    parser.add_argument(
        "--no-caption-normalization",
        action="store_true",
        help="Do not lowercase / append periods to captions before inference.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable bfloat16 autocast even when running on CUDA.",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Use SAM 2 multimask_output=True and pick the best mask per detection.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.4, help="Default Grounding DINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.3, help="Default Grounding DINO text threshold.")
    parser.add_argument("--min-box-area", type=float, default=1_500.0, help="Default minimum bbox area in pixels.")
    parser.add_argument("--top-k", type=int, help="Maximum detections to evaluate per request.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level (default: INFO).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    sam2_config = args.sam2_model_config
    sam2_ckpt = args.sam2_checkpoint
    if sam2_config is None or sam2_ckpt is None:
        default_config, default_ckpt = default_groundedsam2_paths()
        sam2_config = sam2_config or default_config
        sam2_ckpt = sam2_ckpt or default_ckpt
    if sam2_config is None or sam2_ckpt is None:
        raise ValueError("Provide --sam2-model-config and --sam2-checkpoint (or place defaults under Grounded-SAM-2/).")

    grounding_config = args.grounding_config
    grounding_ckpt = args.grounding_checkpoint
    if grounding_config is None or grounding_ckpt is None:
        default_g_config, default_g_ckpt = default_grounding_paths()
        grounding_config = grounding_config or default_g_config
        grounding_ckpt = grounding_ckpt or default_g_ckpt
    if grounding_config is None or grounding_ckpt is None:
        raise ValueError(
            "Provide --grounding-config and --grounding-checkpoint (or keep the defaults under Grounded-SAM-2/)."
        )

    pipeline = GroundedSAM2Pipeline(
        grounding_config_path=grounding_config,
        grounding_checkpoint_path=grounding_ckpt,
        sam2_config_path=sam2_config,
        sam2_checkpoint_path=sam2_ckpt,
        device=args.device,
        normalize_caption=not args.no_caption_normalization,
        autocast=not args.disable_autocast,
        multimask_output=args.multimask_output,
    )

    state = ServerState(
        pipeline=pipeline,
        default_box_threshold=args.box_threshold,
        default_text_threshold=args.text_threshold,
        default_min_box_area=args.min_box_area,
        top_k=args.top_k,
    )

    server = GroundedSAM2Server((args.host, args.port), GroundedSAM2RequestHandler, state)
    LOG.info("Grounded SAM 2 server listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
