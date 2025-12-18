#!/usr/bin/env python

"""HTTP client helper for the GroundingDINO inference server."""

from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

LOG = logging.getLogger(__name__)


@dataclass
class GroundingDINODetection:
    """Single detection result returned by GroundingDINO."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    label: str
    mask: np.ndarray | None = None  # optional HxW bool mask


class GroundingDINOHTTPClient:
    """Minimal JSON-over-HTTP client for the GroundingDINO inference server."""

    def __init__(self, server_url: str, timeout: float = 10.0):
        endpoint = server_url.rstrip("/")
        if not endpoint.endswith("/predict"):
            endpoint = f"{endpoint}/predict"
        self.endpoint = endpoint
        self.timeout = float(timeout)

    def predict(
        self,
        frame_bgr: np.ndarray,
        caption: str,
        *,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
        top_k: int | None = None,
    ) -> list[GroundingDINODetection]:
        if frame_bgr is None:
            raise ValueError("frame_bgr cannot be None")
        ok, buffer = cv2.imencode(".jpg", frame_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed for provided frame.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        payload: dict[str, Any] = {
            "caption": caption,
            "image_b64": encoded,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "min_box_area": min_box_area,
        }
        if top_k is not None:
            payload["top_k"] = int(top_k)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = resp.read()
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"GroundingDINO server error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach GroundingDINO server: {exc.reason}") from exc

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise RuntimeError(f"Invalid JSON response from server: {exc}") from exc

        detections_field = parsed.get("detections", [])
        results: list[GroundingDINODetection] = []
        frame_shape = frame_bgr.shape[:2]
        for det in detections_field:
            try:
                bbox = np.array(det["bbox"], dtype=float)
                confidence = float(det["confidence"])
                label = str(det["label"])
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping malformed detection entry %s: %s", det, exc)
                continue
            mask = self._decode_mask(det, frame_shape)
            results.append(GroundingDINODetection(bbox=bbox, confidence=confidence, label=label, mask=mask))

        # Ensure descending order (server already sorts, but clients rely on this)
        results.sort(key=lambda item: item.confidence, reverse=True)
        return results

    def _decode_mask(self, det: dict[str, Any], frame_shape: tuple[int, int]) -> np.ndarray | None:
        payload = det.get("mask") or det.get("mask_b64") or det.get("mask_png")
        if payload is None:
            return None
        mask: np.ndarray | None = None
        if isinstance(payload, str):
            try:
                raw = base64.b64decode(payload, validate=True)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to base64 decode mask: %s", exc)
                return None
            buffer = np.frombuffer(raw, dtype=np.uint8)
            mask = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                LOG.warning("cv2.imdecode returned None for mask payload")
                return None
        elif isinstance(payload, dict) and {"counts", "size"} <= payload.keys():
            try:
                from pycocotools import mask as coco_mask  # type: ignore[import-not-found]
            except ImportError:
                LOG.warning("pycocotools not installed; cannot decode RLE mask.")
                return None
            try:
                mask = coco_mask.decode({"counts": payload["counts"], "size": payload["size"]})
                if mask.ndim == 3:
                    mask = mask[..., 0]
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to decode RLE mask: %s", exc)
                return None
        if mask is None:
            return None
        if mask.shape != frame_shape:
            mask = cv2.resize(mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
        if mask.dtype != bool:
            threshold = 0.5 if mask.max() <= 1 else 127
            mask = mask > threshold
        return mask
