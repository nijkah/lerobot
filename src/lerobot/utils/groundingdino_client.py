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
        for det in detections_field:
            try:
                bbox = np.array(det["bbox"], dtype=float)
                confidence = float(det["confidence"])
                label = str(det["label"])
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Skipping malformed detection entry %s: %s", det, exc)
                continue
            results.append(GroundingDINODetection(bbox=bbox, confidence=confidence, label=label))

        # Ensure descending order (server already sorts, but clients rely on this)
        results.sort(key=lambda item: item.confidence, reverse=True)
        return results
