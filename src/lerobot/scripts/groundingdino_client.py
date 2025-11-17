#!/usr/bin/env python

"""Utility client to query a remote GroundingDINO inference server."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.utils.groundingdino_client import GroundingDINOHTTPClient

LOG = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True, help="Base URL of the GroundingDINO server (e.g., http://gpu:8080).")
    parser.add_argument("--caption", required=True, help="Text prompt describing the target object.")
    parser.add_argument("--image-path", help="Path to an image file to send.")
    parser.add_argument("--camera-source", help="OpenCV camera index/path for live capture.")
    parser.add_argument("--width", type=int, help="Camera capture width override.")
    parser.add_argument("--height", type=int, help="Camera capture height override.")
    parser.add_argument("--fps", type=float, help="Camera FPS hint when looping.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--min-box-area", type=float, default=1_500.0, help="GroundingDINO minimum box area.")
    parser.add_argument("--top-k", type=int, help="Maximum returned detections.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument("--loop", action="store_true", help="Continuously stream frames from the camera.")
    parser.add_argument("--show", action="store_true", help="Render detections in an OpenCV window.")
    parser.add_argument("--save-json", help="Optional path to dump raw response JSON.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def _load_image(image_path: str) -> np.ndarray:
    path = Path(image_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Failed to read image from {path}")
    return frame


def _open_capture(source: str, width: int | None, height: int | None, fps: float | None) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap_source: int | str = idx
    except (TypeError, ValueError):
        cap_source = source
    capture = cv2.VideoCapture(cap_source)
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera source: {source}")
    return capture


def _draw_detections(frame: np.ndarray, detections) -> None:
    for det in detections:
        x1, y1, x2, y2 = det.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)


def _print_detections(detections) -> None:
    if not detections:
        LOG.info("No detections.")
        return
    LOG.info("Detections:")
    for idx, det in enumerate(detections):
        bbox = ", ".join(f"{v:.1f}" for v in det.bbox)
        LOG.info("  %d) %s conf=%.3f bbox=[%s]", idx + 1, det.label, det.confidence, bbox)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if not args.image_path and not args.camera_source:
        raise ValueError("Provide either --image-path or --camera-source.")
    if args.image_path and args.camera_source:
        raise ValueError("Use either --image-path or --camera-source, not both.")

    client = GroundingDINOHTTPClient(server_url=args.server_url, timeout=args.timeout)

    if args.image_path:
        frame = _load_image(args.image_path)
        detections = client.predict(
            frame_bgr=frame,
            caption=args.caption,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            min_box_area=args.min_box_area,
            top_k=args.top_k,
        )
        _print_detections(detections)
        if args.show:
            vis = frame.copy()
            _draw_detections(vis, detections)
            cv2.imshow("GroundingDINO client", vis)
            LOG.info("Close the window or press any key to exit.")
            cv2.waitKey(0)
        if args.save_json:
            payload = [
                {"bbox": det.bbox.tolist(), "confidence": det.confidence, "label": det.label}
                for det in detections
            ]
            Path(args.save_json).write_text(json.dumps(payload, indent=2))
        return

    capture = _open_capture(args.camera_source, args.width, args.height, args.fps)
    window = "GroundingDINO client"
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                LOG.warning("Failed to grab frame.")
                if not args.loop:
                    break
                continue
            start = time.perf_counter()
            detections = client.predict(
                frame_bgr=frame,
                caption=args.caption,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                min_box_area=args.min_box_area,
                top_k=args.top_k,
            )
            latency = (time.perf_counter() - start) * 1e3
            LOG.info("Latency %.1f ms", latency)
            _print_detections(detections)
            if args.show:
                vis = frame.copy()
                _draw_detections(vis, detections)
                info = f"{args.caption} | latency {latency:.1f} ms"
                cv2.putText(vis, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow(window, vis)
                key = cv2.waitKey(1 if args.loop else 0) & 0xFF
                if key == ord("q"):
                    break
            if not args.loop:
                break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
