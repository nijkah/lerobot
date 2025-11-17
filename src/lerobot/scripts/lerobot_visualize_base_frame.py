#!/usr/bin/env python

"""Live overlay of the robot base frame estimated from hand-eye calibration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from lerobot.utils.rotation import Rotation

LOG = logging.getLogger(__name__)


def _load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    if file_path.suffix.lower() == ".npz":
        payload = np.load(file_path)
        K = np.array(payload["K"], dtype=np.float64)
        D = np.array(payload["D"], dtype=np.float64)
    else:
        payload = json.loads(file_path.read_text())
        matrix = payload.get("K") or payload.get("camera_matrix")
        if matrix is None:
            raise KeyError("Intrinsics JSON missing 'K' or 'camera_matrix'.")
        dist = payload.get("D") or payload.get("distortion_coefficients")
        if dist is None:
            raise KeyError("Intrinsics JSON missing 'D' or 'distortion_coefficients'.")
        K = np.array(matrix, dtype=np.float64)
        D = np.array(dist, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"Expected 3x3 intrinsics matrix, got {K.shape}.")
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    return K, D


def _load_transform(payload: list[list[float]] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.array(payload, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {matrix.shape}.")
    return matrix


def _format_pose(T: np.ndarray) -> str:
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return (
        f"pos(m)=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
        f"rotvec(rad)=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f})"
    )


def _point_in_bounds(point: np.ndarray, width: int, height: int, margin: float = 0.1) -> bool:
    min_x = -margin * width
    max_x = (1.0 + margin) * width
    min_y = -margin * height
    max_y = (1.0 + margin) * height
    return bool(min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)


def _draw_base_axes(
    frame: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    camera_to_base: np.ndarray,
    axis_length: float,
) -> list[str] | None:
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=np.float64,
    )
    rmat = camera_to_base[:3, :3]
    tvec = camera_to_base[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rmat)
    dist = None if D is None else D.reshape(-1, 1)
    proj, _ = cv2.projectPoints(base_points, rvec, tvec, K.astype(np.float64), dist)
    proj = proj.reshape(-1, 2)
    if not np.isfinite(proj).all():
        return None
    if tvec[2, 0] <= 1e-6:
        LOG.debug("Base origin lies behind the camera (z=%.4f).", float(tvec[2, 0]))
        return None
    height, width = frame.shape[:2]
    origin_vals = proj[0]
    if not _point_in_bounds(origin_vals, width, height):
        LOG.debug(
            "Projected base origin (%.1f, %.1f) outside image bounds (w=%d, h=%d).",
            origin_vals[0],
            origin_vals[1],
            width,
            height,
        )
        return None
    origin = tuple(int(round(v)) for v in origin_vals)
    # OpenCV expects BGR tuples; enforce X=red, Y=green, Z=blue.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for idx, color in enumerate(colors, start=1):
        end_vals = proj[idx]
        if not np.isfinite(end_vals).all():
            return None
        end = tuple(int(round(v)) for v in end_vals)
        cv2.arrowedLine(frame, origin, end, color, 2, tipLength=0.2)
    return [
        f"Base origin px=({origin_vals[0]:.1f}, {origin_vals[1]:.1f})",
        f"Base x px=({proj[1, 0]:.1f}, {proj[1, 1]:.1f})",
        f"Base y px=({proj[2, 0]:.1f}, {proj[2, 1]:.1f})",
        f"Base z px=({proj[3, 0]:.1f}, {proj[3, 1]:.1f})",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True, help="Path to hand-eye calibration JSON.")
    parser.add_argument(
        "--camera-source",
        default="0",
        help="Index or path for OpenCV VideoCapture (default: 0).",
    )
    parser.add_argument("--intrinsics", help="Override intrinsics path; defaults to calibration entry.")
    parser.add_argument("--axis-length", type=float, default=0.1, help="Axis length in meters.")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    calibration = json.loads(calibration_path.read_text())
    camera_to_base = _load_transform(calibration["camera_to_base"], "camera_to_base")

    intrinsics_path = args.intrinsics or calibration.get("intrinsics_path")
    if not intrinsics_path:
        raise ValueError("Intrinsics path must be provided either via --intrinsics or calibration JSON.")

    K, D = _load_intrinsics(str(intrinsics_path))
    cap_source: str | int
    try:
        cap_source = int(args.camera_source)
    except ValueError:
        cap_source = args.camera_source

    capture = cv2.VideoCapture(cap_source)
    if args.width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera source: {args.camera_source}")

    overlay_pose = _format_pose(camera_to_base)

    window_title = "Hand-Eye Base Visualization"
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                continue
            axis_len = float(args.axis_length)
            axes_lines = _draw_base_axes(frame, K, D, camera_to_base, axis_len)

            overlay_text = f"Camera->Base {overlay_pose}" if overlay_pose else "Camera->Base unavailable"
            cv2.putText(
                frame,
                overlay_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if overlay_pose else (0, 0, 255),
                1,
            )
            if axes_lines:
                for idx, line in enumerate(axes_lines):
                    cv2.putText(
                        frame,
                        line,
                        (20, 55 + idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                    )
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
