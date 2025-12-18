#!/usr/bin/env python

"""Live overlay of the robot base frame estimated from hand-eye calibration.

Press `g` to trigger GroundingDINO- or Grounded SAM 2-driven grasp planning when a target object
prompt is provided. This mode projects the detected pixel onto the robot base frame, solves IK,
and commands a slow, safe Cartesian approach/grasp/retreat sequence similar to
``lerobot_scripted_cartesian.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.groundingdino_client import (
    GroundingDINOHTTPClient,
    GroundingDINODetection,
)
from lerobot.utils.groundedsam2 import GroundedSAM2Pipeline, default_groundedsam2_paths
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
GROUNDING_REPO = REPO_ROOT / "GroundingDINO"

def set_focus_on_camera(cap: cv2.VideoCapture, focus_value: int = 312) -> None:
    if not cap.set(cv2.CAP_PROP_FOCUS, 1):
        logging.warning("Camera driver did not accept CAP_PROP_FOCUS command.")
    time.sleep(0.15)
    ret, frame = cap.read()
    if not ret:
        logging.warning("Frame grab failed after setting CAP_PROP_FOCUS.")

    if not cap.set(cv2.CAP_PROP_FOCUS, focus_value):
        logging.warning("Camera driver did not accept CAP_PROP_FOCUS command.")
    time.sleep(0.15)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Frame grab failed after setting manual focus.")


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
    base_to_camera: np.ndarray,
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
    rmat = base_to_camera[:3, :3]
    tvec = base_to_camera[:3, 3].reshape(3, 1)
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


def _draw_pose_axes(
    frame: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    base_to_camera: np.ndarray,
    pose_base: np.ndarray,
    axis_length: float,
    label: str,
    color_scale: float = 0.6,
) -> list[str] | None:
    origin = pose_base[:3, 3]
    axes = pose_base[:3, :3]
    axis_points = np.array(
        [
            origin,
            origin + axes[:, 0] * axis_length,
            origin + axes[:, 1] * axis_length,
            origin + axes[:, 2] * axis_length,
        ],
        dtype=np.float64,
    )
    rmat = base_to_camera[:3, :3]
    tvec = base_to_camera[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rmat)
    dist = None if D is None else D.reshape(-1, 1)
    proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K.astype(np.float64), dist)
    proj = proj.reshape(-1, 2)
    if not np.isfinite(proj).all():
        return None
    height, width = frame.shape[:2]
    origin_vals = proj[0]
    if not _point_in_bounds(origin_vals, width, height):
        return None
    origin_px = tuple(int(round(v)) for v in origin_vals)
    colors = [
        (0, 0, int(255 * color_scale)),
        (0, int(255 * color_scale), 0),
        (int(255 * color_scale), 0, 0),
    ]
    for idx, color in enumerate(colors, start=1):
        end_vals = proj[idx]
        if not np.isfinite(end_vals).all():
            continue
        end_px = tuple(int(round(v)) for v in end_vals)
        cv2.arrowedLine(frame, origin_px, end_px, color, 2, tipLength=0.2)
    return [
        f"{label} origin px=({origin_vals[0]:.1f}, {origin_vals[1]:.1f})",
        f"{label} x px=({proj[1, 0]:.1f}, {proj[1, 1]:.1f})",
        f"{label} y px=({proj[2, 0]:.1f}, {proj[2, 1]:.1f})",
        f"{label} z px=({proj[3, 0]:.1f}, {proj[3, 1]:.1f})",
    ]


def _str2bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to parse boolean from '{value}'")


def _parse_vector3(text: str, field: str) -> np.ndarray:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"{field} must be a JSON list of 3 floats") from exc
    if not isinstance(payload, Sequence) or len(payload) != 3:
        raise argparse.ArgumentTypeError(f"{field} must contain exactly 3 elements")
    return np.array([float(payload[0]), float(payload[1]), float(payload[2])], dtype=float)


def _parse_float_list(text: str, field: str) -> np.ndarray:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"{field} must be a JSON list of floats") from exc
    if not isinstance(payload, Sequence) or not payload:
        raise argparse.ArgumentTypeError(f"{field} must be a non-empty JSON list")
    try:
        return np.array([float(v) for v in payload], dtype=float)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"{field} must only contain numeric values") from exc


def _default_grounding_paths() -> tuple[str | None, str | None]:
    config_path = GROUNDING_REPO / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
    ckpt_path = REPO_ROOT / "weights" / "groundingdino_swint_ogc.pth"
    return (
        str(config_path) if config_path.exists() else None,
        str(ckpt_path) if ckpt_path.exists() else None,
    )


class LocalGroundingDinoDetector:
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
    ) -> GroundingDINODetection | None:
        detections, phrases = self.model.predict_with_caption(
            image=frame_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        if detections.xyxy.size == 0:
            return None

        best_idx = -1
        best_score = -np.inf
        best_box: np.ndarray | None = None
        for idx, (bbox, score, phrase) in enumerate(
            zip(detections.xyxy, detections.confidence, phrases, strict=False)
        ):
            x1, y1, x2, y2 = bbox.astype(float)
            area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
            if area < min_box_area:
                continue
            if score > best_score:
                best_score = float(score)
                best_idx = idx
                best_box = np.array([x1, y1, x2, y2], dtype=float)
                best_phrase = phrase

        if best_idx < 0 or best_box is None:
            return None
        return GroundingDINODetection(bbox=best_box, confidence=best_score, label=best_phrase)


class LocalGroundedSAM2Detector:
    """Runs the Grounded SAM 2 pipeline locally for detections with masks."""

    def __init__(
        self,
        *,
        grounding_model_id: str,
        sam2_config_path: str,
        sam2_checkpoint_path: str,
        device: str,
        hf_revision: str | None,
        hf_cache_dir: str | None,
    ) -> None:
        self.pipeline = GroundedSAM2Pipeline(
            grounding_model_id=grounding_model_id,
            sam2_config_path=sam2_config_path,
            sam2_checkpoint_path=sam2_checkpoint_path,
            device=device,
            hf_revision=hf_revision,
            hf_cache_dir=hf_cache_dir,
        )

    def predict(
        self,
        frame_bgr: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
    ) -> GroundingDINODetection | None:
        detections = self.pipeline.predict(
            frame_bgr=frame_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_box_area=min_box_area,
            top_k=1,
        )
        return detections[0] if detections else None


class RemoteGroundingDetector:
    """Queries a remote GroundingDINO-compatible server via HTTP (incl. SAM2)."""

    def __init__(self, client: GroundingDINOHTTPClient):
        self.client = client

    def predict(
        self,
        frame_bgr: np.ndarray,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        min_box_area: float,
    ) -> GroundingDINODetection | None:
        detections = self.client.predict(
            frame_bgr=frame_bgr,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            min_box_area=min_box_area,
        )
        return detections[0] if detections else None


class FrameUndistorter:
    """Lazy holder for OpenCV undistortion maps."""

    def __init__(self, K: np.ndarray, D: np.ndarray | None, image_size: tuple[int, int]) -> None:
        self.enabled = bool(D is not None and np.any(np.abs(D) > 1e-9))
        self.original_K = np.array(K, dtype=float)
        self.camera_matrix = self.original_K.copy()
        self.dist_coeffs: np.ndarray | None = None if self.enabled else (None if D is None else np.array(D, dtype=float))
        self.map1: np.ndarray | None = None
        self.map2: np.ndarray | None = None
        if not self.enabled:
            return
        height = int(image_size[1])
        width = int(image_size[0])
        self.camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.original_K,
            D,
            (width, height),
            0.0,
            (width, height),
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.original_K,
            D,
            None,
            self.camera_matrix,
            (width, height),
            cv2.CV_32FC1,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled or self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


class VisionGraspController:
    """Handles detection -> base frame projection -> IK -> slow motion execution."""

    def __init__(self, args: argparse.Namespace, K: np.ndarray, D: np.ndarray | None, camera_to_base: np.ndarray):
        self.args = args
        self.enable_detection = bool(args.target_object)
        if self.enable_detection and args.target_plane_z is None:
            raise ValueError("--target-plane-z is required when --target-object is set.")
        if self.enable_detection:
            if args.grounding_server_url:
                client = GroundingDINOHTTPClient(
                    server_url=args.grounding_server_url,
                    timeout=args.grounding_server_timeout,
                )
                self.detector = RemoteGroundingDetector(client)
            elif args.use_grounded_sam2:
                self.detector = LocalGroundedSAM2Detector(
                    grounding_model_id=args.sam2_grounding_model,
                    sam2_config_path=args.sam2_model_config,
                    sam2_checkpoint_path=args.sam2_checkpoint,
                    device=args.grounding_device,
                    hf_revision=args.sam2_hf_revision,
                    hf_cache_dir=args.sam2_hf_cache_dir,
                )
            else:
                self.detector = LocalGroundingDinoDetector(
                    config_path=args.grounding_config,
                    checkpoint_path=args.grounding_checkpoint,
                    device=args.grounding_device,
                )
        else:
            self.detector = None
        self.K = np.array(K, dtype=float)
        self.D = None if D is None else np.array(D, dtype=float)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_to_base = np.array(camera_to_base, dtype=float)
        self.base_to_camera = np.linalg.inv(self.camera_to_base)
        self.camera_origin = self.camera_to_base[:3, 3]
        self.camera_rot = self.camera_to_base[:3, :3]
        self.plane_z = float(args.target_plane_z) if args.target_plane_z is not None else None
        self.table_plane_normal_base = np.array([0.0, 0.0, 1.0], dtype=float)
        self.ee_axis_length = float(getattr(args, "ee_axis_length", 0.05))
        self.ee_pose_text = "EE pose unavailable"
        if self.enable_detection:
            self.status_lines: list[str] = [
                f"Vision grasp ready. Prompt='{args.target_object}'. Press 'g' to search."
            ]
        else:
            self.status_lines = ["EE visualization active. Press 'q' to exit."]
        self.last_detection: tuple[GroundingDINODetection, np.ndarray] | None = None
        self.segmentation_alpha = float(np.clip(args.segmentation_alpha, 0.0, 1.0))
        self.segmentation_color = tuple(
            int(v) for v in _parse_vector3(args.segmentation_color, "--segmentation-color")
        )
        self.require_mask_for_overlay = bool(args.require_mask_for_overlay)
        self.show_grasp_marker = not bool(args.hide_grasp_marker)
        self.joint_names: list[str] = []
        self.name_to_idx: dict[str, int] = {}
        self.gripper_key = f"{args.gripper_channel}.pos"
        self.gripper_index: int | None = None
        self.robot = None
        self.kinematics = None
        self.default_pregrasp_rotation = Rotation.from_euler_xyz([0.0, -180.0, 0.0], degrees=True).as_matrix()
        self.pregrasp_rotation = self.default_pregrasp_rotation.copy()
        self.wrist_roll_angle_rad: float | None = None
        self.pre_y_offset = float(args.pre_y_offset)
        self.move_to_initial_pose = not bool(args.skip_initial_pose)
        self.initial_joint_targets = (
            _parse_float_list(args.initial_joints, "--initial-joints") if args.initial_joints else None
        )
        self.trajectory_mode = args.trajectory_mode

        if not args.robot_type:
            raise ValueError("Specify --robot.type/--robot.port to visualize and command the robot.")
        robot_config = _build_robot_config(args)
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        self.joint_names = list(self.robot.bus.motors.keys())  # type: ignore[attr-defined]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.joint_names)}
        if self.initial_joint_targets is not None and len(self.initial_joint_targets) != len(self.joint_names):
            raise ValueError(
                f"--initial-joints expects {len(self.joint_names)} entries, got {len(self.initial_joint_targets)}"
            )
        if self.gripper_key.removesuffix(".pos") not in self.name_to_idx:
            LOG.warning(
                "Gripper channel '%s' missing from robot motors; will keep gripper untouched.",
                self.gripper_key,
            )
        else:
            self.gripper_index = self.name_to_idx[self.gripper_key.removesuffix(".pos")]

        urdf_path = args.urdf_path
        target_frame = args.target_frame_name
        if not urdf_path:
            urdf_path = getattr(self.robot.config, "urdf_path", None)
        if not target_frame:
            target_frame = getattr(self.robot.config, "target_frame_name", "gripper_frame_link")
        if not urdf_path:
            raise ValueError("Provide --urdf-path or set robot config's urdf_path.")
        self.kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=self.joint_names,
        )
        if args.zero_offset_json_path:
            path = Path(args.zero_offset_json_path)
            if path.exists():
                self.kinematics.load_joint_zero_offset(str(path))
                LOG.info("Loaded joint zero offset from %s", path)
            else:
                LOG.warning("Zero offset file %s not found.", path)

        if self.move_to_initial_pose:
            try:
                self._command_initial_pose()
                self.status_lines.append("Moved to initial joint pose.")
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to move to initial pose: %s", exc)
                self.status_lines.append(f"Initial pose move failed: {exc}")

    def shutdown(self):
        if self.robot and getattr(self.robot, "is_connected", False):
            self.robot.disconnect()

    def draw_overlay(self, frame: np.ndarray):
        if self.last_detection is None:
            return
        detection, world_point = self.last_detection
        mask = self._prepare_mask(frame.shape, detection)
        if mask is not None and self.segmentation_alpha > 0:
            self._apply_segmentation_overlay(frame, mask)

        x1, y1, x2, y2 = detection.bbox.astype(int)
        color = tuple(int(c) for c in self.segmentation_color)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        center = self._project_base_point_to_pixel(world_point, frame.shape[1], frame.shape[0])
        if center is None:
            center = (int(round(0.5 * (x1 + x2))), int(round(0.5 * (y1 + y2))))
        if self.show_grasp_marker:
            cv2.drawMarker(frame, center, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=16, thickness=2)
            cv2.circle(frame, center, 4, color, -1)

        coord = f"Base xyz (m): {world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f}"
        text_origin = (max(10, center[0] - 100), min(frame.shape[0] - 10, center[1] + 30))
        cv2.putText(frame, coord, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_ee_pose(self, frame: np.ndarray) -> list[str]:
        if self.robot is None or self.kinematics is None:
            return []
        try:
            obs = self.robot.get_observation()
            pose = self.kinematics.forward_kinematics_from_observation(obs)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Failed to compute EE pose: %s", exc)
            return []
        axis_lines = _draw_pose_axes(
            frame,
            self.K,
            self.D,
            self.base_to_camera,
            pose,
            self.ee_axis_length,
            label="EE",
        )
        pose_line = f"EE pose base: {_format_pose(pose)}"
        self.ee_pose_text = pose_line
        lines: list[str] = []
        if axis_lines:
            lines.extend(axis_lines)
        lines.append(pose_line)
        return lines

    def handle_grasp(self, frame_bgr: np.ndarray, window_title: str | None = None):
        if not self.enable_detection or self.detector is None:
            self.status_lines = ["Detection disabled; enable --target-object to run grasp planning."]
            return
        self.status_lines = ["Detecting target object..."]
        detection = self.detector.predict(
            frame_bgr=frame_bgr,
            caption=self.args.target_object,
            box_threshold=self.args.box_threshold,
            text_threshold=self.args.text_threshold,
            min_box_area=self.args.min_box_area,
        )
        if detection is None:
            self.status_lines = [f"No detection for '{self.args.target_object}'. Try adjusting thresholds."]
            self.last_detection = None
            return
        try:
            world_point = self._pixel_to_base(detection.bbox)
        except ValueError as exc:
            self.status_lines = [f"Projection failed: {exc}"]
            self.last_detection = None
            return

        self.last_detection = (detection, world_point)
        roll_angle = self._estimate_wrist_roll(frame_bgr.shape, detection)
        wrist_roll_msg = self._apply_wrist_roll_angle(roll_angle)
        self.status_lines = [
            f"Detected '{detection.label}' (score={detection.confidence:.2f})",
            "Visualizing grasp candidate...",
        ]
        self._visualize_detection(frame_bgr, window_title)

        if not self.robot or not self.kinematics:
            self.status_lines = [
                f"Detected '{detection.label}' but robot is unavailable.",
                "Provide --robot.type/--robot.port to enable IK and motion.",
            ]
            if wrist_roll_msg:
                self.status_lines.append(wrist_roll_msg)
            return

        self.status_lines = [
            f"Base xyz (m): {world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f}",
            "Solving IK and executing grasp...",
        ]

        try:
            self._execute_grasp(world_point)
            self.status_lines = [
                f"Grasp sequence executed at x={world_point[0]:.3f}, y={world_point[1]:.3f}, z={world_point[2]:.3f}",
                "Press 'g' to run again or 'q' to quit.",
            ]
            if wrist_roll_msg:
                self.status_lines.append(wrist_roll_msg)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Failed to execute grasp: %s", exc)
            self.status_lines = [f"Grasp execution failed: {exc}"]
            if wrist_roll_msg:
                self.status_lines.append(wrist_roll_msg)

    def _pixel_to_base(self, bbox: np.ndarray) -> np.ndarray:
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        return self._pixel_to_base_point((cx, cy))

    def _pixel_to_base_point(self, pixel_xy: tuple[float, float]) -> np.ndarray:
        if self.plane_z is None:
            raise ValueError("target_plane_z must be provided to project pixels to the base frame.")
        pixel = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=float)
        ray_cam = self._pixel_to_camera_ray(pixel)
        intersection_cam = self._intersect_camera_ray_with_table(ray_cam)
        return self._camera_point_to_base(intersection_cam)

    def _pixel_to_camera_ray(self, pixel_homo: np.ndarray) -> np.ndarray:
        direction = self.K_inv @ pixel_homo
        norm = np.linalg.norm(direction)
        if norm <= 1e-9:
            raise ValueError("Invalid pixel direction; cannot normalize camera ray.")
        return direction / norm

    def _intersect_camera_ray_with_table(self, ray_cam: np.ndarray) -> np.ndarray:
        normal_cam = self.base_to_camera[:3, :3] @ self.table_plane_normal_base
        plane_point_base = np.array([0.0, 0.0, self.plane_z], dtype=float)
        plane_point_cam = self._transform_point_base_to_camera(plane_point_base)
        denom = float(np.dot(normal_cam, ray_cam))
        if abs(denom) < 1e-6:
            raise ValueError("Camera ray is parallel to table plane.")
        d = float(np.dot(normal_cam, plane_point_cam))
        t = d / denom
        if t <= 0:
            raise ValueError("Table-plane intersection lies behind the camera origin.")
        return ray_cam * t

    def _camera_point_to_base(self, point_cam: np.ndarray) -> np.ndarray:
        point_h = np.ones(4, dtype=float)
        point_h[:3] = point_cam
        return (self.camera_to_base @ point_h)[:3]

    def _transform_point_base_to_camera(self, point_base: np.ndarray) -> np.ndarray:
        rotation = self.base_to_camera[:3, :3]
        translation = self.base_to_camera[:3, 3]
        return rotation @ point_base + translation

    def _project_base_point_to_pixel(
        self,
        point_base: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int] | None:
        point_h = np.ones(4, dtype=float)
        point_h[:3] = point_base
        point_cam = (self.base_to_camera @ point_h)[:3]
        if point_cam[2] <= 1e-6:
            return None
        normalized = point_cam / point_cam[2]
        pixel = self.K @ normalized
        if not np.isfinite(pixel[:2]).all():
            return None
        if not _point_in_bounds(pixel[:2], frame_width, frame_height):
            return None
        return (int(round(pixel[0])), int(round(pixel[1])))

    def _make_pose(self, position: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = position
        pose[:3, :3] = rotation if rotation is not None else self.pregrasp_rotation
        return pose

    def _pose_from_joints(self, joints: np.ndarray) -> np.ndarray | None:
        if not self.kinematics:
            return None
        try:
            return self.kinematics.forward_kinematics(joints)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Failed to compute FK from joints: %s", exc)
            return None

    def _command_initial_pose(self) -> None:
        current = self._read_joint_positions()
        if self.initial_joint_targets is not None:
            target = np.array(self.initial_joint_targets, dtype=float)
        else:
            target = np.zeros(len(self.joint_names), dtype=float)
        self._send_joint_trajectory(current, target)

    def _estimate_wrist_roll(
        self,
        frame_shape: tuple[int, int, int],
        detection: GroundingDINODetection,
    ) -> float | None:
        if self.plane_z is None:
            return None
        mask = self._prepare_mask(frame_shape, detection)
        if mask is None or not mask.any():
            return None
        base_points = self._mask_points_to_base_plane(mask)
        if base_points.shape[0] < 3:
            return None
        base_points32 = base_points.astype(np.float32)
        rect = cv2.minAreaRect(base_points32)
        box = cv2.boxPoints(rect)
        edges = np.roll(box, -1, axis=0) - box
        lengths = np.linalg.norm(edges, axis=1)
        major_idx = int(np.argmax(lengths))
        major_len = float(lengths[major_idx])
        if major_len <= 1e-6:
            return None
        direction = edges[major_idx] / major_len
        angle = float(np.arctan2(direction[1], direction[0]))
        angle += float(np.pi / 2.0)
        angle = float((angle + np.pi) % (2.0 * np.pi) - np.pi)
        return angle

    def _mask_points_to_base_plane(self, mask: np.ndarray, max_samples: int = 400) -> np.ndarray:
        coords = np.column_stack(np.nonzero(mask))
        if coords.size == 0:
            return np.empty((0, 2), dtype=float)
        if coords.shape[0] > max_samples:
            step = max(1, coords.shape[0] // max_samples)
            coords = coords[::step]
        points: list[np.ndarray] = []
        for row, col in coords:
            try:
                base_point = self._pixel_to_base_point((float(col), float(row)))
            except Exception:  # noqa: BLE001
                continue
            points.append(base_point[:2])
        if not points:
            return np.empty((0, 2), dtype=float)
        return np.asarray(points, dtype=float)

    def _apply_wrist_roll_angle(self, angle_rad: float | None) -> str:
        if angle_rad is None:
            self.pregrasp_rotation = self.default_pregrasp_rotation.copy()
            self.wrist_roll_angle_rad = None
            return "Wrist roll: default orientation"
        target_angle = float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)
        if self.wrist_roll_angle_rad is not None:
            delta = target_angle - self.wrist_roll_angle_rad
            delta = float((delta + np.pi) % (2.0 * np.pi) - np.pi)
            target_angle = self.wrist_roll_angle_rad + delta
        target_angle = float((target_angle + np.pi) % (2.0 * np.pi) - np.pi)
        roll_rot = Rotation.from_euler_xyz([0.0, 0.0, target_angle], degrees=False).as_matrix()
        self.pregrasp_rotation = roll_rot @ self.default_pregrasp_rotation
        self.wrist_roll_angle_rad = target_angle
        return f"Wrist roll: {np.degrees(target_angle):.1f} deg"

    def _execute_grasp(self, world_point: np.ndarray):
        pre = world_point.copy()
        pre[2] += self.args.approach_offset
        pre_target = pre.copy()
        pre_target[1] += self.pre_y_offset
        grasp = world_point.copy()
        grasp[2] += self.args.grasp_offset
        retreat = world_point.copy()
        retreat[2] += self.args.retreat_offset

        current = self._read_joint_positions()
        current_pose = self._pose_from_joints(current)
        if self.trajectory_mode == "joint":
            self._execute_joint_trajectory_mode(current, pre_target, grasp, retreat)
            return
        xy_align = pre.copy()
        if current_pose is not None:
            current_z = float(current_pose[2, 3])
            xy_align[2] = max(current_z, pre[2])
        xy_rotation = current_pose[:3, :3] if current_pose is not None else None
        xy_pose = self._make_pose(xy_align, rotation=xy_rotation)
        current = self._move_to_pose(current, xy_pose, self.args.gripper_open_pos, hold=0.0)
        current = self._move_to_pose(
            current,
            self._make_pose(pre_target),
            self.args.gripper_open_pos,
            hold=self.args.pregrasp_hold_s,
        )
        current = self._move_to_pose(current, self._make_pose(grasp), self.args.gripper_open_pos, hold=0.0)
        current = self._move_to_pose(current, self._make_pose(grasp), self.args.gripper_close_pos, hold=self.args.grasp_hold_s)
        current = self._move_to_pose(current, self._make_pose(retreat), self.args.gripper_close_pos, hold=self.args.retreat_hold_s)
        # Open gripper after retreat if requested
        if self.args.reopen_after_retreat:
            self._send_joint_trajectory(current, self._set_gripper_target(current.copy(), self.args.gripper_open_pos))

    def _execute_joint_trajectory_mode(
        self,
        current: np.ndarray,
        pre: np.ndarray,
        grasp: np.ndarray,
        retreat: np.ndarray,
    ) -> None:
        sequence = [
            (self._make_pose(pre), self.args.gripper_open_pos, self.args.pregrasp_hold_s),
            (self._make_pose(grasp), self.args.gripper_open_pos, 0.0),
            (self._make_pose(grasp), self.args.gripper_close_pos, self.args.grasp_hold_s),
            (self._make_pose(retreat), self.args.gripper_close_pos, self.args.retreat_hold_s),
        ]
        for pose, gripper_target, hold in sequence:
            current = self._move_to_pose(current, pose, gripper_target, hold)
        if self.args.reopen_after_retreat:
            self._send_joint_trajectory(
                current,
                self._set_gripper_target(current.copy(), self.args.gripper_open_pos),
            )

    def _read_joint_positions(self) -> np.ndarray:
        assert self.robot is not None
        if hasattr(self.robot, "bus"):
            joints = self.robot.bus.sync_read("Present_Position")  # type: ignore[attr-defined]
            return np.array([float(joints[name]) for name in self.joint_names], dtype=float)
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{name}.pos"]) for name in self.joint_names], dtype=float)

    def _set_gripper_target(self, joints: np.ndarray, value: float) -> np.ndarray:
        if self.gripper_index is not None:
            joints[self.gripper_index] = value
        return joints

    def _visualize_detection(self, frame: np.ndarray, window_title: str | None) -> None:
        if self.last_detection is None:
            return
        self.draw_overlay(frame)
        if window_title:
            cv2.imshow(window_title, frame)
            # ensure OpenCV flushes the buffer before IK/motion run
            cv2.waitKey(1)

    def _prepare_mask(self, frame_shape: tuple[int, int, int], detection: GroundingDINODetection) -> np.ndarray | None:
        mask = detection.mask
        if mask is not None:
            mask_arr = np.asarray(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            if mask_arr.dtype == bool:
                prepared = mask_arr
            else:
                threshold = 0.5 if mask_arr.max() <= 1 else 127
                prepared = mask_arr > threshold
            if prepared.shape != frame_shape[:2]:
                resized = cv2.resize(
                    prepared.astype(np.uint8),
                    (frame_shape[1], frame_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                prepared = resized.astype(bool)
            return prepared
        if self.require_mask_for_overlay:
            return None
        return self._bbox_to_mask(frame_shape[:2], detection.bbox)

    @staticmethod
    def _bbox_to_mask(shape: tuple[int, int], bbox: np.ndarray) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=bool)
        x1 = max(int(np.floor(bbox[0])), 0)
        y1 = max(int(np.floor(bbox[1])), 0)
        x2 = min(int(np.ceil(bbox[2])), width)
        y2 = min(int(np.ceil(bbox[3])), height)
        if x2 <= x1 or y2 <= y1:
            return mask
        mask[y1:y2, x1:x2] = True
        return mask

    def _apply_segmentation_overlay(self, frame: np.ndarray, mask: np.ndarray) -> None:
        if not mask.any():
            return
        overlay = frame.copy()
        overlay[mask] = self.segmentation_color
        cv2.addWeighted(overlay, self.segmentation_alpha, frame, 1.0 - self.segmentation_alpha, 0, dst=frame)

    def _move_to_pose(self, current: np.ndarray, pose: np.ndarray, gripper_target: float, hold: float) -> np.ndarray:
        assert self.kinematics is not None
        goal = self.kinematics.inverse_kinematics(
            current_joint_pos=current,
            desired_ee_pose=pose,
            position_weight=self.args.position_weight,
            orientation_weight=self.args.orientation_weight,
        )
        goal = self._set_gripper_target(goal, gripper_target)
        reached = self._send_joint_trajectory(current, goal)
        if hold > 0:
            busy_wait(hold)
        return reached

    def _send_joint_trajectory(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        assert self.robot is not None
        delta = np.abs(goal - start).max()
        step_deg = max(self.args.joint_step_deg, 0.1)
        steps = max(1, int(np.ceil(delta / step_deg)))
        for idx in range(1, steps + 1):
            alpha = idx / steps
            waypoint = start + alpha * (goal - start)
            self._send_joint_command(waypoint)
            if self.args.motion_dt > 0:
                busy_wait(self.args.motion_dt)
        return goal.copy()

    def _send_joint_command(self, joints: np.ndarray) -> None:
        assert self.robot is not None
        cmd = {f"{name}.pos": float(joints[i]) for i, name in enumerate(self.joint_names)}
        self.robot.send_action(cmd)


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
    parser.add_argument("--ee-axis-length", type=float, default=0.05, help="Axis length for EE pose overlay (m).")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)

    detect = parser.add_argument_group("grounding dino + grasp planner")
    detect.add_argument("--target-object", help="Text prompt describing the object to detect/grasp.")
    detect.add_argument("--grounding-config", help="GroundingDINO config path.")
    detect.add_argument("--grounding-checkpoint", help="GroundingDINO checkpoint path (.pth).")
    detect.add_argument("--grounding-device", default="cuda", help="Torch device for GroundingDINO.")
    detect.add_argument("--grounding-server-url", help="HTTP endpoint for remote GroundingDINO server.")
    detect.add_argument(
        "--grounding-server-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout (s) for remote GroundingDINO requests.",
    )
    detect.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    detect.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    detect.add_argument("--min-box-area", type=float, default=1_500.0, help="Ignore detections below this pixel area.")
    detect.add_argument("--target-plane-z", type=float, help="Table/plane height in base frame meters.")
    detect.add_argument("--approach-offset", type=float, default=0.08, help="Height above target for pre-grasp (m).")
    detect.add_argument(
        "--pre-y-offset",
        type=float,
        default=-0.05,
        help="Translate the pre-grasp pose along base +Y before descending (m).",
    )
    detect.add_argument("--grasp-offset", type=float, default=-0.01, help="Offset applied during grasp (m).")
    detect.add_argument("--retreat-offset", type=float, default=0.12, help="Retreat height above base plane (m).")
    detect.add_argument("--position-weight", type=float, default=1.0, help="IK position task weight.")
    detect.add_argument("--orientation-weight", type=float, default=0.05, help="IK orientation task weight.")
    detect.add_argument("--joint-step-deg", type=float, default=2.0, help="Max joint delta per step (deg).")
    detect.add_argument("--motion-dt", type=float, default=0.1, help="Pause between joint steps (s).")
    detect.add_argument("--pregrasp-hold-s", type=float, default=0.5, help="Hold at pre-grasp pose (s).")
    detect.add_argument("--grasp-hold-s", type=float, default=0.5, help="Hold after closing gripper (s).")
    detect.add_argument("--retreat-hold-s", type=float, default=0.5, help="Hold after retreat pose (s).")
    detect.add_argument(
        "--trajectory-mode",
        choices=["cartesian", "joint"],
        default="cartesian",
        help="Choose staged cartesian motion (default) or a simplified joint-space trajectory.",
    )
    detect.add_argument("--gripper-open-pos", type=float, default=30.0, help="Open gripper command (deg or %%).")
    detect.add_argument("--gripper-close-pos", type=float, default=5.0, help="Closed gripper command (deg or %%).")
    detect.add_argument("--gripper-channel", default="gripper", help="Motor name for gripper.")
    detect.add_argument(
        "--initial-joints",
        help="JSON list of joint angles (deg) to reach at startup; defaults to zeros.",
    )
    detect.add_argument(
        "--skip-initial-pose",
        action="store_true",
        help="Skip moving the robot to the initial joint pose on startup.",
    )
    detect.add_argument("--reopen-after-retreat", action="store_true", help="Open gripper after retreat.")
    detect.add_argument(
        "--segmentation-alpha",
        type=float,
        default=0.35,
        help="Alpha blend (0-1) applied to the segmentation overlay.",
    )
    detect.add_argument(
        "--segmentation-color",
        default="[0, 165, 255]",
        help="BGR color used for the segmentation overlay and marker.",
    )
    detect.add_argument(
        "--require-mask-for-overlay",
        action="store_true",
        help="Skip segmentation shading when the detector does not provide a mask.",
    )
    detect.add_argument(
        "--hide-grasp-marker",
        action="store_true",
        help="Disable the 2D crosshair drawn at the projected grasp point.",
    )
    detect.add_argument(
        "--use-grounded-sam2",
        action="store_true",
        help="Use the local Grounded SAM 2 pipeline instead of classic GroundingDINO.",
    )
    detect.add_argument("--sam2-model-config", help="SAM 2 config path (used with --use-grounded-sam2).")
    detect.add_argument("--sam2-checkpoint", help="SAM 2 checkpoint path (used with --use-grounded-sam2).")
    detect.add_argument(
        "--sam2-grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face Grounding DINO model id used inside SAM 2 pipeline.",
    )
    detect.add_argument("--sam2-hf-revision", help="Optional HF revision for the SAM 2 Grounding DINO model.")
    detect.add_argument("--sam2-hf-cache-dir", help="Optional HF cache directory for SAM 2 components.")

    robot = parser.add_argument_group("robot config")
    robot.add_argument("--robot.type", dest="robot_type", help="Robot choice name (e.g., so101_follower).")
    robot.add_argument("--robot.port", dest="robot_port", help="Serial port/device path.")
    robot.add_argument("--robot.id", dest="robot_id")
    robot.add_argument("--robot.calibration_dir", dest="robot_calibration_dir")
    robot.add_argument("--robot.use_degrees", dest="robot_use_degrees")
    robot.add_argument("--robot.disable_torque_on_disconnect", dest="robot_disable_torque_on_disconnect")
    robot.add_argument("--robot.max_relative_target", dest="robot_max_relative_target")
    parser.add_argument("--urdf-path", help="Override URDF path for IK.")
    parser.add_argument("--target-frame-name", default="gripper_frame_link", help="IK target frame name.")
    parser.add_argument("--zero-offset-json-path", help="Joint zero offset calibration JSON.")
    parser.add_argument(
        "--camera.focus_value",
        dest="camera_focus_value",
        type=float,
        help="Optional manual focus value passed to VideoCapture (CAP_PROP_FOCUS).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.target_object and not args.grounding_server_url:
        if args.use_grounded_sam2:
            if args.sam2_model_config is None or args.sam2_checkpoint is None:
                default_config, default_ckpt = default_groundedsam2_paths()
                if args.sam2_model_config is None and default_config:
                    args.sam2_model_config = default_config
                if args.sam2_checkpoint is None and default_ckpt:
                    args.sam2_checkpoint = default_ckpt
            if args.sam2_model_config is None or args.sam2_checkpoint is None:
                raise ValueError(
                    "Grounded SAM 2 paths missing. Provide --sam2-model-config and --sam2-checkpoint or "
                    "set --grounding-server-url."
                )
        elif args.grounding_config is None or args.grounding_checkpoint is None:
            default_config, default_ckpt = _default_grounding_paths()
            if args.grounding_config is None and default_config:
                args.grounding_config = default_config
            if args.grounding_checkpoint is None and default_ckpt:
                args.grounding_checkpoint = default_ckpt
            if args.grounding_config is None or args.grounding_checkpoint is None:
                raise ValueError(
                    "GroundingDINO paths missing. Provide --grounding-config and --grounding-checkpoint "
                    "or specify --grounding-server-url."
                )

    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    calibration = json.loads(calibration_path.read_text())
    camera_to_base = _load_transform(calibration["camera_to_base"], "camera_to_base")
    base_to_camera = np.linalg.inv(camera_to_base)

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
    if args.camera_focus_value is not None:
        set_focus_on_camera(capture, int(args.camera_focus_value))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open camera source: {args.camera_source}")

    overlay_pose = _format_pose(camera_to_base)
    manipulator: VisionGraspController | None = None
    undistorter: FrameUndistorter | None = None
    effective_K = K
    effective_D = D

    window_title = "Hand-Eye Base Visualization"
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                continue
            if undistorter is None:
                image_size = (frame.shape[1], frame.shape[0])
                undistorter = FrameUndistorter(K, D, image_size)
                effective_K = undistorter.camera_matrix
                effective_D = None if undistorter.enabled else D
                if undistorter.enabled:
                    LOG.info(
                        "Undistorting %dx%d frames using precomputed rectification maps.",
                        image_size[0],
                        image_size[1],
                    )
                needs_manipulator = bool(args.target_object or args.robot_type)
                if needs_manipulator and manipulator is None:
                    manipulator = VisionGraspController(args, effective_K, effective_D, camera_to_base)
            frame_vis = undistorter.undistort(frame) if undistorter else frame
            axis_len = float(args.axis_length)
            axes_lines = _draw_base_axes(frame_vis, effective_K, effective_D, base_to_camera, axis_len)

            overlay_text = f"Camera->Base {overlay_pose}" if overlay_pose else "Camera->Base unavailable"
            cv2.putText(
                frame_vis,
                overlay_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255) if overlay_pose else (0, 0, 255),
                1,
            )
            text_y = 55
            if axes_lines:
                for line in axes_lines:
                    cv2.putText(
                        frame_vis,
                        line,
                        (20, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (200, 200, 200),
                        1,
                    )
                    text_y += 20
            if manipulator:
                ee_lines = manipulator.draw_ee_pose(frame_vis)
                if ee_lines:
                    for line in ee_lines:
                        cv2.putText(
                            frame_vis,
                            line,
                            (20, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (180, 200, 255),
                            1,
                        )
                        text_y += 20
                manipulator.draw_overlay(frame_vis)
                base_y = frame_vis.shape[0] - 20
                for idx, line in enumerate(manipulator.status_lines):
                    cv2.putText(
                        frame_vis,
                        line,
                        (20, base_y - idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            cv2.imshow(window_title, frame_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g") and manipulator and manipulator.enable_detection:
                manipulator.handle_grasp(frame_vis.copy(), window_title)
    finally:
        if manipulator:
            manipulator.shutdown()
        capture.release()
        cv2.destroyAllWindows()


def _build_robot_config(args: argparse.Namespace) -> RobotConfig:
    if not args.robot_type:
        raise ValueError("robot.type is required to create a robot.")
    config_cls = RobotConfig.get_choice_class(args.robot_type)
    cfg_kwargs = {}
    mapping = {
        "id": args.robot_id,
        "port": args.robot_port,
        "calibration_dir": Path(args.robot_calibration_dir).expanduser()
        if args.robot_calibration_dir
        else None,
        "use_degrees": None if args.robot_use_degrees is None else _str2bool(str(args.robot_use_degrees)),
        "disable_torque_on_disconnect": None
        if args.robot_disable_torque_on_disconnect is None
        else _str2bool(str(args.robot_disable_torque_on_disconnect)),
        "max_relative_target": (
            json.loads(args.robot_max_relative_target)
            if args.robot_max_relative_target and args.robot_max_relative_target.strip().startswith(("[", "{"))
            else (
                float(args.robot_max_relative_target)
                if args.robot_max_relative_target is not None
                else None
            )
        ),
    }
    for field_name, value in mapping.items():
        if value is not None:
            cfg_kwargs[field_name] = value
    return config_cls(**cfg_kwargs)


if __name__ == "__main__":
    main()
