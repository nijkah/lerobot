#!/usr/bin/env python

"""Live overlay of the robot base frame estimated from hand-eye calibration.

Press `g` to trigger GroundingDINO-driven grasp planning when a target object prompt is provided.
This mode projects the detected pixel onto the robot base frame, solves IK, and commands a slow,
safe Cartesian approach/grasp/retreat sequence similar to ``lerobot_scripted_cartesian.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
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
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation

LOG = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
GROUNDING_REPO = REPO_ROOT / "GroundingDINO"


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


class RemoteGroundingDinoDetector:
    """Queries a remote GroundingDINO server via HTTP."""

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


class VisionGraspController:
    """Handles detection -> base frame projection -> IK -> slow motion execution."""

    def __init__(self, args: argparse.Namespace, K: np.ndarray, camera_to_base: np.ndarray):
        if args.target_plane_z is None:
            raise ValueError("--target-plane-z is required when --target-object is set.")
        self.args = args
        if args.grounding_server_url:
            client = GroundingDINOHTTPClient(
                server_url=args.grounding_server_url,
                timeout=args.grounding_server_timeout,
            )
            self.detector = RemoteGroundingDinoDetector(client)
        else:
            self.detector = LocalGroundingDinoDetector(
                config_path=args.grounding_config,
                checkpoint_path=args.grounding_checkpoint,
                device=args.grounding_device,
            )
        self.K = np.array(K, dtype=float)
        self.K_inv = np.linalg.inv(self.K)
        self.camera_to_base = np.array(camera_to_base, dtype=float)
        self.camera_origin = self.camera_to_base[:3, 3]
        self.camera_rot = self.camera_to_base[:3, :3]
        self.plane_z = float(args.target_plane_z)
        self.status_lines: list[str] = [
            f"Vision grasp ready. Prompt='{args.target_object}'. Press 'g' to search."
        ]
        self.last_detection: tuple[GroundingDINODetection, np.ndarray] | None = None
        self.dry_run = bool(args.dry_run)
        self.joint_names: list[str] = []
        self.name_to_idx: dict[str, int] = {}
        self.gripper_key = f"{args.gripper_channel}.pos"
        self.gripper_index: int | None = None
        self.robot = None
        self.kinematics = None
        self.orientation_rotvec = Rotation.from_euler_xyz(
            _parse_vector3(args.ee_rpy_deg, "--ee-rpy-deg"), degrees=True
        ).as_rotvec()

        if not self.dry_run:
            if not args.robot_type:
                raise ValueError("Specify --robot.type/--robot.port or use --dry-run.")
            robot_config = _build_robot_config(args)
            self.robot = make_robot_from_config(robot_config)
            self.robot.connect()
            self.joint_names = list(self.robot.bus.motors.keys())  # type: ignore[attr-defined]
            self.name_to_idx = {name: idx for idx, name in enumerate(self.joint_names)}
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

    def shutdown(self):
        if self.robot and getattr(self.robot, "is_connected", False):
            self.robot.disconnect()

    def draw_overlay(self, frame: np.ndarray):
        if self.last_detection is None:
            return
        detection, world_point = self.last_detection
        x1, y1, x2, y2 = detection.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        coord = f"x={world_point[0]:.3f}, y={world_point[1]:.3f}, z={world_point[2]:.3f}"
        cv2.putText(frame, coord, (x1, min(frame.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    def handle_grasp(self, frame_bgr: np.ndarray):
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
        if self.dry_run or not self.robot or not self.kinematics:
            self.status_lines = [
                f"Dry run detection '{detection.label}' score={detection.confidence:.2f}",
                f"Base xyz (m): {world_point[0]:.3f}, {world_point[1]:.3f}, {world_point[2]:.3f}",
            ]
            return

        try:
            self._execute_grasp(world_point)
            self.status_lines = [
                f"Grasp sequence executed at x={world_point[0]:.3f}, y={world_point[1]:.3f}, z={world_point[2]:.3f}",
                "Press 'g' to run again or 'q' to quit.",
            ]
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Failed to execute grasp: %s", exc)
            self.status_lines = [f"Grasp execution failed: {exc}"]

    def _pixel_to_base(self, bbox: np.ndarray) -> np.ndarray:
        cx = 0.5 * (bbox[0] + bbox[2])
        cy = 0.5 * (bbox[1] + bbox[3])
        pixel = np.array([cx, cy, 1.0], dtype=float)
        ray_cam = self.K_inv @ pixel
        ray_cam /= np.linalg.norm(ray_cam)
        ray_base = self.camera_rot @ ray_cam
        denom = ray_base[2]
        if abs(denom) < 1e-6:
            raise ValueError("Camera ray is parallel to target plane.")
        t = (self.plane_z - self.camera_origin[2]) / denom
        if t <= 0:
            raise ValueError("Projected point lies behind the camera.")
        return self.camera_origin + t * ray_base

    def _make_pose(self, position: np.ndarray) -> np.ndarray:
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = position
        pose[:3, :3] = Rotation.from_rotvec(self.orientation_rotvec).as_matrix()
        return pose

    def _execute_grasp(self, world_point: np.ndarray):
        pre = world_point.copy()
        pre[2] += self.args.approach_offset
        grasp = world_point.copy()
        grasp[2] += self.args.grasp_offset
        retreat = world_point.copy()
        retreat[2] += self.args.retreat_offset

        current = self._read_joint_positions()
        current = self._move_to_pose(current, self._make_pose(pre), self.args.gripper_open_pos, hold=self.args.pregrasp_hold_s)
        current = self._move_to_pose(current, self._make_pose(grasp), self.args.gripper_open_pos, hold=0.0)
        current = self._move_to_pose(current, self._make_pose(grasp), self.args.gripper_close_pos, hold=self.args.grasp_hold_s)
        current = self._move_to_pose(current, self._make_pose(retreat), self.args.gripper_close_pos, hold=self.args.retreat_hold_s)
        # Open gripper after retreat if requested
        if self.args.reopen_after_retreat:
            self._send_joint_trajectory(current, self._set_gripper_target(current.copy(), self.args.gripper_open_pos))

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
            cmd = {f"{name}.pos": float(waypoint[i]) for i, name in enumerate(self.joint_names)}
            self.robot.send_action(cmd)
            busy_wait(self.args.motion_dt)
        return goal.copy()


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
    detect.add_argument("--ee-rpy-deg", default="[0.0, 0.0, 0.0]", help="Gripper orientation as [roll,pitch,yaw] degrees.")
    detect.add_argument("--approach-offset", type=float, default=0.08, help="Height above target for pre-grasp (m).")
    detect.add_argument("--grasp-offset", type=float, default=-0.01, help="Offset applied during grasp (m).")
    detect.add_argument("--retreat-offset", type=float, default=0.12, help="Retreat height above base plane (m).")
    detect.add_argument("--position-weight", type=float, default=1.0, help="IK position task weight.")
    detect.add_argument("--orientation-weight", type=float, default=0.05, help="IK orientation task weight.")
    detect.add_argument("--joint-step-deg", type=float, default=2.0, help="Max joint delta per step (deg).")
    detect.add_argument("--motion-dt", type=float, default=0.1, help="Pause between joint steps (s).")
    detect.add_argument("--pregrasp-hold-s", type=float, default=0.5, help="Hold at pre-grasp pose (s).")
    detect.add_argument("--grasp-hold-s", type=float, default=0.5, help="Hold after closing gripper (s).")
    detect.add_argument("--retreat-hold-s", type=float, default=0.5, help="Hold after retreat pose (s).")
    detect.add_argument("--gripper-open-pos", type=float, default=100.0, help="Open gripper command (deg or %%).")
    detect.add_argument("--gripper-close-pos", type=float, default=5.0, help="Closed gripper command (deg or %%).")
    detect.add_argument("--gripper-channel", default="gripper", help="Motor name for gripper.")
    detect.add_argument("--reopen-after-retreat", action="store_true", help="Open gripper after retreat.")
    detect.add_argument("--dry-run", action="store_true", help="Skip robot control; only report pose.")

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
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if (
        args.target_object
        and not args.grounding_server_url
        and (args.grounding_config is None or args.grounding_checkpoint is None)
    ):
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
    manipulator: VisionGraspController | None = None
    if args.target_object:
        manipulator = VisionGraspController(args, K, camera_to_base)

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
            if manipulator:
                manipulator.draw_overlay(frame)
                base_y = frame.shape[0] - 20
                for idx, line in enumerate(manipulator.status_lines):
                    cv2.putText(
                        frame,
                        line,
                        (20, base_y - idx * 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g") and manipulator:
                manipulator.handle_grasp(frame.copy())
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
