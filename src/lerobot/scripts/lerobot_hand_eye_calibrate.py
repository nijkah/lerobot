#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Eye-to-hand calibration assistant that pairs robot forward kinematics with AprilTag detections.

Workflow (third-view camera):
  1. Connect to the follower robot defined in --robot.* and leave it in gravity-comp or passive mode.
  2. Mount an AprilTag on the end effector, manually move the arm to diverse poses, and capture samples.
  3. The script records base->gripper poses from FK plus camera images; press SPACE (or enable --auto_capture_s)
     to confirm each sample when the AprilTag is visible.
  4. Once enough poses are collected, OpenCV's `calibrateHandEye` solves the eye-to-hand transform and optionally
     reports residuals + saves matrices in JSON.

Requirements:
  - Camera intrinsics (K, D) saved beforehand via `np.savez` or JSON.
  - `pupil-apriltags` available for pose estimation (`pip install pupil-apriltags`).

Example:
    python -m lerobot.scripts.lerobot_hand_eye_calibrate \\
        --robot.name so101_follower --robot.port=/dev/ttyUSB0 \\
        --camera.index_or_path=0 --camera.width=640 --camera.height=480 \\
        --intrinsics_path=calibration/camera_intrinsics_top.npz \\
        --tag_size_m=0.045 --required_samples=20 --output_path=calibration/hand_eye_top.json
"""

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

try:
    import pybullet as p  # type: ignore[import-not-found]
    import pybullet_data  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    p = None  # type: ignore[assignment]
    pybullet_data = None  # type: ignore[assignment]

try:
    from pupil_apriltags import Detector as AprilTagDetector  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    AprilTagDetector = None

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging

from lerobot.robots import (  # noqa: F401  (registered robot configs for the parser)
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)

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

class CaptureSample(NamedTuple):
    # Transform returned by FK: maps gripper-frame coordinates into the base/world frame (base <- gripper).
    gripper_to_base: np.ndarray
    target_to_camera: np.ndarray
    corners: np.ndarray
    timestamp_s: float
    decision_margin: float
    image_path: str | None = None


@dataclass
class CameraCaptureConfig:
    index_or_path: str = "0"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    show_preview: bool = True
    auto_capture_s: float = 0.0
    auto_focus: bool | None = False
    focus_value: int | None = None


@dataclass(kw_only=True)
class HandEyeCalibrationConfig:
    robot: RobotConfig
    camera: CameraCaptureConfig = field(default_factory=CameraCaptureConfig)

    intrinsics_path: str
    tag_family: str = "tag36h11"
    tag_id: int | None = None
    tag_size_m: float = 0.05
    axis_length_m: float = 0.05
    required_samples: int = 20

    perform_capture: bool = True
    load_samples_path: str | None = None
    record_samples_path: str | None = None
    pybullet_visualization: bool = False
    pybullet_gui: bool = True
    pybullet_disable_ui: bool = True
    pybullet_camera_distance: float = 0.5
    pybullet_camera_yaw: float = 45.0
    pybullet_camera_pitch: float = -60.0

    urdf_path: str | None = None
    target_frame_name: str = "gripper_frame_link"
    zero_offset_json_path: str | None = None

    save_frames_dir: str | None = None
    output_path: str = "calibration/hand_eye_result.json"
    solver_method: str = "tsai"
    validate: bool = True

    filter_outliers: bool = False
    max_filter_passes: int = 1
    max_base_rot_deg: float | None = None
    max_base_trans_mm: float | None = None
    max_tag_rot_deg: float | None = None
    max_tag_trans_mm: float | None = None

    min_decision_margin: float = 30.0
    detector_decimate: float = 2.0
    detector_sigma: float = 0.8


@dataclass
class HandEyeResult:
    samples: list[CaptureSample]
    T_gc: np.ndarray | None = None
    T_cg: np.ndarray | None = None
    T_cb: np.ndarray | None = None
    T_bc: np.ndarray | None = None
    T_gt: np.ndarray | None = None
    base_camera_errors: list[tuple[float, float]] | None = None
    tag_errors: list[tuple[float, float] | None] | None = None
    passes: int = 0
    filtered_total: int = 0


def _resolve_video_source(raw: str) -> int | str:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return raw


def _load_intrinsics(path: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera intrinsics file not found: {path}")
    if p.suffix.lower() == ".npz":
        data = np.load(p)
        K = np.array(data["K"], dtype=np.float64)
        D = np.array(data["D"], dtype=np.float64)
    else:
        payload = json.loads(p.read_text())
        if "K" in payload:
            K = np.array(payload["K"], dtype=np.float64)
        elif "camera_matrix" in payload:
            K = np.array(payload["camera_matrix"], dtype=np.float64)
        else:
            raise KeyError("intrinsics JSON must contain 'K' or 'camera_matrix'")
        if "D" in payload:
            D = np.array(payload["D"], dtype=np.float64)
        elif "distortion_coefficients" in payload:
            D = np.array(payload["distortion_coefficients"], dtype=np.float64)
        else:
            raise KeyError("intrinsics JSON must contain 'D' or 'distortion_coefficients'")
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    D = D.reshape(-1, 1)
    return K, D


def _make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def _mean_rotation(mats: list[np.ndarray]) -> np.ndarray:
    quats = np.array([Rotation.from_matrix(m).as_quat() for m in mats], dtype=float)
    if quats.size == 0:
        raise ValueError("Cannot compute mean rotation of empty set")
    ref = quats[0]
    aligned = [ref]
    for q in quats[1:]:
        aligned.append(-q if np.dot(q, ref) < 0.0 else q)
    mean = np.mean(aligned, axis=0)
    mean /= np.linalg.norm(mean)
    return Rotation.from_quat(mean).as_matrix()


def _matrix_to_list(T: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in T]


def _relativize_path(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _serialize_sample(sample: CaptureSample, base_dir: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "gripper_to_base": _matrix_to_list(sample.gripper_to_base),
        "target_to_camera": _matrix_to_list(sample.target_to_camera),
        "timestamp": float(sample.timestamp_s),
        "decision_margin": float(sample.decision_margin),
        "corners": sample.corners.astype(float).tolist(),
    }
    if sample.image_path:
        payload["image_path"] = _relativize_path(Path(sample.image_path), base_dir)
    return payload


def _matrix_from_payload(data: Any, name: str) -> np.ndarray:
    arr = np.array(data, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {arr.shape}")
    return arr


def _corners_from_payload(data: Any) -> np.ndarray:
    if data is None:
        return np.zeros((4, 2), dtype=np.float32)
    arr = np.array(data, dtype=np.float32)
    if arr.shape != (4, 2):
        raise ValueError(f"corners must be 4x2, got {arr.shape}")
    return arr


def _load_samples_file(path: Path) -> tuple[list[CaptureSample], dict[str, Any]]:
    payload = json.loads(path.read_text())
    version = payload.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported samples file version: {version}")
    entries = payload.get("samples")
    if not isinstance(entries, list):
        raise ValueError("Samples file missing 'samples' list")
    base_dir = path.parent
    samples: list[CaptureSample] = []
    for idx, item in enumerate(entries):
        try:
            gripper_to_base = _matrix_from_payload(item["gripper_to_base"], "gripper_to_base")
            target_to_camera = _matrix_from_payload(item["target_to_camera"], "target_to_camera")
        except KeyError as err:
            raise KeyError(f"Sample {idx} missing key: {err.args[0]}") from err
        corners = _corners_from_payload(item.get("corners"))
        timestamp = float(item.get("timestamp", 0.0))
        decision_margin = float(item.get("decision_margin", 0.0))
        image_path_val = item.get("image_path")
        if image_path_val:
            img_path = Path(image_path_val)
            if not img_path.is_absolute():
                img_path = (base_dir / img_path).resolve()
            image_path = str(img_path)
        else:
            image_path = None
        samples.append(
            CaptureSample(
                gripper_to_base=gripper_to_base,
                target_to_camera=target_to_camera,
                corners=corners,
                timestamp_s=timestamp,
                decision_margin=decision_margin,
                image_path=image_path,
            )
        )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return samples, metadata


def _write_samples_file(
    samples: list[CaptureSample],
    dest: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    base_dir = dest.parent
    meta = dict(metadata or {})
    meta["updated_at"] = time.time()
    payload = {
        "version": 1,
        "metadata": meta,
        "samples": [_serialize_sample(sample, base_dir) for sample in samples],
    }
    dest.write_text(json.dumps(payload, indent=2))


class TagPoseEstimator:
    def __init__(
        self,
        family: str,
        tag_size_m: float,
        K: np.ndarray,
        decimate: float,
        sigma: float,
        min_margin: float,
    ):
        if AprilTagDetector is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "pupil-apriltags is required. Install with `pip install pupil-apriltags`."
            )
        self.detector = AprilTagDetector(
            families=family,
            nthreads=1,
            quad_decimate=decimate,
            quad_sigma=sigma,
            refine_edges=True,
        )
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.tag_size = tag_size_m
        self.min_margin = min_margin

    def detect(self, gray: np.ndarray, tag_id: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=self.tag_size,
        )
        if not detections:
            return None
        det = None
        for cand in detections:
            if tag_id is not None and cand.tag_id != tag_id:
                continue
            if cand.decision_margin < self.min_margin:
                continue
            if det is None or cand.decision_margin > det.decision_margin:
                det = cand
        if det is None:
            return None
        R = np.array(det.pose_R, dtype=np.float64)
        t = np.array(det.pose_t, dtype=np.float64).reshape(3)
        corners = np.array(det.corners, dtype=np.float32)
        return R, t, corners, float(det.decision_margin)


def _resolve_urdf(cfg: HandEyeCalibrationConfig, robot) -> str:
    if cfg.urdf_path is not None:
        return cfg.urdf_path
    urdf = getattr(cfg.robot, "urdf_path", None)
    if urdf is None:
        raise ValueError("URDF path is required. Pass --urdf_path or add it to the robot config.")
    return urdf


def _save_frame(frame: np.ndarray, save_dir: Path, idx: int) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fpath = save_dir / f"sample_{idx:03d}.png"
    cv2.imwrite(str(fpath), frame)
    return fpath


def _format_pose(T: np.ndarray) -> str:
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return (
        f"pos(m)=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), "
        f"rotvec(rad)=({rot[0]:.4f}, {rot[1]:.4f}, {rot[2]:.4f})"
    )



def _draw_tag_axes(
    frame: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    axis_length_m: float,
) -> None:
    """
    Draw the 3D coordinate axes of an AprilTag on the image frame.

    This function projects the 3D axes (X, Y, Z) of a detected AprilTag onto the 2D image
    plane and draws them as colored arrows, with the origin at the tag's center.

    Args:
        frame (np.ndarray): The input image frame on which to draw the axes. This array
            is modified in-place.
        R (np.ndarray): The 3x3 rotation matrix representing the tag's orientation in
            camera coordinates.
        t (np.ndarray): The 3D translation vector representing the tag's position in
            camera coordinates.
        K (np.ndarray): The 3x3 camera intrinsic matrix containing focal lengths and
            principal point.
        D (np.ndarray | None): The distortion coefficients for the camera. Can be None
            if no distortion correction is needed.
        axis_length_m (float): The length of each axis in meters. If the value is invalid
            (<= 0 or not convertible to float), the function returns early without drawing.

    Returns:
        None: The function modifies the frame in-place and does not return a value.

    Notes:
        - The axes are drawn in BGR color space: X=red (0,0,255), Y=green (0,255,0),
          Z=blue (255,0,0).
        - Each axis is drawn as an arrowed line with a tip length ratio of 0.2.
        - If axis_length_m is invalid or <= 0.0, no axes are drawn.
    """
    try:
        axis_len = float(axis_length_m)
    except (TypeError, ValueError):
        axis_len = 0.05
    if axis_len <= 0.0:
        return
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    dist = None
    if D is not None:
        dist = D.reshape(-1, 1)
    proj, _ = cv2.projectPoints(axis_points, rvec, tvec, K.astype(np.float64), dist)
    proj = proj.reshape(-1, 2)
    origin = tuple(int(round(v)) for v in proj[0])
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: X=red, Y=green, Z=blue
    for idx, color in enumerate(colors, start=1):
        end = tuple(int(round(v)) for v in proj[idx])
        cv2.arrowedLine(frame, origin, end, color, 2, tipLength=0.2)


def _is_valid_transform(T: np.ndarray | None) -> bool:
    if T is None:
        return False
    if T.shape != (4, 4):
        return False
    return np.isfinite(T).all()


def _draw_base_axes(
    frame: np.ndarray,
    T_bc: np.ndarray,
    K: np.ndarray,
    D: np.ndarray | None,
    axis_length_m: float,
) -> list[str] | None:
    if not _is_valid_transform(T_bc):
        return None
    try:
        axis_len = float(axis_length_m)
    except (TypeError, ValueError):
        axis_len = 0.05
    if axis_len <= 0.0:
        return None
    base_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],
            [0.0, axis_len, 0.0],
            [0.0, 0.0, axis_len],
        ],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(T_bc[:3, :3].astype(np.float64))
    tvec = T_bc[:3, 3].reshape(3, 1).astype(np.float64)
    if not np.isfinite(tvec).all():
        logging.debug("Base->Camera translation contains non-finite values; skipping axes render.")
        return None
    if tvec[2, 0] <= 1e-6:
        logging.debug("Base frame lies behind camera (z=%.4f); skipping axes render.", float(tvec[2, 0]))
        return None
    dist = None if D is None else D.reshape(-1, 1)
    proj, _ = cv2.projectPoints(base_points, rvec, tvec, K.astype(np.float64), dist)
    proj = proj.reshape(-1, 2)
    if not np.isfinite(proj).all():
        return None
    origin_vals = proj[0]
    if origin_vals.size != 2 or not np.isfinite(origin_vals).all():
        return None
    origin = (int(round(float(origin_vals[0]))), int(round(float(origin_vals[1]))))
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR: X=blue, Y=green, Z=red for base frame
    coord_lines = [
        f"Base origin px=({origin_vals[0]:.1f}, {origin_vals[1]:.1f})",
        f"Base x px=({proj[1, 0]:.1f}, {proj[1, 1]:.1f})",
        f"Base y px=({proj[2, 0]:.1f}, {proj[2, 1]:.1f})",
        f"Base z px=({proj[3, 0]:.1f}, {proj[3, 1]:.1f})",
    ]
    for idx, color in enumerate(colors, start=1):
        end_vals = proj[idx]
        if end_vals.size != 2 or not np.isfinite(end_vals).all():
            continue
        end = (int(round(float(end_vals[0]))), int(round(float(end_vals[1]))))
        try:
            cv2.arrowedLine(frame, origin, end, color, 2, tipLength=0.2)
        except cv2.error:  # pragma: no cover - guard against driver oddities
            logging.debug("Failed to draw base axis arrow; skipping this frame.")
    return coord_lines


class PyBulletVisualizer:
    def __init__(
        self,
        urdf_path: str,
        joint_names: list[str],
        axis_length_m: float,
        use_gui: bool,
        disable_ui: bool,
        camera_distance: float,
        camera_yaw: float,
        camera_pitch: float,
    ) -> None:
        if p is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "pybullet is required for visualization. Install with `pip install pybullet`."
            )
        self.client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.resetSimulation(physicsClientId=self.client)
        if pybullet_data is not None:
            try:
                p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
            except Exception:  # pragma: no cover - diagnostics only
                logging.debug("Failed to set pybullet additional search path.")
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        # Attempt to drop a plane for orientation; ignore failures silently.
        try:  # pragma: no cover - visualization aid only
            p.loadURDF("plane.urdf", physicsClientId=self.client)
        except Exception:
            logging.debug("Could not load plane.urdf into PyBullet scene.")
        try:
            self.robot_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=self.client)
        except Exception as exc:  # pragma: no cover
            p.disconnect(self.client)
            raise RuntimeError(f"Failed to load URDF into PyBullet: {exc}") from exc

        if disable_ui:
            try:  # pragma: no cover - visualization helper
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
                p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=self.client)
                p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0, physicsClientId=self.client)
                p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0, physicsClientId=self.client)
            except Exception:
                logging.debug("Could not disable PyBullet UI overlays.")

        try:  # pragma: no cover - visualization helper
            p.resetDebugVisualizerCamera(
                cameraDistance=float(camera_distance),
                cameraYaw=float(camera_yaw),
                cameraPitch=float(camera_pitch),
                cameraTargetPosition=[0.0, 0.0, 0.5],
                physicsClientId=self.client,
            )
        except Exception:
            logging.debug("Could not set PyBullet camera parameters.")

        self.joint_name_to_id: dict[str, int] = {}
        self._missing_joint_names: set[str] = set()
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client)
        for jid in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, jid, physicsClientId=self.client)
            name = joint_info[1].decode("utf-8")
            self.joint_name_to_id[name] = jid
        self.joint_names = list(joint_names)
        self.axis_length = float(axis_length_m)
        self.axis_ids = [-1, -1, -1]

    def update(
        self,
        observation: dict[str, Any],
        kinematics: RobotKinematics,
        gripper_to_base: np.ndarray | None,
    ) -> None:
        if observation is None:
            return
        try:
            joint_abs_deg = kinematics.observation_to_joint_array(observation)
        except KeyError as err:
            logging.debug("PyBullet viz missing joint key: %s", err)
            return
        joint_rel_deg = kinematics.apply_zero_offset(joint_abs_deg)
        for name, deg in zip(kinematics.joint_names, joint_rel_deg, strict=False):
            joint_id = self.joint_name_to_id.get(name)
            if joint_id is None:
                if name not in self._missing_joint_names:
                    self._missing_joint_names.add(name)
                    logging.debug("PyBullet joint not found: %s", name)
                continue
            p.resetJointState(
                self.robot_id,
                joint_id,
                math.radians(deg),
                physicsClientId=self.client,
            )

        if gripper_to_base is None:
            return
        origin = np.asarray(gripper_to_base[:3, 3], dtype=float)
        axes = np.asarray(gripper_to_base[:3, :3], dtype=float)
        if not np.isfinite(origin).all() or not np.isfinite(axes).all():
            return
        if self.axis_length <= 0.0:
            return
        colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        for idx, color in enumerate(colors):
            axis_vec = axes[:, idx]
            end = origin + axis_vec * self.axis_length
            self.axis_ids[idx] = p.addUserDebugLine(
                origin.tolist(),
                end.tolist(),
                color,
                lineWidth=2.0,
                replaceItemUniqueId=self.axis_ids[idx],
                physicsClientId=self.client,
            )

    def close(self) -> None:
        if p is not None and getattr(self, "client", None) is not None:
            try:
                p.disconnect(self.client)
            finally:
                self.client = None


def _try_disable_torque(robot) -> bool:
    disable_method = getattr(robot, "disable_torque", None)
    if callable(disable_method):
        disable_method()
        return True
    bus = getattr(robot, "bus", None)
    if bus is not None and hasattr(bus, "disable_torque"):
        bus.disable_torque()
        return True
    return False


def _solve_hand_eye(
    samples: list[CaptureSample],
    method_flag: int,
    cfg: HandEyeCalibrationConfig,
) -> HandEyeResult:
    if len(samples) < 3:
        raise RuntimeError(f"Need >=3 samples, captured {len(samples)}")

    base_rot_limit = np.radians(cfg.max_base_rot_deg) if cfg.max_base_rot_deg is not None else None
    base_trans_limit = cfg.max_base_trans_mm / 1000.0 if cfg.max_base_trans_mm is not None else None
    tag_rot_limit = np.radians(cfg.max_tag_rot_deg) if cfg.max_tag_rot_deg is not None else None
    tag_trans_limit = cfg.max_tag_trans_mm / 1000.0 if cfg.max_tag_trans_mm is not None else None

    remaining = list(samples)
    filtered_total = 0
    passes_done = 0

    while True:
        if len(remaining) < 3:
            raise RuntimeError("Outlier filtering left fewer than 3 samples.")

        R_gripper2base, t_gripper2base = [], []
        R_target2cam, t_target2cam = [], []
        for s in remaining:
            T_gb = s.gripper_to_base  # base <- gripper (pose of EE expressed in base)
            # R_gripper2base.append(T_gb[:3, :3].astype(np.float64))
            # t_gripper2base.append(T_gb[:3, 3].reshape(3, 1).astype(np.float64))

            # invert to base -> gripper for eye-to-hand
            T_bg = _invert_transform(T_gb)
            R_gripper2base.append(T_bg[:3, :3].astype(np.float64))
            t_gripper2base.append(T_bg[:3, 3].reshape(3, 1).astype(np.float64))

            T_ct = s.target_to_camera
            R_target2cam.append(T_ct[:3, :3].astype(np.float64))
            t_target2cam.append(T_ct[:3, 3].reshape(3, 1).astype(np.float64))

        # R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        #     R_gripper2base,
        #     t_gripper2base,
        #     R_target2cam,
        #     t_target2cam,
        #     method=method_flag,
        # )
        # eye-to-hand formulation returns camera -> base
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            method=method_flag,
        )

        T_bc = _make_homogeneous(R_cam2base, t_cam2base.flatten())  # camera -> base
        T_cb = _invert_transform(T_bc)  # base -> camera


        base_camera_errors: list[tuple[float, float]] = []
        tag_errors: list[tuple[float, float] | None] = [None] * len(remaining)
        compute_tag_metrics = cfg.validate or tag_rot_limit is not None or tag_trans_limit is not None

        # for idx, (T_bc_i, sample) in enumerate(zip(base_to_camera_candidates, remaining, strict=False)):
        #     rot_err = Rotation.from_matrix(T_bc_i[:3, :3]).inv() * Rotation.from_matrix(T_bc[:3, :3])
        #     ang = np.linalg.norm(rot_err.as_rotvec())
        #     trans = np.linalg.norm(T_bc_i[:3, 3] - T_bc[:3, 3])
        #     base_camera_errors.append((ang, trans))

            # if compute_tag_metrics:
            #     T_bt = sample.gripper_to_base @ T_gt  # base <- tag
            #     T_ct_pred = T_cb @ T_bt  # camera <- tag
            #     rot_tag = Rotation.from_matrix(sample.target_to_camera[:3, :3]).inv() * Rotation.from_matrix(
            #         T_ct_pred[:3, :3]
            #     )
            #     tag_ang = np.linalg.norm(rot_tag.as_rotvec())
            #     tag_trans = np.linalg.norm(sample.target_to_camera[:3, 3] - T_ct_pred[:3, 3])
            #     tag_errors[idx] = (tag_ang, tag_trans)
        T_gc = None
        T_cg = None
        T_gt = None

        result = HandEyeResult(
            samples=remaining,
            T_gc=T_gc,
            T_cg=T_cg,
            T_cb=T_cb,
            T_bc=T_bc,
            T_gt=T_gt,
            base_camera_errors=base_camera_errors,
            tag_errors=tag_errors,
            passes=passes_done,
            filtered_total=filtered_total,
        )

        if not cfg.filter_outliers:
            return result

        bad_indices: set[int] = set()
        if base_rot_limit is not None:
            for idx, (ang, _) in enumerate(base_camera_errors):
                if ang > base_rot_limit:
                    bad_indices.add(idx)
        if base_trans_limit is not None:
            for idx, (_, trans) in enumerate(base_camera_errors):
                if trans > base_trans_limit:
                    bad_indices.add(idx)
        if tag_rot_limit is not None:
            for idx, err in enumerate(tag_errors):
                if err is not None and err[0] > tag_rot_limit:
                    bad_indices.add(idx)
        if tag_trans_limit is not None:
            for idx, err in enumerate(tag_errors):
                if err is not None and err[1] > tag_trans_limit:
                    bad_indices.add(idx)

        if not bad_indices:
            return result

        if passes_done >= cfg.max_filter_passes:
            logging.warning(
                "Reached max_filter_passes (%d); keeping %d flagged samples.",
                cfg.max_filter_passes,
                len(bad_indices),
            )
            return result

        filtered_total += len(bad_indices)
        passes_done += 1
        logging.warning("Filtering %d outlier samples (pass %d).", len(bad_indices), passes_done)

        remaining = [sample for idx, sample in enumerate(remaining) if idx not in bad_indices]


@parser.wrap()
def run_hand_eye_calibration(cfg: HandEyeCalibrationConfig):
    init_logging()
    logging.info("Config: %s", asdict(cfg))

    samples: list[CaptureSample] = []
    loaded_metadata: dict[str, Any] = {}
    loaded_count = 0

    load_path = Path(cfg.load_samples_path) if cfg.load_samples_path else None
    if load_path:
        if not load_path.exists():
            raise FileNotFoundError(f"Samples file not found: {load_path}")
        loaded_samples, loaded_metadata = _load_samples_file(load_path)
        samples.extend(loaded_samples)
        loaded_count = len(loaded_samples)
        logging.info("Loaded %d sample(s) from %s", loaded_count, load_path)

    captured_count = 0
    cap: cv2.VideoCapture | None = None
    robot = None

    if cfg.perform_capture:
        K, D = _load_intrinsics(cfg.intrinsics_path)
        cap_source = _resolve_video_source(cfg.camera.index_or_path)
        cap = cv2.VideoCapture(cap_source)
        if cfg.camera.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        if cfg.camera.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
        if cfg.camera.fps:
            cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)
        if cfg.camera.auto_focus is not None:
            if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if cfg.camera.auto_focus else 0):
                logging.warning("Camera driver did not accept CAP_PROP_AUTOFOCUS command.")
        if cfg.camera.focus_value is not None:
            if cfg.camera.auto_focus is not True:
                if not cap.set(cv2.CAP_PROP_FOCUS, 1):
                    logging.warning("Camera driver did not accept CAP_PROP_FOCUS command.")
                time.sleep(0.15)
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Frame grab failed after setting CAP_PROP_FOCUS.")

                if not cap.set(cv2.CAP_PROP_FOCUS, cfg.camera.focus_value):
                    logging.warning("Camera driver did not accept CAP_PROP_FOCUS command.")
                time.sleep(0.15)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Frame grab failed after setting manual focus.")
            else:
                logging.warning("focus_value specified but auto_focus=True; skipping manual focus set.")
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {cfg.camera.index_or_path}")

        save_dir = Path(cfg.save_frames_dir) if cfg.save_frames_dir else None
        last_capture = 0.0
        visualizer: PyBulletVisualizer | None = None

        try:
            tag_detector = TagPoseEstimator(
                family=cfg.tag_family,
                tag_size_m=cfg.tag_size_m,
                K=K,
                decimate=cfg.detector_decimate,
                sigma=cfg.detector_sigma,
                min_margin=cfg.min_decision_margin,
            )

            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            torque_disabled = _try_disable_torque(robot)
            if torque_disabled:
                logging.info("Robot torque disabled; arm should move freely for manual positioning.")
            else:
                logging.warning(
                    "Unable to automatically disable torque; move the arm using your standard procedure."
                )
            robot.disable_torque_on_disconnect = True  # optional

            urdf_path = _resolve_urdf(cfg, robot)
            bus = getattr(robot, "bus", None)
            motor_map = getattr(bus, "motors", None) if bus is not None else None
            if not motor_map:
                raise RuntimeError("Robot did not expose motor names via robot.bus.motors; FK unavailable.")
            motor_names = list(motor_map.keys())
            kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name=cfg.target_frame_name,
                joint_names=motor_names,
            )
            if cfg.zero_offset_json_path:
                zero_path = Path(cfg.zero_offset_json_path)
                if zero_path.exists():
                    kinematics.load_joint_zero_offset(str(zero_path))
                    logging.info("Loaded zero-offset from %s", zero_path)

            if cfg.pybullet_visualization:
                try:
                    visualizer = PyBulletVisualizer(
                        urdf_path=urdf_path,
                        joint_names=kinematics.joint_names,
                        axis_length_m=cfg.axis_length_m,
                        use_gui=cfg.pybullet_gui,
                        disable_ui=cfg.pybullet_disable_ui,
                        camera_distance=cfg.pybullet_camera_distance,
                        camera_yaw=cfg.pybullet_camera_yaw,
                        camera_pitch=cfg.pybullet_camera_pitch,
                    )
                    logging.info(
                        "PyBullet visualization active (%s renderer).",
                        "GUI" if cfg.pybullet_gui else "TinyRenderer",
                    )
                except Exception as viz_err:
                    visualizer = None
                    logging.error("Failed to initialize PyBullet visualization: %s", viz_err)

            logging.info("Move the robot manually. Press SPACE to capture a pose, 'q' to stop early.")
            if cfg.camera.auto_capture_s > 0:
                logging.info(
                    "Auto capture active: taking a sample every %.1f s when tag is visible.",
                    cfg.camera.auto_capture_s,
                )

            last_base_to_camera_preview: np.ndarray | None = None
            last_camera_to_base_preview: np.ndarray | None = None

            while len(samples) < cfg.required_samples:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read frame; retrying.")
                    continue
                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gripper_to_base_preview = None
                try:
                    preview_obs = robot.get_observation()
                    gripper_to_base_preview = kinematics.forward_kinematics_from_observation(preview_obs)
                    if not _is_valid_transform(gripper_to_base_preview):
                        gripper_to_base_preview = None
                    if visualizer is not None and gripper_to_base_preview is not None:
                        visualizer.update(preview_obs, kinematics, gripper_to_base_preview)
                except Exception as fk_err:  # pragma: no cover - visualization aid only
                    logging.debug("Preview FK failed: %s", fk_err)

                detection = tag_detector.detect(gray, cfg.tag_id)
                detected = detection is not None

                if detected:
                    R_tc, t_tc, corners, margin = detection
                    for i in range(4):
                        pt1 = tuple(int(v) for v in corners[i])
                        pt2 = tuple(int(v) for v in corners[(i + 1) % 4])
                        cv2.line(display_frame, pt1, pt2, (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"Tag margin: {margin:.1f}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    _draw_tag_axes(display_frame, R_tc, t_tc, K, D, cfg.axis_length_m)
                else:
                    cv2.putText(
                        display_frame,
                        "Tag not detected",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                overlay = f"samples {len(samples)}/{cfg.required_samples} | press SPACE to record"
                cv2.putText(
                    display_frame,
                    overlay,
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                pose_lines: list[str] = []
                camera_to_base_preview: np.ndarray | None = None
                base_to_camera_preview: np.ndarray | None = None
                if detected:
                    tag_pose_h = _make_homogeneous(R_tc, t_tc)
                    pose_lines.append(f"Tag->Camera {_format_pose(tag_pose_h)}")
                    if gripper_to_base_preview is not None:
                        # Approximate the camera pose relative to the base assuming tag and gripper frames coincide.
                        camera_to_tag = _invert_transform(tag_pose_h)
                        candidate_cam_to_base = camera_to_tag @ gripper_to_base_preview
                        if _is_valid_transform(candidate_cam_to_base):
                            camera_to_base_preview = candidate_cam_to_base
                            base_to_camera_preview = _invert_transform(candidate_cam_to_base)
                            last_base_to_camera_preview = base_to_camera_preview
                            last_camera_to_base_preview = camera_to_base_preview
                else:
                    pose_lines.append("Tag->Camera pose unavailable")
                if gripper_to_base_preview is not None:
                    pose_lines.append(f"Gripper->Base {_format_pose(gripper_to_base_preview)}")
                else:
                    pose_lines.append("Gripper->Base pose unavailable")
                if camera_to_base_preview is not None:
                    pose_lines.append(f"Camera->Base≈ {_format_pose(camera_to_base_preview)}")
                elif last_camera_to_base_preview is not None:
                    pose_lines.append(f"Camera->Base≈ {_format_pose(last_camera_to_base_preview)}")
                base_axes_drawn = False
                base_axes_coords: list[str] | None = None
                if _is_valid_transform(base_to_camera_preview):
                    base_axes_coords = _draw_base_axes(display_frame, base_to_camera_preview, K, D, cfg.axis_length_m)
                    base_axes_drawn = base_axes_coords is not None
                elif _is_valid_transform(last_base_to_camera_preview):
                    base_axes_coords = _draw_base_axes(display_frame, last_base_to_camera_preview, K, D, cfg.axis_length_m)
                    base_axes_drawn = base_axes_coords is not None
                if not base_axes_drawn:
                    pose_lines.append("Camera->Base axes unavailable")
                elif base_axes_coords:
                    pose_lines.extend(base_axes_coords)
                    logging.debug("Base frame pixel coords: %s", base_axes_coords)
                for idx, line in enumerate(pose_lines):
                    cv2.putText(
                        display_frame,
                        line,
                        (20, 90 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (210, 210, 210),
                        1,
                    )
                if cfg.camera.show_preview:
                    cv2.imshow("Hand-Eye Calibration", display_frame)
                key = cv2.waitKey(1) & 0xFF if cfg.camera.show_preview else 255
                should_capture = key in (ord(" "), ord("s"))
                if key == ord("q"):
                    logging.info("Early exit requested by user.")
                    break
                if not should_capture and cfg.camera.auto_capture_s > 0.0:
                    should_capture = (time.time() - last_capture) >= cfg.camera.auto_capture_s

                if should_capture and detected:
                    obs = robot.get_observation()
                    gripper_to_base = kinematics.forward_kinematics_from_observation(obs)

                    if not _is_valid_transform(gripper_to_base):
                        logging.warning("FK returned invalid gripper->base transform; skipping sample.")
                        continue

                    target_to_camera = _make_homogeneous(R_tc, t_tc)

                    image_path: str | None = None
                    saved_frame_path: Path | None = None
                    if save_dir:
                        next_idx = len(samples) + 1
                        candidate_path = save_dir / f"sample_{next_idx:03d}.png"
                        while candidate_path.exists():
                            next_idx += 1
                            candidate_path = save_dir / f"sample_{next_idx:03d}.png"
                        saved_frame_path = _save_frame(display_frame, save_dir, next_idx)
                        image_path = str(saved_frame_path)
                    sample = CaptureSample(
                        gripper_to_base=gripper_to_base,
                        target_to_camera=target_to_camera,
                        corners=corners,
                        timestamp_s=time.time(),
                        decision_margin=margin,
                        image_path=image_path,
                    )
                    samples.append(sample)
                    captured_count += 1
                    last_capture = time.time()
                    if visualizer is not None:
                        try:
                            visualizer.update(obs, kinematics, gripper_to_base)
                        except Exception as viz_err:  # pragma: no cover - optional
                            logging.debug("PyBullet update failed: %s", viz_err)
                    if saved_frame_path:
                        logging.info("Sample %d stored (%s)", len(samples), saved_frame_path)
                    else:
                        logging.info("Sample %d stored.", len(samples))
                    logging.debug("Gripper->Base %s", _format_pose(gripper_to_base))
                    logging.debug("Camera<-Tag %s", _format_pose(target_to_camera))
                elif should_capture and not detected:
                    logging.warning("Capture requested but AprilTag not found.")
        except Exception as e:
            logging.error("Calibration failed during capture: %s", str(e))
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            if robot is not None:
                robot.disconnect()
            if visualizer is not None:
                visualizer.close()
    else:
        logging.info("perform_capture=False, skipping live capture.")

    total_samples = len(samples)
    if total_samples < cfg.required_samples:
        logging.warning("Stopped with %d/%d samples.", total_samples, cfg.required_samples)
    if total_samples < 3:
        raise RuntimeError(f"Need >=3 samples, captured {total_samples}")
    logging.info("Collected %d valid pose(s).", total_samples)

    if cfg.record_samples_path:
        dataset_meta = dict(loaded_metadata)
        dataset_meta.setdefault("intrinsics_path", cfg.intrinsics_path)
        dataset_meta.setdefault("tag_family", cfg.tag_family)
        dataset_meta.setdefault("tag_id", cfg.tag_id)
        dataset_meta.setdefault("tag_size_m", cfg.tag_size_m)
        dataset_meta["captured_samples"] = captured_count
        dataset_meta["loaded_samples"] = loaded_count
        dataset_meta["total_samples"] = total_samples
        try:
            record_path = Path(cfg.record_samples_path)
            _write_samples_file(samples, record_path, dataset_meta)
            logging.info("Wrote %d samples to %s", len(samples), record_path)
        except Exception as err:
            logging.error("Failed to save samples: %s", err)

    method_map = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method_flag = method_map.get(cfg.solver_method.lower())
    if method_flag is None:
        raise ValueError(f"Unsupported solver_method '{cfg.solver_method}'. Choose from {list(method_map.keys())}.")

    solver_result = _solve_hand_eye(samples, method_flag, cfg)
    used_samples = len(solver_result.samples)

    if solver_result.filtered_total:
        logging.info(
            "Filtered %d sample(s) across %d pass(es).",
            solver_result.filtered_total,
            solver_result.passes,
        )

    base_rot_errors = [e[0] for e in solver_result.base_camera_errors]
    base_trans_errors = [e[1] for e in solver_result.base_camera_errors]
    rot_err_mean = float(np.degrees(np.mean(base_rot_errors)))
    trans_err_mean = float(np.mean(base_trans_errors) * 1000.0)

    logging.info("Camera->Base (avg): %s", _format_pose(solver_result.T_cb))
    logging.info("Base->Camera: %s", _format_pose(solver_result.T_bc))
    logging.info("Average camera->base residual: %.3f deg, %.2f mm", rot_err_mean, trans_err_mean)

    tag_error_values = [err for err in solver_result.tag_errors if err is not None]
    metrics: dict[str, Any] = {
        "samples": used_samples,
        "solver_method": cfg.solver_method,
        "camera_to_base_mean_rot_deg": rot_err_mean,
        "camera_to_base_mean_trans_mm": trans_err_mean,
    }
    metrics["loaded_samples"] = loaded_count
    metrics["captured_samples"] = captured_count
    metrics["filtered_samples"] = solver_result.filtered_total
    metrics["filter_passes"] = solver_result.passes
    metrics["total_samples_before_filter"] = total_samples
    if cfg.load_samples_path:
        metrics["loaded_samples_path"] = cfg.load_samples_path
    if cfg.record_samples_path:
        metrics["recorded_samples_path"] = cfg.record_samples_path
    if tag_error_values:
        metrics["tag_pose_mean_rot_deg"] = float(np.degrees(np.mean([e[0] for e in tag_error_values])))
        metrics["tag_pose_mean_trans_mm"] = float(np.mean([e[1] for e in tag_error_values]) * 1000.0)

    output = {
        "timestamp": time.time(),
        "intrinsics_path": cfg.intrinsics_path,
        "tag_size_m": cfg.tag_size_m,
        "method": cfg.solver_method,
        "metrics": metrics,
    }
    if solver_result.T_bc is not None:
        output["camera_to_base"] = _matrix_to_list(solver_result.T_bc)
    if solver_result.T_cb is not None:
        output["base_to_camera"] = _matrix_to_list(solver_result.T_cb)
    if solver_result.T_gc is not None:
        output["camera_to_gripper"] = _matrix_to_list(solver_result.T_gc)
    if solver_result.T_cg is not None:
        output["gripper_to_camera"] = _matrix_to_list(solver_result.T_cg)
    if solver_result.T_gt is not None:
        output["tag_to_camera"] = _matrix_to_list(solver_result.T_gt)

    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logging.info("Saved calibration to %s", out_path)


if __name__ == "__main__":  # pragma: no cover
    run_hand_eye_calibration()