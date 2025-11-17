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

"""Scripted Cartesian waypoint runner.

Features:
    - Absolute EE waypoints (x,y,z meters + orientation as rotvec or rpy_deg).
    - Optional per-waypoint gripper_pos and hold_s dwell.
    - Duration-based or speed-based interpolation (linear + orientation). 
    - Choice of interpolation space: Cartesian (IK every step) or joint (IK once then joint lerp).
    - Easing functions for smooth ramp (linear | smoothstep | ease_in_out_quad).
    - Early finish when EE pose converges within tolerances for N consecutive frames.
    - Optional EE workspace bounds.

Key config fields:
    fps: control loop pacing (Hz).
    move_duration_s: fixed duration override; else derived from linear/angular speeds.
    linear_speed_mps / angular_speed_radps: used if move_duration_s not provided.
    interpolate: False -> single IK jump per waypoint; True -> multi-step interpolation.
    interpolation_mode: 'cartesian' or 'joint'.
    interpolation_easing: easing profile for alpha progression.
    ee_pos_tolerance_m / ee_ori_tolerance_rad / target_settle_frames: early finish criteria.

Usage (inline JSON example):
    python -m lerobot.scripts.lerobot_scripted_cartesian \
        --robot.name so101_follower --robot.port=/dev/ttyUSB0 \
        --robot.id=follower_arm \ 
        --urdf_path=path/to/so101_follower.urdf \
        --waypoints='[{"x":0.1,"y":0.0,"z":0.15,"rpy_deg":[0,0,0]}]' \
"""

import json
import ast
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math
import numpy as np

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.rotation import Rotation
from lerobot.utils.utils import init_logging


from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    InverseKinematicsEEToJoints,
)

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

@dataclass(kw_only=True)
class ScriptedCartesianConfig:
    robot: RobotConfig

    # IK/URDF
    urdf_path: str | None = None  # If None, will try to use robot config's urdf_path when available
    target_frame_name: str = "gripper_frame_link"
    # Optional joint zero offset JSON (persisted calibration). If provided and exists, it will be loaded.
    zero_offset_json_path: str | None = None

    # Waypoints can be provided inline (as a JSON string) or via a JSON file
    waypoints_path: str | None = None
    waypoints: str | None = None  # JSON string: a list of waypoint dicts

    # Optional EE bounds (meters). Accept JSON strings like "[0.3, -0.2, 0.1]" to avoid CLI parsing issues.
    ee_bounds_min: str | None = None
    ee_bounds_max: str | None = None

    # Dwell behavior and simple pacing
    default_hold_s: float = 0.0  # seconds to dwell after each waypoint if hold_s not specified
    fps: int = 30  # pacing for the control loop while holding
    # Movement timing between waypoints
    move_duration_s: float | None = None  # If set, use this duration to move between waypoints
    linear_speed_mps: float | None = 0.01  # If duration not set, use speed to derive duration
    angular_speed_radps: float | None = 0.05
    interpolate: bool = True  # If False, send only final target per waypoint
    interpolation_mode: str = "cartesian"  # 'cartesian' or 'joint'
    # Easing profile for interpolation fraction alpha: 'linear' | 'smoothstep' | 'ease_in_out_quad'
    interpolation_easing: str = "linear"  # motion profile shaping
    # Early-finish criteria: stop moving as soon as EE pose is close enough
    ee_pos_tolerance_m: float = 0.01  # translational convergence threshold (meters)
    ee_ori_tolerance_rad: float = 0.08  # rotational convergence threshold (radians)
    target_settle_frames: int = 5  # consecutive frames within tolerance before early finish


def _apply_easing(x: float, mode: str) -> float:
    """Apply easing to interpolation fraction x in [0,1]."""
    x = float(max(0.0, min(1.0, x)))
    m = (mode or "linear").lower()
    if m == "linear":
        return x
    if m == "smoothstep":
        # 3x^2 - 2x^3
        return x * x * (3.0 - 2.0 * x)
    if m == "ease_in_out_quad":
        # piecewise quadratic ease-in-out
        return 2.0 * x * x if x < 0.5 else 1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    # Fallback
    return x


def _load_waypoints(cfg: ScriptedCartesianConfig) -> list[dict[str, Any]]:
    if cfg.waypoints_path is not None:
        p = Path(cfg.waypoints_path)
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError("waypoints_path must contain a JSON list of waypoints")
        return data
    if cfg.waypoints is not None:
        # Accept strict JSON or Python-literal style (single quotes) for convenience
        try:
            data = json.loads(cfg.waypoints)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(cfg.waypoints)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    "--waypoints must be a JSON list (use double quotes) or a Python literal list.\n"
                    "Example JSON: \n"
                    "  --waypoints=\"[ {\\\"x\\\": 0.4, \\\"y\\\": 0.0, \\\"z\\\": 0.25} ]\"\n"
                ) from e
        if not isinstance(data, list):
            raise ValueError("--waypoints must be a JSON list of waypoint objects")
        return data
    return []


def _maybe_parse_bounds(v: list[float] | str | None) -> list[float] | None:
    if v is None:
        return None
    if isinstance(v, str):
        # Accept JSON or Python-literal list
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(v)
        if not isinstance(parsed, list) or len(parsed) != 3:
            raise ValueError("Bounds must be a JSON list of 3 floats, e.g. [0.3, -0.2, 0.1]")
        return [float(parsed[0]), float(parsed[1]), float(parsed[2])]
    return [float(v[0]), float(v[1]), float(v[2])]


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions [x,y,z,w]."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # Linear fallback
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = s0 * q0 + s1 * q1
    return q / np.linalg.norm(q)


def _rotvec_slerp(w0: np.ndarray, w1: np.ndarray, t: float) -> np.ndarray:
    r0 = Rotation.from_rotvec(w0)
    r1 = Rotation.from_rotvec(w1)
    q0 = r0.as_quat()
    q1 = r1.as_quat()
    qt = _quat_slerp(q0, q1, t)
    return Rotation.from_quat(qt).as_rotvec()


def _resolve_urdf(cfg: ScriptedCartesianConfig, robot: Robot) -> tuple[str, str]:
    # Prefer explicit override
    if cfg.urdf_path is not None:
        return cfg.urdf_path, cfg.target_frame_name

    # Try to read from robot-specific config if available (e.g., SO101FollowerConfig)
    urdf = getattr(cfg.robot, "urdf_path", None)
    target = getattr(cfg.robot, "target_frame_name", cfg.target_frame_name)
    if urdf is None:
        raise ValueError(
            "URDF path is required. Provide --urdf_path or ensure robot config exposes 'urdf_path'."
        )
    return urdf, target


def _build_pipeline(
    kinematics: RobotKinematics,
    motor_names: list[str],
    cfg: ScriptedCartesianConfig,
) -> RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction]:
    steps = []

    # Optional bounds
    bmin = _maybe_parse_bounds(cfg.ee_bounds_min)
    bmax = _maybe_parse_bounds(cfg.ee_bounds_max)
    if bmin is not None and bmax is not None:
        steps.append(
            EEBoundsAndSafety(
                end_effector_bounds={"min": bmin, "max": bmax}
            )
        )

    steps.append(
        InverseKinematicsEEToJoints(
            kinematics=kinematics,
            motor_names=motor_names,
            initial_guess_current_joints=True,
        )
    )

    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=steps,
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def _to_action_dict(wp: dict[str, Any], default_gripper: float) -> dict[str, Any]:
    # Required position
    x = float(wp.get("x"))
    y = float(wp.get("y"))
    z = float(wp.get("z"))

    # Orientation: rotvec wx/wy/wz or rpy_deg -> rotvec
    if all(k in wp for k in ("wx", "wy", "wz")):
        wx = float(wp["wx"])
        wy = float(wp["wy"])
        wz = float(wp["wz"])
    elif "rpy_deg" in wp:
        rpy = wp["rpy_deg"]
        if not (isinstance(rpy, (list, tuple)) and len(rpy) == 3):
            raise ValueError("rpy_deg must be a 3-element list [roll_deg, pitch_deg, yaw_deg]")
        rv = Rotation.from_euler_xyz([rpy[0], rpy[1], rpy[2]], degrees=True).as_rotvec()
        wx, wy, wz = float(rv[0]), float(rv[1]), float(rv[2])
    else:
        # Default to no rotation change
        wx = wy = wz = 0.0

    gripper_pos = float(wp.get("gripper_pos", default_gripper))

    return {
        "ee.x": x,
        "ee.y": y,
        "ee.z": z,
        "ee.wx": wx,
        "ee.wy": wy,
        "ee.wz": wz,
        "ee.gripper_pos": gripper_pos,
    }


def _hold_until(cfg: ScriptedCartesianConfig, hold_s: float):
    # Simple time-based dwell. Could be extended with joint/EE tolerance checks.
    busy_wait(max(hold_s, 0.0))


@parser.wrap()
def run_scripted_cartesian(cfg: ScriptedCartesianConfig):
    init_logging()
    logging.info(asdict(cfg))

    # Init robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        # Build kinematics
        urdf_path, target_frame = _resolve_urdf(cfg, robot)
        motor_names = list(robot.bus.motors.keys())
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=motor_names,
        )

        # Load persisted joint zero offset if provided
        if cfg.zero_offset_json_path is not None:
            try:
                p = Path(cfg.zero_offset_json_path)
                if p.exists():
                    kinematics.load_joint_zero_offset(str(p))
                    logging.info("Loaded joint zero offset from %s", str(p))
                else:
                    logging.warning("zero_offset_json_path provided but file does not exist: %s", str(p))
            except Exception as e:  # noqa: BLE001
                logging.error("Failed to load zero offset from %s: %s", cfg.zero_offset_json_path, e)

        # Build processing pipeline
        pipeline = _build_pipeline(kinematics, motor_names, cfg)
        
        # Load waypoints
        waypoints = _load_waypoints(cfg)
        if len(waypoints) == 0:
            raise ValueError("No waypoints provided. Pass --waypoints_path or inline --waypoints list.")

        # Establish a default gripper position from current state if needed
        obs = robot.get_observation()
        default_gripper = float(obs.get("gripper.pos", 0.0))

        # Determine starting EE pose from current FK
        def _current_ee_from_obs(o: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
            """Helper using kinematics observation utilities (zero-offset aware)."""
            t = kinematics.forward_kinematics_from_observation(o)
            pos = t[:3, 3]
            rotvec = Rotation.from_matrix(t[:3, :3]).as_rotvec()
            grip = float(o.get("gripper.pos", default_gripper))
            return pos, rotvec, grip

        # Optionally log initial EE pose for diagnostics
        initial_ee = _current_ee_from_obs(obs)
        logging.info(
            "Initial EE: pos=(%.4f, %.4f, %.4f), rotvec=(%.4f, %.4f, %.4f), grip=%.3f",
            initial_ee[0][0], initial_ee[0][1], initial_ee[0][2],
            initial_ee[1][0], initial_ee[1][1], initial_ee[1][2],
            float(initial_ee[2]),
        )

        # Also log current joint angles (absolute and zeroed) to verify calibration
        try:
            abs_deg = kinematics.observation_to_joint_array(obs)
            rel_deg = kinematics.apply_zero_offset(abs_deg)
            abs_named = ", ".join(f"{name}={abs_deg[i]:.2f}" for i, name in enumerate(motor_names))
            rel_named = ", ".join(f"{name}={rel_deg[i]:.2f}" for i, name in enumerate(motor_names))
            logging.info("Initial joints abs (deg): %s", abs_named)
            logging.info("Initial joints zeroed (deg): %s", rel_named)
        except Exception as e:  # noqa: BLE001
            logging.warning("Failed to log initial joint angles: %s", e)

        obs0 = robot.get_observation()
        pos_curr, w_curr, grip_curr = _current_ee_from_obs(obs0)

        for idx, wp in enumerate(waypoints):
            target = _to_action_dict(wp, default_gripper)
            pos_goal = np.array([target["ee.x"], target["ee.y"], target["ee.z"]], dtype=float)
            w_goal = np.array([target["ee.wx"], target["ee.wy"], target["ee.wz"]], dtype=float)
            grip_goal = float(target["ee.gripper_pos"])

            # Determine movement duration (explicit move_s overrides speed-based inference)
            move_s = float(wp.get("move_s", cfg.move_duration_s or 0.0))
            if move_s <= 0.0:
                lin_dist = float(np.linalg.norm(pos_goal - pos_curr))
                rel_rot = Rotation.from_rotvec(w_goal) * Rotation.from_rotvec(w_curr).inv()
                ang_dist = float(np.linalg.norm(rel_rot.as_rotvec()))
                t_lin = lin_dist / cfg.linear_speed_mps if cfg.linear_speed_mps else 0.0
                t_ang = ang_dist / cfg.angular_speed_radps if cfg.angular_speed_radps else 0.0
                # At least one frame of motion
                move_s = max(t_lin, t_ang, 1.0 / cfg.fps)

            steps = 1 if not cfg.interpolate else max(1, int(math.ceil(move_s * cfg.fps)))

            if cfg.interpolation_mode.lower() == "joint":
                # Compute joint-space endpoints
                # Current joints from obs
                obs = robot.get_observation()
                q_curr = kinematics.observation_to_joint_array(obs)
                # IK for goal EE pose
                t_goal = np.eye(4, dtype=float)
                t_goal[:3, :3] = Rotation.from_rotvec(w_goal).as_matrix()
                t_goal[:3, 3] = pos_goal
                q_goal = kinematics.inverse_kinematics(q_curr, t_goal)

                settle_count = 0
                for step in range(1, steps + 1):
                    alpha_lin = step / steps if steps > 1 else 1.0
                    alpha = _apply_easing(alpha_lin, cfg.interpolation_easing)
                    q_t = (1 - alpha) * q_curr + alpha * q_goal
                    grip_t = (1 - alpha) * grip_curr + alpha * grip_goal

                    # Command joints directly
                    action_joint: dict[str, float] = {f"{name}.pos": float(q_t[i]) for i, name in enumerate(motor_names) if name != "gripper"}
                    action_joint["gripper.pos"] = float(grip_t)

                    # Send command
                    _ = robot.send_action(action_joint)
                    busy_wait(1.0 / cfg.fps)

                    # Measure current EE pose and check tolerance
                    obs_meas = robot.get_observation()
                    pos_meas, w_meas, _ = _current_ee_from_obs(obs_meas)
                    pos_err = float(np.linalg.norm(pos_goal - pos_meas))
                    rel_rot = Rotation.from_rotvec(w_goal) * Rotation.from_rotvec(w_meas).inv()
                    ang_err = float(np.linalg.norm(rel_rot.as_rotvec()))
                    if pos_err <= cfg.ee_pos_tolerance_m and ang_err <= cfg.ee_ori_tolerance_rad:
                        settle_count += 1
                    else:
                        settle_count = 0
                    if settle_count >= max(1, int(cfg.target_settle_frames)):
                        break
            else:
                # Cartesian-space interpolation (default)

                settle_count = 0
                for step in range(1, steps + 1):
                    alpha_lin = step / steps if steps > 1 else 1.0
                    alpha = _apply_easing(alpha_lin, cfg.interpolation_easing)
                    pos_t = (1 - alpha) * pos_curr + alpha * pos_goal
                    w_t = _rotvec_slerp(w_curr, w_goal, alpha) if steps > 1 else w_goal
                    grip_t = (1 - alpha) * grip_curr + alpha * grip_goal

                    action_ee = {
                        "ee.x": float(pos_t[0]),
                        "ee.y": float(pos_t[1]),
                        "ee.z": float(pos_t[2]),
                        "ee.wx": float(w_t[0]),
                        "ee.wy": float(w_t[1]),
                        "ee.wz": float(w_t[2]),
                        "ee.gripper_pos": float(grip_t),
                    }

                    obs = robot.get_observation()
                    joint_action = pipeline((action_ee, obs))
                    _ = robot.send_action(joint_action)

                    # Pace
                    busy_wait(1.0 / cfg.fps)

                    # Measure current EE pose and check tolerance
                    obs_meas = robot.get_observation()
                    pos_meas, w_meas, _ = _current_ee_from_obs(obs_meas)
                    pos_err = float(np.linalg.norm(pos_goal - pos_meas))
                    rel_rot = Rotation.from_rotvec(w_goal) * Rotation.from_rotvec(w_meas).inv()
                    ang_err = float(np.linalg.norm(rel_rot.as_rotvec()))
                    logging.debug(
                        "wp %d/%d step %d/%d pos_err=%.4f ang_err=%.4f", idx + 1, len(waypoints), step, steps, pos_err, ang_err
                    )
                    if pos_err <= cfg.ee_pos_tolerance_m and ang_err <= cfg.ee_ori_tolerance_rad:
                        settle_count += 1
                    else:
                        settle_count = 0
                    if settle_count >= max(1, int(cfg.target_settle_frames)):
                        break

            logging.info(
                f"Waypoint {idx+1}/{len(waypoints)} reached: pos=({pos_goal[0]:.3f},{pos_goal[1]:.3f},{pos_goal[2]:.3f})"
            )

            # Update current pose state
            pos_curr, w_curr, grip_curr = pos_goal, w_goal, grip_goal

            # Dwell at waypoint
            hold_s = float(wp.get("hold_s", cfg.default_hold_s))
            _hold_until(cfg, hold_s)

        logging.info("Scripted Cartesian sequence complete.")

    finally:
        robot.disconnect()


def main():
    run_scripted_cartesian()


if __name__ == "__main__":
    main()
