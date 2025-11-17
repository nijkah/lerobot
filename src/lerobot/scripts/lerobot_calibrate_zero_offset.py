#!/usr/bin/env python
"""Calibrate and persist joint zero offsets for a robot.

Usage examples:
    # Capture current joint angles as zero and save JSON
    python -m lerobot.scripts.lerobot_calibrate_zero_offset \
        --robot.name so101_follower --robot.port /dev/ttyUSB0 \
        --robot.id follower_arm \
        --urdf_path path/to/so101_follower.urdf \
        --output_json so101_zero_offset.json

    # Dry run (print only, do not save)
    python -m lerobot.scripts.lerobot_calibrate_zero_offset --dry_run true ...

    # Show FK of current pose before and after applying zero offset
    python -m lerobot.scripts.lerobot_calibrate_zero_offset --show_fk true ...

What it does:
    1. Connects to the robot and reads current joint positions (degrees).
    2. Initializes RobotKinematics with URDF + joint names.
    3. Optionally computes FK for the current pose (absolute reference).
    4. Calls set_joint_zero_offset so current pose becomes internal zero.
    5. Optionally computes FK again using the same absolute joint positions to show pose stability.
    6. Saves JSON with joint names + zero offset degrees unless --dry_run.

JSON schema saved:
    {
      "joint_names": ["joint1", "joint2", ...],
      "joint_zero_offset_deg": [12.3, -4.5, ...]
    }

After saving, pass --zero_offset_json_path to lerobot_scripted_cartesian or other scripts using RobotKinematics.
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import numpy as np

from lerobot.configs import parser
from lerobot.utils.utils import init_logging
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    koch_follower,
    hope_jr,
    bi_so100_follower,
)


@dataclass(kw_only=True)
class CalibrateZeroOffsetConfig:
    robot: RobotConfig
    urdf_path: str | None = None
    target_frame_name: str = "gripper_frame_link"
    output_json: str = "zero_offset.json"
    dry_run: bool = False
    show_fk: bool = True


def _resolve_urdf(cfg: CalibrateZeroOffsetConfig, robot: Any) -> tuple[str, str]:
    if cfg.urdf_path is not None:
        return cfg.urdf_path, cfg.target_frame_name
    urdf = getattr(cfg.robot, "urdf_path", None)
    target = getattr(cfg.robot, "target_frame_name", cfg.target_frame_name)
    if urdf is None:
        raise ValueError("URDF path is required. Provide --urdf_path or ensure robot config exposes 'urdf_path'.")
    return urdf, target


@parser.wrap()
def calibrate_zero_offset(cfg: CalibrateZeroOffsetConfig):
    init_logging()
    logging.info(asdict(cfg))

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        urdf_path, target_frame = _resolve_urdf(cfg, robot)
        motor_names = list(robot.bus.motors.keys())
        kinematics = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=motor_names,
        )

        # Capture current joint positions
        obs = robot.get_observation()
        joint_pos_deg = np.array(
            [float(obs[f"{name}.pos"]) for name in motor_names if f"{name}.pos" in obs],
            dtype=float,
        )
        logging.info("Captured %d joints: %s", len(joint_pos_deg), joint_pos_deg.tolist())

        if cfg.show_fk:
            T_before = kinematics.forward_kinematics(joint_pos_deg)
            logging.info("FK before calibration (world->EE):\n%s", np.array2string(T_before, precision=4))

        kinematics.set_joint_zero_offset(joint_pos_deg)

        if cfg.show_fk:
            # Forward kinematics with the same absolute joints should be unchanged
            T_after = kinematics.forward_kinematics(joint_pos_deg)
            logging.info("FK after calibration (should match before):\n%s", np.array2string(T_after, precision=4))

        if not cfg.dry_run:
            out_path = Path(cfg.output_json)
            kinematics.save_joint_zero_offset(str(out_path))
            logging.info("Zero offset JSON saved to %s", str(out_path.resolve()))
        else:
            logging.info("Dry run: zero offset NOT saved")

        logging.info("Calibration complete.")
    finally:
        robot.disconnect()


def main():
    calibrate_zero_offset()


if __name__ == "__main__":
    main()
