#!/usr/bin/env python

"""Print the end-effector pose from the URDF for a default (zero) joint pose or current robot state.

Usage examples:
  1) Pure URDF FK at zero joints (no hardware connection):
       python -m lerobot.scripts.lerobot_check_ee_pose \
         --urdf_path path/to/so101_follower.urdf \
         --target_frame_name gripper_frame_link

  2) Using a robot config to fetch current joints (will connect to hardware):
       python -m lerobot.scripts.lerobot_check_ee_pose \
         --robot.name so101_follower \
         --urdf_path path/to/so101_follower.urdf

  3) Provide an explicit joint list (degrees) overriding zero pose:
       python -m lerobot.scripts.lerobot_check_ee_pose \
         --urdf_path path/to/so101_follower.urdf \
         --joints "[0,10,-5,20,0,15]"

Notes:
  - Joint order must match the URDF joint order (or provided RobotKinematics joint_names ordering).
  - If --robot.name is supplied we attempt to read current joint positions from the device; otherwise
    we fall back to explicit --joints or zeros.
"""


import ast
import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.rotation import Rotation

from lerobot.robots import (  # noqa: F401  (expose registered robot configs for parser)
    RobotConfig,
    so101_follower,
    so100_follower,
    hope_jr,
    koch_follower,
    bi_so100_follower,
    make_robot_from_config,
)


@dataclass(kw_only=True)
class CheckEEPoseConfig:
    # Optional robot config (if provided we connect & read joints)
    robot: RobotConfig | None = None
    # Required URDF path (cannot infer reliably from all robot configs)
    urdf_path: str
    target_frame_name: str = "gripper_frame_link"
    # Explicit joints (degrees). If None and robot not given -> zeros.
    joints: str | None = None  # JSON or python-literal list
    # If true and robot provided, use live measured joints instead of zeros / provided list.
    use_robot_current: bool = True
    # Print as homogeneous matrix
    show_matrix: bool = True
    # Also print rotation vector and quaternion
    show_rotvec: bool = True
    show_quat: bool = True
    # PyBullet visualization options
    visualize_pybullet: bool = False
    pybullet_gui: bool = True  # if False and visualize_pybullet, use DIRECT (headless)
    pybullet_time_s: float = 5.0  # seconds to keep window open (GUI) / run steps
    pybullet_add_axes: bool = True  # draw colored axes at EE frame
    pybullet_show_joint_names: bool = True
    pybullet_gravity: bool = False


def _parse_joint_list(raw: str) -> list[float]:
    try:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = ast.literal_eval(raw)
    except Exception as e:  # noqa: BLE001
        raise ValueError("--joints must be a JSON or Python literal list of numbers") from e
    if not isinstance(data, (list, tuple)):
        raise ValueError("--joints must be a list")
    return [float(x) for x in data]


def _format_pose(T: np.ndarray) -> str:
    pos = T[:3, 3]
    rot = Rotation.from_matrix(T[:3, :3])
    rotvec = rot.as_rotvec()
    quat = rot.as_quat()  # [x,y,z,w]
    # Euler XYZ (intrinsic) from R = Rz * Ry * Rx
    R = T[:3, :3]
    # roll (x), pitch (y), yaw (z)
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arcsin(-R[2, 0]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    euler_deg = np.degrees([roll, pitch, yaw])
    lines = [
        f"Position (m): x={pos[0]:.6f}  y={pos[1]:.6f}  z={pos[2]:.6f}",
        f"Rotvec  (rad): [{rotvec[0]:.6f}, {rotvec[1]:.6f}, {rotvec[2]:.6f}] (angle={np.linalg.norm(rotvec):.6f})",
        f"Quat [x,y,z,w]: [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]",
        f"Euler xyz (deg): [roll={euler_deg[0]:.2f}, pitch={euler_deg[1]:.2f}, yaw={euler_deg[2]:.2f}]",
    ]
    return "\n".join(lines)


@parser.wrap()
def check_ee_pose(cfg: CheckEEPoseConfig):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Config: %s", cfg)

    robot = None
    joint_source = "zeros"
    try:
        if cfg.robot is not None and cfg.use_robot_current:
            logging.info("Connecting to robot to read current joints …")
            robot = make_robot_from_config(cfg.robot)
            robot.connect()
            obs = robot.get_observation()
            # Collect joint positions (assumed fields name.pos) in the iteration order of motors
            motor_names = list(robot.bus.motors.keys())
            joints = [float(obs.get(f"{n}.pos", 0.0)) for n in motor_names]
            joint_source = "robot_current"
        elif cfg.joints is not None:
            joints = _parse_joint_list(cfg.joints)
            motor_names = None  # Let kinematics infer or use URDF order
            joint_source = "provided_list"
        else:
            joints = []
            motor_names = None
            joint_source = "zeros"

        # If no joints provided, we will build a zero vector sized from URDF joint_names
        kinematics = RobotKinematics(
            urdf_path=cfg.urdf_path,
            target_frame_name=cfg.target_frame_name,
            joint_names=None,  # Let implementation decide ordering
        )

        if not joints:
            # Determine number of controllable joints from kinematics
            n = len(kinematics.joint_names)
            joints = [0.0] * n
            logging.info("Using zero joint pose (%d joints) from URDF", n)

        T = kinematics.forward_kinematics(np.array(joints, dtype=float))

        print("=== End-Effector Pose (source=%s) ===" % joint_source)
        # Joint angles (deg) listing
        jnames = getattr(kinematics, "joint_names", [])
        if jnames and joints:
            print("Joint angles (deg):")
            for i, name in enumerate(jnames):
                if i < len(joints):
                    print(f"  {name}: {joints[i]:.3f}")
            compact = ", ".join(f"{joints[i]:.3f}" for i in range(min(len(joints), len(jnames))))
            print(f"Joint list (deg): [{compact}]")
        if cfg.show_matrix:
            print("Homogeneous transform (4x4):")
            with np.printoptions(precision=6, suppress=True):
                print(T)
        summary = _format_pose(T)
        print(summary)

        # Optional PyBullet visualization
        if cfg.visualize_pybullet:
            try:
                import pybullet as p  # type: ignore
                import pybullet_data  # type: ignore
            except ImportError:
                logging.error("pybullet not installed. pip install pybullet to enable visualization")
            else:
                connection_mode = p.GUI if cfg.pybullet_gui else p.DIRECT
                cid = p.connect(connection_mode)
                if cid < 0:
                    logging.error("Failed to connect to PyBullet")
                else:
                    logging.info("PyBullet connected (GUI=%s)", cfg.pybullet_gui)
                    p.setAdditionalSearchPath(pybullet_data.getDataPath())
                    if cfg.pybullet_gravity:
                        p.setGravity(0, 0, -9.81)
                    plane_id = p.loadURDF("plane.urdf") if cfg.pybullet_gravity else None
                    # Load robot URDF
                    robot_id = p.loadURDF(cfg.urdf_path, useFixedBase=True)

                    # Map joint names -> index
                    name_to_index = {}
                    num_joints = p.getNumJoints(robot_id)
                    for j in range(num_joints):
                        info = p.getJointInfo(robot_id, j)
                        jname = info[1].decode("utf-8")
                        name_to_index[jname] = j
                        if cfg.pybullet_show_joint_names:
                            p.addUserDebugText(jname, p.getLinkState(robot_id, j)[0], textColorRGB=[0.7, 0.7, 0.7], lifeTime=cfg.pybullet_time_s)

                    # Attempt to set joint states (assuming revolute/prismatic)
                    # Convert degrees to radians for pybullet
                    joints_rad = np.radians(joints)
                    # Use ordering from kinematics.joint_names when possible
                    for i, jdeg in enumerate(joints):
                        if i < len(kinematics.joint_names):
                            jname = kinematics.joint_names[i]
                            if jname in name_to_index:
                                p.resetJointState(robot_id, name_to_index[jname], joints_rad[i])

                    # Find EE link index (try exact match first)
                    ee_index = None
                    for j in range(num_joints):
                        info = p.getJointInfo(robot_id, j)
                        link_name = info[12].decode("utf-8")
                        if link_name == cfg.target_frame_name:
                            ee_index = j
                            break
                    if ee_index is None:
                        logging.warning("Could not find target_frame_name '%s' in pybullet links", cfg.target_frame_name)
                    else:
                        ls = p.getLinkState(robot_id, ee_index, computeForwardKinematics=True)
                        pos_pb = ls[4]  # worldLinkFramePosition
                        orn_pb = ls[5]  # worldLinkFrameOrientation quaternion (x,y,z,w)
                        logging.info("PyBullet EE position: (%.4f, %.4f, %.4f)", *pos_pb)
                        logging.info("PyBullet EE quat: (%.4f, %.4f, %.4f, %.4f)", *orn_pb)

                        if cfg.pybullet_add_axes:
                            # Draw axes lines from EE frame using orientation
                            axis_len = 0.05
                            # Use our Rotation utility to get a rotation matrix from quaternion
                            R = Rotation.from_quat(np.array(orn_pb)).as_matrix()
                            origin = np.array(pos_pb)
                            axes = [R[:, 0], R[:, 1], R[:, 2]]
                            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                            for a, c in zip(axes, colors):
                                p.addUserDebugLine(origin, origin + axis_len * a, c, lineWidth=2, lifeTime=cfg.pybullet_time_s)

                    # Simple wait loop (if GUI) or step simulation headless
                    import time
                    t_end = time.time() + max(0.0, cfg.pybullet_time_s)
                    while time.time() < t_end:
                        p.stepSimulation()
                        if not cfg.pybullet_gui:
                            time.sleep(1.0 / 240.0)
                    p.disconnect()
                    logging.info("PyBullet visualization finished")

    finally:
        if robot is not None:
            robot.disconnect()


def main():  # pragma: no cover - entry point
    check_ee_pose()


if __name__ == "__main__":  # pragma: no cover
    main()
