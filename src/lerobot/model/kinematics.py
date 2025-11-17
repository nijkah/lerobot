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

import numpy as np


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path (str): Path to the robot URDF file
            target_frame_name (str): Name of the end-effector frame in the URDF
            joint_names (list[str] | None): List of joint names to use for the kinematics solver
        """
        try:
            import placo  # type: ignore[import-not-found] # C++ library with Python bindings, no type stubs available. TODO: Create stub file or request upstream typing support.
        except ImportError as e:
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)  # Fix the base

        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Joint zero offset (degrees). If non-zero, the kinematics will treat the provided
        # joint positions as (q_provided - offset) internally. This lets you declare the
        # current physical posture as the new "zero" without editing the URDF. Calibrate by
        # calling `set_joint_zero_offset(current_joint_pos_deg)` once at startup.
        self.joint_zero_offset_deg = np.zeros(len(self.joint_names), dtype=float)

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration given the target frame name in the constructor.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Apply zero offset: interpret input as absolute degrees referenced to original robot base.
        # Internally subtract offset before computing FK.
        joint_rel_deg = joint_pos_deg[: len(self.joint_names)] - self.joint_zero_offset_deg
        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_rel_deg)

        # Update joint positions in placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])

        # Update kinematics
        self.robot.update_kinematics()

        # Get the transformation matrix
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using placo solver.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Apply zero offset before forming the initial guess
        current_rel_deg = current_joint_pos[: len(self.joint_names)] - self.joint_zero_offset_deg
        current_joint_rad = np.deg2rad(current_rel_deg)

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])

        # Update the target pose for the frame task
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task based on position_only flag
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        # Solve IK
        self.solver.solve(True)
        self.robot.update_kinematics()

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)

        # Convert back to degrees and add offset back to return absolute positions
        joint_pos_deg = np.rad2deg(joint_pos_rad) + self.joint_zero_offset_deg

        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg

    def set_joint_zero_offset(self, joint_pos_deg: np.ndarray):
        """Calibrate the kinematics so that the provided absolute joint positions become the new zeros.

        After calling this, supplying the same joint_pos_deg to forward_kinematics will be treated
        as all zeros internally. Inverse kinematics solutions will be returned in the original absolute
        frame (offset added back).

        Args:
            joint_pos_deg: Array of current absolute joint angles (degrees) of length >= len(joint_names).
        """
        if joint_pos_deg is None:
            raise ValueError("joint_pos_deg is required for zero offset calibration")
        if len(joint_pos_deg) < len(self.joint_names):
            raise ValueError(
                f"Provided joint_pos_deg length {len(joint_pos_deg)} < number of kinematic joints {len(self.joint_names)}"
            )
        self.joint_zero_offset_deg = np.array(joint_pos_deg[: len(self.joint_names)], dtype=float)
        # Optional: log calibration details
        try:
            import logging as _logging
            _logging.info(
                "Set joint zero offset (deg) for %d joints: %s", len(self.joint_names), self.joint_zero_offset_deg.tolist()
            )
        except Exception:  # noqa: BLE001
            pass

    def save_joint_zero_offset(self, path: str):
        """Persist current joint zero offset to a JSON file.

        JSON schema:
        {
            "joint_names": ["joint1", "joint2", ...],
            "joint_zero_offset_deg": [0.0, 12.3, ...]
        }
        """
        import json, os
        data = {
            "joint_names": self.joint_names,
            "joint_zero_offset_deg": self.joint_zero_offset_deg.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            import logging as _logging
            _logging.info("Saved joint zero offset to %s", os.path.abspath(path))
        except Exception:  # noqa: BLE001
            pass

    def load_joint_zero_offset(self, path: str):
        """Load joint zero offset from JSON and apply it.

        Expects same schema produced by save_joint_zero_offset.
        Extra joints in file are ignored; missing joints raise an error.
        """
        import json, os
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_joint_names = data.get("joint_names")
        offsets = data.get("joint_zero_offset_deg")
        if not isinstance(file_joint_names, list) or not isinstance(offsets, list):
            raise ValueError("Invalid zero offset JSON format")
        if len(file_joint_names) != len(offsets):
            raise ValueError("joint_names and joint_zero_offset_deg length mismatch in zero offset file")
        # Ensure ordering matches current kinematics joint_names
        if file_joint_names != self.joint_names:
            # Attempt to map if they contain same set
            if set(file_joint_names) != set(self.joint_names):
                raise ValueError("Joint names in zero offset file do not match current kinematics")
            name_to_offset = {n: offsets[i] for i, n in enumerate(file_joint_names)}
            ordered_offsets = [name_to_offset[n] for n in self.joint_names]
            self.joint_zero_offset_deg = np.array(ordered_offsets, dtype=float)
        else:
            self.joint_zero_offset_deg = np.array(offsets, dtype=float)
        try:
            import logging as _logging
            _logging.info(
                "Loaded joint zero offset from %s: %s", os.path.abspath(path), self.joint_zero_offset_deg.tolist()
            )
        except Exception:  # noqa: BLE001
            pass

    # -------- Observation helpers --------
    def observation_to_joint_array(self, observation: dict, include_gripper: bool = True) -> np.ndarray:
        """Extract joint array (deg) from a robot observation dict using this kinematics' joint order.

        Args:
            observation: Mapping with keys like "<joint>.pos" containing degrees
            include_gripper: If True and a gripper name is in joint_names, include it; otherwise ignore

        Returns:
            np.ndarray of shape (N,) with degrees in the order of self.joint_names (optionally including gripper)
        """
        vals: list[float] = []
        for name in self.joint_names:
            if (not include_gripper) and name.lower().startswith("gripper"):
                # Still append a value (0) to preserve vector length/order
                vals.append(0.0)
                continue
            key = f"{name}.pos"
            if key not in observation:
                raise KeyError(f"Observation missing key: {key}")
            vals.append(float(observation[key]))
        return np.array(vals, dtype=float)

    def apply_zero_offset(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Convert absolute joint positions (deg) to zero-offset-applied values (deg) used internally for FK.

        Equivalent to (joint_pos_deg - joint_zero_offset_deg) for the controlled joints length.
        """
        return np.array(joint_pos_deg[: len(self.joint_names)], dtype=float) - self.joint_zero_offset_deg

    def unapply_zero_offset(self, joint_pos_rel_deg: np.ndarray) -> np.ndarray:
        """Convert zero-offset-relative joints (deg) back to absolute (deg)."""
        return np.array(joint_pos_rel_deg[: len(self.joint_names)], dtype=float) + self.joint_zero_offset_deg

    def observation_to_zeroed_joint_array(self, observation: dict, include_gripper: bool = True) -> np.ndarray:
        """Convenience: observation -> absolute joints (deg) -> apply zero offset -> relative joints (deg)."""
        abs_deg = self.observation_to_joint_array(observation, include_gripper=include_gripper)
        return self.apply_zero_offset(abs_deg)

    def forward_kinematics_from_observation(self, observation: dict, include_gripper: bool = True) -> np.ndarray:
        """Compute EE transform directly from a robot observation dict.

        Uses absolute joints extracted from the observation; forward_kinematics internally applies zero offset.
        """
        abs_deg = self.observation_to_joint_array(observation, include_gripper=include_gripper)
        return self.forward_kinematics(abs_deg)
