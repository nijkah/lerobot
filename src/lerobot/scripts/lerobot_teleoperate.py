# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    MapDeltaActionToRobotActionStep,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
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
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def _setup_keyboard_end_effector_pipeline(
    cfg: TeleoperateConfig,
    robot: Robot,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
) -> tuple[
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
    bool,
]:
    """Configure processors so a keyboard can drive an SO101 follower in end-effector space."""

    if not isinstance(cfg.teleop, KeyboardEndEffectorTeleopConfig):
        return teleop_action_processor, robot_action_processor, robot_observation_processor, False

    if not isinstance(cfg.robot, SO101FollowerConfig) or not isinstance(robot, SO101Follower):
        raise ValueError("keyboard_ee teleoperation currently requires robot.type=so101_follower")

    if cfg.robot.urdf_path is None:
        raise ValueError(
            "keyboard_ee teleoperation requires providing --robot.urdf_path with a valid SO-ARM URDF file"
        )

    motor_names = list(robot.bus.motors.keys())
    kinematics = RobotKinematics(
        urdf_path=str(cfg.robot.urdf_path),
        target_frame_name=cfg.robot.target_frame_name,
        joint_names=motor_names,
    )

    step_sizes = dict(cfg.robot.keyboard_end_effector_step_sizes)
    bounds_cfg = cfg.robot.keyboard_end_effector_bounds
    if "min" not in bounds_cfg or "max" not in bounds_cfg:
        raise ValueError("keyboard_end_effector_bounds must define 'min' and 'max' entries")
    bounds = {
        "min": list(bounds_cfg["min"]),
        "max": list(bounds_cfg["max"]),
    }

    teleop_action_processor = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapDeltaActionToRobotActionStep(
                position_scale=cfg.teleop.position_scale,
                rotation_scale=cfg.teleop.rotation_scale,
                noise_threshold=cfg.teleop.noise_threshold,
            ),
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes=step_sizes,
                motor_names=motor_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=bounds,
                max_ee_step_m=cfg.robot.keyboard_end_effector_max_step_m,
            ),
            GripperVelocityToJoint(
                speed_factor=cfg.robot.keyboard_gripper_speed_factor,
                clip_min=cfg.robot.keyboard_gripper_clip_min,
                clip_max=cfg.robot.keyboard_gripper_clip_max,
                discrete_gripper=cfg.teleop.use_gripper,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    teleop_action_processor.reset()
    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics,
                motor_names=motor_names,
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )
    robot_observation_processor.reset()
    return teleop_action_processor, robot_action_processor, robot_observation_processor, True


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    keyboard_ee_active: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
        keyboard_ee_active: When True, ensures keyboard end-effector commands include required fields.
    """

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        if keyboard_ee_active:
            raw_action = {} if raw_action is None else dict(raw_action)
            raw_action.setdefault("delta_x", 0.0)
            raw_action.setdefault("delta_y", 0.0)
            raw_action.setdefault("delta_z", 0.0)
            raw_action.setdefault("delta_wx", 0.0)
            raw_action.setdefault("delta_wy", 0.0)
            raw_action.setdefault("delta_wz", 0.0)
            raw_action.setdefault("gripper", 0.0)
            for key in ("delta_x", "delta_y", "delta_z", "delta_wx", "delta_wy", "delta_wz", "gripper"):
                value = raw_action.get(key)
                raw_action[key] = float(value) if value is not None else 0.0

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return dict[str, Any])
        _ = robot.send_action(robot_action_to_send)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
            )

            ee_keys = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
            if all(k in obs_transition for k in ee_keys):
                pos = (obs_transition["ee.x"], obs_transition["ee.y"], obs_transition["ee.z"])
                rot = (obs_transition["ee.wx"], obs_transition["ee.wy"], obs_transition["ee.wz"])
                print(
                    f"EE pos [m]: ({pos[0]: .3f}, {pos[1]: .3f}, {pos[2]: .3f}) | "
                    f"rotvec [rad]: ({rot[0]: .3f}, {rot[1]: .3f}, {rot[2]: .3f})"
                )
                move_cursor_up(1)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        loop_s = time.perf_counter() - loop_start
        # TODO: delete TODOl
        # print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
        keyboard_ee_active,
    ) = _setup_keyboard_end_effector_pipeline(
        cfg,
        robot,
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    )

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            keyboard_ee_active=keyboard_ee_active,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_devices()
    teleoperate()


if __name__ == "__main__":
    main()
