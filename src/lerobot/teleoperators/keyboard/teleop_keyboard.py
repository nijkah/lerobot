#!/usr/bin/env python

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

import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig

PYNPUT_AVAILABLE = True
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )

        if PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logging.info("pynput not available - skipping local keyboard listener.")
            self.listener = None

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        key_id = self._normalize_key(key)
        if key_id is not None:
            self.event_queue.put((key_id, True))

    def _on_release(self, key):
        key_id = self._normalize_key(key)
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()
            return

        if key_id is not None:
            self.event_queue.put((key_id, False))

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_id, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_id] = True
            else:
                self.current_pressed.pop(key_id, None)

    @staticmethod
    def _normalize_key(key):
        if key is None:
            return None

        char = getattr(key, "char", None)
        if char is not None:
            return char.lower()

        return key

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        names = {
            "delta_x": 0,
            "delta_y": 1,
            "delta_z": 2,
            "delta_wx": 3,
            "delta_wy": 4,
            "delta_wz": 5,
        }

        if self.config.use_gripper:
            names["gripper"] = 6
            shape = (7,)
        else:
            shape = (6,)

        return {
            "dtype": "float32",
            "shape": shape,
            "names": names,
        }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        delta_wx = 0.0
        delta_wy = 0.0
        delta_wz = 0.0
        gripper_action = 1.0

        translation_unit = float(self.config.translation_unit)
        rotation_unit = float(self.config.rotation_unit)

        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if not val:
                continue

            if key == keyboard.Key.up:
                delta_y -= translation_unit
            elif key == keyboard.Key.down:
                delta_y += translation_unit
            elif key == keyboard.Key.left:
                delta_x += translation_unit
            elif key == keyboard.Key.right:
                delta_x -= translation_unit
            elif key == "z":
                delta_z -= translation_unit
            elif key == "x":
                delta_z += translation_unit
            elif key == "w":
                delta_wx += rotation_unit
            elif key == "s":
                delta_wx -= rotation_unit
            elif key == "a":
                delta_wy -= rotation_unit
            elif key == "d":
                delta_wy += rotation_unit
            elif key == "q":
                delta_wz += rotation_unit
            elif key == "e":
                delta_wz -= rotation_unit
            elif key == "c" and self.config.use_gripper:
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = 2.0
            elif key == "v" and self.config.use_gripper:
                gripper_action = 0.0
            else:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)
        # Clear any keys that were released in the meantime
        self.current_pressed = {key: val for key, val in self.current_pressed.items() if val}

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "delta_wx": delta_wx,
            "delta_wy": delta_wy,
            "delta_wz": delta_wz,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action
        else:
            action_dict["gripper"] = 0.0

        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the keyboard such as intervention status,
        episode termination, success indicators, etc.

        Keyboard mappings:
        - Any movement keys pressed = intervention active
        - 's' key = success (terminate episode successfully)
        - 'r' key = rerecord episode (terminate and rerecord)
        - 'q' key = quit episode (terminate without success)

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if not self.is_connected:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Check if any movement keys are currently pressed (indicates intervention)
        movement_keys = [
            keyboard.Key.up,
            keyboard.Key.down,
            keyboard.Key.left,
            keyboard.Key.right,
            "z",
            "x",
            "w",
            "s",
            "a",
            "d",
            "q",
            "e",
        ]
        if self.config.use_gripper:
            movement_keys.extend(["c", "v"])
        is_intervention = any(self.current_pressed.get(key, False) for key in movement_keys)

        # Check for episode control commands from misc_keys_queue
        terminate_episode = False
        success = False
        rerecord_episode = False

        # Process any pending misc keys
        while not self.misc_keys_queue.empty():
            key = self.misc_keys_queue.get_nowait()
            if key == "s":
                success = True
            elif key == "r":
                terminate_episode = True
                rerecord_episode = True
            elif key == "q":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }
