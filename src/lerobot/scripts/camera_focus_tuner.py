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

"""Interactive helper to dial in a manual focus value for UVC cameras.

Usage:
    python -m lerobot.scripts.camera_focus_tuner \
        --camera-index=0 \
        --width=1280 --height=720 \
        --focus-min=0 --focus-max=255 --initial-focus=140

Controls:
    - Trackbar: drag to set absolute focus value.
    - '[' / ']': nudge focus by --focus-step (default 5).
    - 'a': toggle autofocus on/off.
    - 's': print the current focus value to the terminal.
    - 'q' or ESC: quit.

Notes:
    Focus ranges vary per camera/driver. If values 0-255 do nothing, try 0-1023.
    Some cameras need CAP_V4L2 backend on Linux; pass --backend=v4l2 to enforce it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Literal

import cv2


Backend = Literal["any", "v4l2", "dshow", "msmf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive manual focus tuner for cameras.")
    parser.add_argument("--camera-index", type=str, default="0", help="Camera index or path (default: %(default)s)")
    parser.add_argument("--width", type=int, default=1280, help="Preview width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=720, help="Preview height (default: %(default)s)")
    parser.add_argument("--backend", type=str, choices=["any", "v4l2", "dshow", "msmf"], default="any")
    parser.add_argument("--focus-min", type=int, default=0, help="Minimum focus value supported by the driver.")
    parser.add_argument("--focus-max", type=int, default=255, help="Maximum focus value supported by the driver.")
    parser.add_argument(
        "--initial-focus",
        type=int,
        default=None,
        help="Initial focus value. Defaults to current driver value or midpoint.",
    )
    parser.add_argument("--focus-step", type=int, default=5, help="Increment used by '[' and ']' hotkeys.")
    parser.add_argument("--disable-autofocus", action="store_true", help="Force manual focus on startup.")
    parser.add_argument("--window", type=str, default="Focus Tuner", help="Name of the preview window.")
    args = parser.parse_args()
    if args.focus_max <= args.focus_min:
        parser.error("--focus-max must be greater than --focus-min.")
    return args


def _resolve_backend(flag: Backend) -> int:
    if flag == "v4l2":
        return cv2.CAP_V4L2
    if flag == "dshow":
        return cv2.CAP_DSHOW
    if flag == "msmf":
        return cv2.CAP_MSMF
    return 0


def _set_focus(cap: cv2.VideoCapture, value: int, focus_min: int, focus_max: int) -> int:
    clamped = max(focus_min, min(focus_max, value))
    cap.set(cv2.CAP_PROP_FOCUS, clamped)
    return clamped


def _set_autofocus(cap: cv2.VideoCapture, enabled: bool) -> bool:
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if enabled else 0)
    return enabled


def _get_focus_value(cap: cv2.VideoCapture) -> int:
    val = cap.get(cv2.CAP_PROP_FOCUS)
    return int(round(val)) if val >= 0 else -1


def _get_autofocus_state(cap: cv2.VideoCapture) -> bool:
    return cap.get(cv2.CAP_PROP_AUTOFOCUS) > 0.5


def main() -> int:
    args = parse_args()

    backend = _resolve_backend(args.backend)
    cam_index = int(args.camera_index) if args.camera_index.isdigit() else args.camera_index
    cap = cv2.VideoCapture(cam_index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera_index}", file=sys.stderr)
        return 1

    if args.disable_autofocus:
        _set_autofocus(cap, False)

    focus_value = (
        args.initial_focus
        if args.initial_focus is not None
        else (_get_focus_value(cap) if _get_focus_value(cap) >= 0 else (args.focus_min + args.focus_max) // 2)
    )
    focus_value = _set_focus(cap, focus_value, args.focus_min, args.focus_max)

    window = args.window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)

    trackbar_range = args.focus_max - args.focus_min

    def _on_trackbar(pos: int) -> None:
        nonlocal focus_value
        focus_value = _set_focus(cap, args.focus_min + pos, args.focus_min, args.focus_max)

    cv2.createTrackbar("Focus", window, focus_value - args.focus_min, trackbar_range, _on_trackbar)

    print("Controls: '[' / ']' adjust focus, 'a' toggle autofocus, 's' prints value, 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame; check camera connection.", file=sys.stderr)
                break

            focus_value = _get_focus_value(cap)
            auto_enabled = _get_autofocus_state(cap)
            status = f"focus={focus_value}  autofocus={'ON' if auto_enabled else 'OFF'}"
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("a"):
                _set_autofocus(cap, not auto_enabled)
            if key == ord("["):
                focus_value = _set_focus(cap, focus_value - args.focus_step, args.focus_min, args.focus_max)
                cv2.setTrackbarPos("Focus", window, focus_value - args.focus_min)
            if key == ord("]"):
                focus_value = _set_focus(cap, focus_value + args.focus_step, args.focus_min, args.focus_max)
                cv2.setTrackbarPos("Focus", window, focus_value - args.focus_min)
            if key == ord("s"):
                print(f"Current focus value: {focus_value}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Final focus value: {focus_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
