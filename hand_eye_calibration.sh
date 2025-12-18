#!/usr/bin/env bash

# Example usage of the eye-to-hand calibration helper.
# Adjust the robot port, camera index, intrinsics file, and AprilTag size/id
# for your setup before running.

set -euo pipefail

python -m lerobot.scripts.lerobot_hand_eye_calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.use_degrees=true \
  --camera.index_or_path=0 \
  --camera.width=1280 \
  --camera.height=720 \
  --camera.show_preview=true \
  --camera.focus_value=312 \
  --intrinsics_path=calibration/camera_calibration.json \
  --tag_family=tag36h11 \
  --tag_id=6 \
  --tag_size_m=0.05 \
  --required_samples=30 \
  --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --zero_offset_json_path=so101_zero_offset.json \
  --save_frames_dir=hand_eye_samples \
  --record_samples_path=hand_eye_samples/records \
  --output_path=calibration/hand_eye_top_camera.json 
  #--pybullet_visualization=true 

"""
python -m lerobot.scripts.lerobot_hand_eye_calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --camera.index_or_path=0 \
  --camera.width=1280 \
  --camera.height=720 \
  --camera.show_preview=true \
  --camera.focus_value=312 \
  --intrinsics_path=calibration/camera_calibration.json \
  --tag_family=tag36h11 \
  --tag_id=6 \
  --tag_size_m=0.05 \
  --required_samples=30 \
  --urdf_path=SO-ARM100/Simulation/SO100/so100.urdf \
  --target_frame_name=jaw \
  --zero_offset_json_path=so100_zero_offset.json \
  --save_frames_dir=captured_images/hand_eye_samples \
  --output_path=calibration/hand_eye_top_camera.json \
  --solver_method=tsai
"""
