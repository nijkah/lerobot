python -m lerobot.scripts.lerobot_calibrate_zero_offset \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.use_degrees=true \
  --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --output_json=so101_zero_offset.json
  # --target_frame_name='jaw' \
  # --urdf_path=SO-ARM100/Simulation/SO100/so100.urdf \
  # --output_json=so100_zero_offset.json
