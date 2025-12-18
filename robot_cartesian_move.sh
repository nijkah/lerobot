python -m lerobot.scripts.lerobot_scripted_cartesian \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --zero_offset_json_path=so101_zero_offset.json \
  --waypoints_path=examples/waypoints/so101_cartesian_example2.json
  # --ee_bounds_min="[0.0, -0.25, 0.05]" \
  # --ee_bounds_max="[0.65, 0.30, 0.45]" \
  # --fps=30 \
  # --move_duration_s=3 \
  # --interpolation_mode=joint \

python -m lerobot.scripts.lerobot_scripted_cartesian \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf \
  --target_frame_name=gripper_frame_link \
  --ee_bounds_min="[0.25, -0.25, 0.05]" \
  --ee_bounds_max="[0.65, 0.25, 0.45]" \
  --fps=30 \
  --linear_speed_mps=0.06 \
  --angular_speed_radps=0.6 \
  --waypoints_path=examples/waypoints/so101_cartesian_example.csv
  # --waypoints_path=examples/waypoints/so101_cartesian_example.json
