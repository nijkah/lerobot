#!/usr/bin/env bash

# Helper script to run the Grounded SAM 2-assisted grasp visualizer.
# Update the calibration, robot, and SAM 2 parameters for your setup.

set -euo pipefail

# Camera + calibration assets.
CALIB_JSON="calibration/hand_eye_top_camera.json"
INTRINSICS="calibration/camera_calibration.json"
CAMERA_SOURCE="0"
AXIS_LENGTH="0.2"
WIDTH="1280"
HEIGHT="720"
FPS="30"

# Detection + grasp planning.
TARGET_OBJECT="pen"
TARGET_PLANE_Z="0.02"
APPROACH_OFFSET="0.08"
GRASP_OFFSET="-0.01"
RETREAT_OFFSET="0.12"
POSITION_WEIGHT="1.0"
ORIENTATION_WEIGHT="0.05"
JOINT_STEP_DEG="2.0"
MOTION_DT="0.02"
PREGRASP_HOLD_S="0.5"
GRASP_HOLD_S="0.6"
RETREAT_HOLD_S="0.5"
GRIPPER_OPEN_POS="30.0"
GRIPPER_CLOSE_POS="5.0"
GRIPPER_CHANNEL="gripper"
REOPEN_AFTER_RETREAT="false"
SEGMENTATION_ALPHA="0.35"
SEGMENTATION_COLOR="[0, 165, 255]"
REQUIRE_MASK_FOR_OVERLAY="false"
HIDE_GRASP_MARKER="false"

# Grounded SAM 2 options. Leave host empty to run locally.
GROUNDING_DEVICE="cuda"
SAM2_MODEL_CONFIG="Grounded-SAM-2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT="Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_GROUNDING_MODEL="IDEA-Research/grounding-dino-tiny"
SAM2_HF_REVISION=""
SAM2_HF_CACHE_DIR=""
GROUNDING_SERVER_HOST="192.168.2.249:9811"
GROUNDING_SERVER_SCHEME="http"
GROUNDING_SERVER_TIMEOUT="10.0"
BOX_THRESHOLD="0.35"
TEXT_THRESHOLD="0.25"
MIN_BOX_AREA="1500"

# Robot configuration (matches the scripted Cartesian example).
ROBOT_TYPE="so101_follower"
ROBOT_PORT="/dev/ttyACM1"
ROBOT_ID="follower_arm"
URDF_PATH="SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
TARGET_FRAME_NAME="gripper_frame_link"
ZERO_OFFSET_JSON="so101_zero_offset.json"

SERVER_URL="${GROUNDING_SERVER_HOST}"
if [[ -n "${SERVER_URL}" && "${SERVER_URL}" != http://* && "${SERVER_URL}" != https://* ]]; then
  SERVER_URL="${GROUNDING_SERVER_SCHEME}://${SERVER_URL}"
fi

SAM2_ARGS=()
if [[ -n "${SERVER_URL}" ]]; then
  SAM2_ARGS+=("--grounding-server-url" "${SERVER_URL}")
  SAM2_ARGS+=("--grounding-server-timeout" "${GROUNDING_SERVER_TIMEOUT}")
else
  if [[ ! -f "${SAM2_MODEL_CONFIG}" ]]; then
    echo "SAM2 config not found: ${SAM2_MODEL_CONFIG}" >&2
    exit 1
  fi
  if [[ ! -f "${SAM2_CHECKPOINT}" ]]; then
    echo "SAM2 checkpoint not found: ${SAM2_CHECKPOINT}" >&2
    exit 1
  fi
  SAM2_ARGS+=("--use-grounded-sam2")
  SAM2_ARGS+=("--grounding-device" "${GROUNDING_DEVICE}")
  SAM2_ARGS+=("--sam2-model-config" "${SAM2_MODEL_CONFIG}")
  SAM2_ARGS+=("--sam2-checkpoint" "${SAM2_CHECKPOINT}")
  SAM2_ARGS+=("--sam2-grounding-model" "${SAM2_GROUNDING_MODEL}")
  if [[ -n "${SAM2_HF_REVISION}" ]]; then
    SAM2_ARGS+=("--sam2-hf-revision" "${SAM2_HF_REVISION}")
  fi
  if [[ -n "${SAM2_HF_CACHE_DIR}" ]]; then
    SAM2_ARGS+=("--sam2-hf-cache-dir" "${SAM2_HF_CACHE_DIR}")
  fi
fi

REOPEN_FLAG=()
if [[ "${REOPEN_AFTER_RETREAT}" == "true" ]]; then
  REOPEN_FLAG+=("--reopen-after-retreat")
fi

MASK_FLAG=()
if [[ "${REQUIRE_MASK_FOR_OVERLAY}" == "true" ]]; then
  MASK_FLAG+=("--require-mask-for-overlay")
fi

MARKER_FLAG=()
if [[ "${HIDE_GRASP_MARKER}" == "true" ]]; then
  MARKER_FLAG+=("--hide-grasp-marker")
fi

python -m lerobot.scripts.lerobot_visualize_base_frame_grasp \
  --calibration "${CALIB_JSON}" \
  --intrinsics "${INTRINSICS}" \
  --camera-source "${CAMERA_SOURCE}" \
  --axis-length "${AXIS_LENGTH}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --target-object "${TARGET_OBJECT}" \
  --box-threshold "${BOX_THRESHOLD}" \
  --text-threshold "${TEXT_THRESHOLD}" \
  --min-box-area "${MIN_BOX_AREA}" \
  --target-plane-z "${TARGET_PLANE_Z}" \
  --approach-offset "${APPROACH_OFFSET}" \
  --grasp-offset "${GRASP_OFFSET}" \
  --retreat-offset "${RETREAT_OFFSET}" \
  --position-weight "${POSITION_WEIGHT}" \
  --orientation-weight "${ORIENTATION_WEIGHT}" \
  --joint-step-deg "${JOINT_STEP_DEG}" \
  --motion-dt "${MOTION_DT}" \
  --pregrasp-hold-s "${PREGRASP_HOLD_S}" \
  --grasp-hold-s "${GRASP_HOLD_S}" \
  --retreat-hold-s "${RETREAT_HOLD_S}" \
  --gripper-open-pos "${GRIPPER_OPEN_POS}" \
  --gripper-close-pos "${GRIPPER_CLOSE_POS}" \
  --gripper-channel "${GRIPPER_CHANNEL}" \
  --segmentation-alpha "${SEGMENTATION_ALPHA}" \
  --segmentation-color "${SEGMENTATION_COLOR}" \
  --robot.type "${ROBOT_TYPE}" \
  --robot.port "${ROBOT_PORT}" \
  --robot.id "${ROBOT_ID}" \
  --robot.use_degrees=true \
  --camera.focus_value=312 \
  --urdf-path "${URDF_PATH}" \
  --target-frame-name "${TARGET_FRAME_NAME}" \
  --zero-offset-json-path "${ZERO_OFFSET_JSON}" \
  "${SAM2_ARGS[@]}" \
  "${REOPEN_FLAG[@]}" \
  "${MASK_FLAG[@]}" \
  "${MARKER_FLAG[@]}"
