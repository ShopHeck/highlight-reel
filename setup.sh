#!/usr/bin/env bash
# highlight-reel-production: idempotent setup. Builds the venv (default: <skill>/.venv, override HR_VENV),
# apt-installs missing system libs (libegl1 etc. for mediapipe, ffmpeg, fonts), downloads the mediapipe models.
set -euo pipefail
SK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; VENV="${HR_VENV:-$SK/.venv}"
SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo -n"
need=(); for p in libegl1 libgl1 libgles2 ffmpeg fonts-dejavu-core python3-venv; do dpkg -s "$p" >/dev/null 2>&1 || need+=("$p"); done
if [ ${#need[@]} -gt 0 ]; then echo "[setup] apt-get install ${need[*]}"; $SUDO apt-get update -qq; DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${need[@]}"; else echo "[setup] system packages present"; fi
[ -x "$VENV/bin/python" ] || { echo "[setup] creating venv $VENV"; python3 -m venv "$VENV"; }
"$VENV/bin/python" -m pip install -q --no-cache-dir --upgrade pip
"$VENV/bin/python" -m pip install -q --no-cache-dir -r "$SK/requirements.txt"
mkdir -p "$SK/models"; MP=https://storage.googleapis.com/mediapipe-models
[ -s "$SK/models/pose_lite.task" ]    || curl -fsSL -o "$SK/models/pose_lite.task"    $MP/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
[ -s "$SK/models/blaze_face.tflite" ] || curl -fsSL -o "$SK/models/blaze_face.tflite" $MP/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite
echo "$VENV" > "$SK/.venv_path"
"$VENV/bin/python" -c "import cv2,mediapipe,librosa,soundfile,numpy,scipy;from mediapipe.tasks.python import vision;print('[setup] OK cv2',cv2.__version__,'mediapipe',mediapipe.__version__,'librosa',librosa.__version__)"
