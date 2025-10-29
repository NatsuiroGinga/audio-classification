#!/bin/bash
set -euo pipefail

echo '[install] Install Python dependencies and download default models'
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

cd "${SCRIPT_DIR}" || exit 1

# -----------------------------------------------------------------------------
# 1) Python dependencies
# -----------------------------------------------------------------------------
# By default we install CUDA-enabled wheels pinned in requirements.txt.
# If you want CPU-only, set CPU=1 before running this script.

CPU_MODE=${CPU:-0}

python3 -m pip install --upgrade pip

if [[ "${CPU_MODE}" == "1" ]]; then
  echo '[install] CPU mode: installing CPU wheels'
  # Minimal CPU set (override GPU-pinned ones). Adjust versions if needed.
  python3 -m pip install --no-cache-dir \
    torch==2.5.0 \
    torchaudio==2.5.0 \
    onnxruntime==1.20.0 \
    numpy==2.3.4 \
    soundfile sounddevice tqdm \
    huggingface_hub==0.34.4 \
    psutil==5.9.0 \
    matplotlib==3.10.7 \
    asteroid==0.7.0 \
    pyannote_audio==3.4.0 \
    pyannote.core==5.0.0 pyannote.database==5.1.3 pyannote.metrics==3.2.1 pyannote.pipeline==3.0.1
  # sherpa-onnx CPU
  python3 -m pip install --no-cache-dir sherpa-onnx==1.11.1 -f https://k2-fsa.github.io/sherpa/onnx/cpu.html
else
  echo '[install] CUDA mode: installing pinned GPU wheels from requirements.txt'
  # sherpa-onnx CUDA wheels index is needed for the "+cuda" build
  python3 -m pip install --no-cache-dir -r "${ROOT_DIR}/requirements.txt" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
fi

echo '[install] Python dependencies done.'

# -----------------------------------------------------------------------------
# 2) Download default models (speaker embedding + ASR)
# -----------------------------------------------------------------------------
mkdir -p "${ROOT_DIR}/models/asr" "${ROOT_DIR}/models/speaker-recognition"

echo '[install] Downloading 3dspeaker (speaker embedding ONNX) ...'
wget -q --show-progress -O "${ROOT_DIR}/models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

echo '[install] Downloading sherpa-onnx SenseVoice (ASR) ...'
ASR_TARBALL="${ROOT_DIR}/models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
wget -q --show-progress -O "${ASR_TARBALL}" \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf "${ASR_TARBALL}" -C "${ROOT_DIR}/models/asr"
rm -f "${ASR_TARBALL}"

echo '[install] Models ready.'

# -----------------------------------------------------------------------------
# 3) Notes for datasets
# -----------------------------------------------------------------------------
echo '[install] Dataset setup:'
echo '  - 三路分离脚本依赖 LibriMix/Libri3Mix，请自行准备并设置环境变量 LIBRIMIX_ROOT 指向其上级目录。'
echo '  - 参考：https://github.com/JorisCos/LibriMix'
echo '  - 注意：因 LibriMix 数据集过大，可以选择不同子集下载以减少不必要的存储占用。'
echo '  - 示例：export LIBRIMIX_ROOT=/abs/path/to/LibriMix'

echo '[install] Done.'
echo 'Models tree:'
ls -R "${ROOT_DIR}/models"
