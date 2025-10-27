#!/bin/bash
echo '[offline_overlap_3src] OSD + Separation + SpeakerRecognition + ASR (LibriMix / Libri3Mix, 3-src)'

# Optional: source CUDA env (comment out if already set)
if [ -f ../activate-cuda-11.8.sh ]; then
  source ../activate-cuda-11.8.sh
fi

echo "CUDA_HOME=${CUDA_HOME}"
python3 ../version.py

BASE_OUT=../../test/overlap3
mkdir -p "${BASE_OUT}" ../../cache

echo "HF_TOKEN=${HF_TOKEN}"
echo 'Backends (pyannote.audio & asteroid) are REQUIRED.'

# Required dataset root for LibriMix/Libri3Mix
LIBRIMIX_ROOT=${LIBRIMIX_ROOT:-"../../dataset/LibriMix"}

# Speaker verification threshold
SV_THRESHOLD=${SV_THRESHOLD:-0.6}

# ASR model (sense-voice example). You can switch to paraformer by replacing below two lines.
ASR_MODEL=${ASR_MODEL:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"}
TOKENS=${TOKENS:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"}

# Speaker embedding model
SPK_EMBED_MODEL=${SPK_EMBED_MODEL:-"../../models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}

# Enrollment wavs
ENROLL_WAVS=${ENROLL_WAVS:-"/path/to/target1.wav,/path/to/target2.wav"}

# Limit processed mixtures: export MAX_FILES=50 (0 = all)
MAX_FILES=${MAX_FILES:-10}

python3 ./offline_overlap_3src.py \
  --librimix-root "${LIBRIMIX_ROOT}" \
  --spk-embed-model "${SPK_EMBED_MODEL}" \
  --enroll-wavs "${ENROLL_WAVS}" \
  --sv-threshold "${SV_THRESHOLD}" \
  --sense-voice "${ASR_MODEL}" \
  --tokens "${TOKENS}" \
  --provider cuda \
  --num-threads 2 \
  --language auto \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --min-overlap-dur 0.4 \
  --max-files "${MAX_FILES}" \
  --out-dir "${BASE_OUT}"

echo "[offline_overlap_3src] Done. See ${BASE_OUT}/<timestamp>/ for outputs (segments + summary)."
