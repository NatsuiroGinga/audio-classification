#!/bin/bash
echo '[offline_overlap_mvp] OSD + Separation + ASR (Libri2Mix 8k)'

# Optional: source CUDA env (comment out if already set)
if [ -f ../activate-cuda-11.8.sh ]; then
  source ../activate-cuda-11.8.sh
fi

echo "CUDA_HOME=${CUDA_HOME}"
python3 ../version.py

BASE_OUT=../../test/overlap
mkdir -p "${BASE_OUT}" ../../cache

echo "HF_TOKEN=${HF_TOKEN}"
echo 'Backends (pyannote.audio & asteroid) are REQUIRED.'

# ASR model (sense-voice example). You can switch to paraformer by replacing below two lines.
ASR_MODEL=${ASR_MODEL:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"}
TOKENS=${TOKENS:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"}

# Limit processed mixtures: export MAX_FILES=50 (0 = all)
MAX_FILES=${MAX_FILES:-10}

python3 ./offline_overlap_mvp.py \
  --sense-voice "${ASR_MODEL}" \
  --tokens "${TOKENS}" \
  --provider cuda \
  --num-threads 2 \
  --language zh \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --min-overlap-dur 0.4 \
  --max-files "${MAX_FILES}" \
  --out-dir "${BASE_OUT}"

echo "[offline_overlap_mvp] Done. See ${BASE_OUT}/<timestamp>/ for outputs (segments + summary)."
