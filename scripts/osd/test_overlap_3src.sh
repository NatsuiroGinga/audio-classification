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
LIBRIMIX_ROOT=${LIBRIMIX_ROOT:-"../../../../dataset/LibriMix"}

# Speaker verification threshold
SV_THRESHOLD=${SV_THRESHOLD:-0.6}

# ASR model (sense-voice example). You can switch to paraformer by replacing below two lines.
ASR_MODEL=${ASR_MODEL:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"}
TOKENS=${TOKENS:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"}

# Speaker embedding model
SPK_EMBED_MODEL=${SPK_EMBED_MODEL:-"../../models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}

# Limit processed mixtures: export MAX_FILES=50 (0 = all)
MAX_FILES=${MAX_FILES:-100}

# File mode: provide INPUT_WAVS (space-separated list) and TARGET_WAV
INPUT_WAVS=${INPUT_WAVS:-"/data/workspace/llm/audio-classification/dataset/cn/s2/mix.wav"}
TARGET_WAV=${TARGET_WAV:-"/data/workspace/llm/audio-classification/dataset/cn/s2/D21_861.wav"}
# Optional references for file-mode separation evaluation
REFS_CSV=${REFS_CSV:-}
REF_WAVS=${REF_WAVS:-"/data/workspace/llm/audio-classification/dataset/cn/s2/D21_861.wav 
/data/workspace/llm/audio-classification/dataset/cn/s2/D31_772.wav 
/data/workspace/llm/audio-classification/dataset/cn/s2/D32_776.wav"}

EVAL_SEP=${EVAL_SEP:-1}
SAVE_SEP_DETAILS=${SAVE_SEP_DETAILS:-1}

EXTRA_EVAL_ARGS=()
if [[ -n "${EVAL_SEP}" && "${EVAL_SEP}" != "0" ]]; then
  EXTRA_EVAL_ARGS+=("--eval-separation" "--enable-metrics")
fi
if [[ -n "${SAVE_SEP_DETAILS}" && "${SAVE_SEP_DETAILS}" != "0" ]]; then
  EXTRA_EVAL_ARGS+=("--save-sep-details")
fi

# Reproducibility: forward --seed if SEED is provided
if [[ -n "${SEED}" ]]; then
  EXTRA_EVAL_ARGS+=("--seed" "${SEED}")
fi

# Build mode-specific args
MODE_ARGS=()
if [[ -n "${INPUT_WAVS}" ]]; then
  echo "[offline_overlap_3src] File mode: using --input-wavs and --target-wav"
  # Split INPUT_WAVS into array
  read -r -a INPUT_ARR <<< "${INPUT_WAVS}"
  if [[ -z "${TARGET_WAV}" ]]; then
    echo "[offline_overlap_3src] TARGET_WAV is required when INPUT_WAVS is provided" >&2
    exit 2
  fi
  MODE_ARGS+=("--input-wavs")
  for f in "${INPUT_ARR[@]}"; do MODE_ARGS+=("${f}"); done
  MODE_ARGS+=("--target-wav" "${TARGET_WAV}")
  if [[ -n "${REFS_CSV}" ]]; then
    MODE_ARGS+=("--refs-csv" "${REFS_CSV}")
  fi
  if [[ -n "${REF_WAVS}" ]]; then
    read -r -a REF_ARR <<< "${REF_WAVS}"
    MODE_ARGS+=("--ref-wavs")
    for r in "${REF_ARR[@]}"; do MODE_ARGS+=("${r}"); done
  fi
fi

python3 ./offline_overlap_3src.py \
  --librimix-root "${LIBRIMIX_ROOT}" \
  --spk-embed-model "${SPK_EMBED_MODEL}" \
  --sv-threshold "${SV_THRESHOLD}" \
  --sense-voice "${ASR_MODEL}" \
  --tokens "${TOKENS}" \
  --provider cuda \
  --num-threads 2 \
  --language auto \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --min-overlap-dur 0.2 \
  --max-files "${MAX_FILES}" \
  --out-dir "${BASE_OUT}" \
  "${MODE_ARGS[@]}" \
  "${EXTRA_EVAL_ARGS[@]}"

echo "[offline_overlap_3src] Done. See ${BASE_OUT}/<timestamp>/ for outputs (segments + summary)."
