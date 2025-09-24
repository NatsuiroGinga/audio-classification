#!/bin/bash
echo '[evaluate_with_sources] OSD + Separation Evaluation (Libri2Mix 8k sources)'

# Optional: source CUDA env
if [ -f ../activate-cuda-11.8.sh ]; then
  source ../activate-cuda-11.8.sh || true
fi

echo "CUDA_HOME=${CUDA_HOME:-}"
python3 ../version.py || true

echo "HF_TOKEN=${HF_TOKEN:-}"
echo 'Backends (pyannote.audio & asteroid) are REQUIRED.'

echo '==== Environment Overrides (export before running to change) ===='
# Core evaluation params (with defaults)
MAX_FILES=${MAX_FILES:-30}
ACTIVITY_THR=${ACTIVITY_THR:-0.03}
MIN_OVERLAP_DUR=${MIN_OVERLAP_DUR:-0.4}
OSD_THR=${OSD_THR:-0.5}
OSD_WIN=${OSD_WIN:-0.5}
OSD_HOP=${OSD_HOP:-0.1}
PROVIDER=${PROVIDER:-cpu}
SEP_CHECKPOINT=${SEP_CHECKPOINT:-}
SAVE_DETAILS=${SAVE_DETAILS:-1}   # 1=enable overlap_details.csv
OUT_BASE=${OUT_BASE:-../../test_overlap_eval}

# Optional ASR evaluation (pseudo-reference) parameters
ENABLE_ASR=${ENABLE_ASR:-1}          # 1 to enable --enable-asr
ASR_MODEL_TYPE=${ASR_MODEL_TYPE:-sense-voice}  # sense-voice | paraformer
SENSE_VOICE_MODEL=${SENSE_VOICE_MODEL:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"}
PARAFORMER_MODEL=${PARAFORMER_MODEL:-""}
TOKENS=${TOKENS:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"}
DECODING_METHOD=${DECODING_METHOD:-greedy_search}
ASR_NUM_THREADS=${ASR_NUM_THREADS:-2}
ASR_LANGUAGE=${ASR_LANGUAGE:-auto}

mkdir -p "${OUT_BASE}" ../../cache

echo "MAX_FILES=${MAX_FILES}"
echo "ACTIVITY_THR=${ACTIVITY_THR}"
echo "MIN_OVERLAP_DUR=${MIN_OVERLAP_DUR}"
echo "OSD_THR=${OSD_THR} OSD_WIN=${OSD_WIN} OSD_HOP=${OSD_HOP}"
echo "PROVIDER=${PROVIDER}"
[ -n "${SEP_CHECKPOINT}" ] && echo "SEP_CHECKPOINT=${SEP_CHECKPOINT}" || echo 'SEP_CHECKPOINT=<auto-download/default>'
[ "${SAVE_DETAILS}" = "1" ] && echo 'SAVE_DETAILS=on' || echo 'SAVE_DETAILS=off'
if [ "${ENABLE_ASR}" = "1" ]; then
  echo 'ASR Evaluation: ENABLED'
  echo "ASR_MODEL_TYPE=${ASR_MODEL_TYPE}"
  echo "DECODING_METHOD=${DECODING_METHOD} NUM_THREADS=${ASR_NUM_THREADS} LANGUAGE=${ASR_LANGUAGE}"
else
  echo 'ASR Evaluation: disabled (export ENABLE_ASR=1 to enable)'
fi

echo '==== Run Evaluation ===='
CMD=(python3 ./evaluate_with_sources.py \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --osd-thr "${OSD_THR}" \
  --osd-win "${OSD_WIN}" \
  --osd-hop "${OSD_HOP}" \
  --activity-thr "${ACTIVITY_THR}" \
  --min-overlap-dur "${MIN_OVERLAP_DUR}" \
  --max-files "${MAX_FILES}" \
  --provider "${PROVIDER}" \
  --out-dir "${OUT_BASE}" )

if [ "${ENABLE_ASR}" = "1" ]; then
  CMD+=(--enable-asr --tokens "${TOKENS}" --decoding-method "${DECODING_METHOD}" --num-threads "${ASR_NUM_THREADS}" --language "${ASR_LANGUAGE}")
  case "${ASR_MODEL_TYPE}" in
    sense-voice)
      CMD+=(--sense-voice "${SENSE_VOICE_MODEL}")
      ;;
    paraformer)
      if [ -z "${PARAFORMER_MODEL}" ]; then
        echo "[WARN] ENABLE_ASR=1 but PARAFORMER_MODEL empty; falling back to sense-voice." >&2
        CMD+=(--sense-voice "${SENSE_VOICE_MODEL}")
      else
        CMD+=(--paraformer "${PARAFORMER_MODEL}")
      fi
      ;;
    *)
      echo "[WARN] Unknown ASR_MODEL_TYPE=${ASR_MODEL_TYPE}; using sense-voice." >&2
      CMD+=(--sense-voice "${SENSE_VOICE_MODEL}")
      ;;
  esac
fi

if [ -n "${SEP_CHECKPOINT}" ]; then
  CMD+=(--sep-checkpoint "${SEP_CHECKPOINT}")
fi
if [ "${SAVE_DETAILS}" = "1" ]; then
  CMD+=(--save-details)
fi

printf 'Command: %q ' "${CMD[@]}"; echo
"${CMD[@]}"

echo '[evaluate_with_sources] Done. See test_overlap_eval/<timestamp>/evaluation.json'
