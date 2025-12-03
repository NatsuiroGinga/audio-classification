#!/bin/bash
echo '[streaming_overlap_3src] Streaming OSD + Separation + SpeakerRecognition + ASR'

# 设置环境
if [ -f ../activate-cuda-11.8.sh ]; then
  source ../activate-cuda-11.8.sh
fi

echo "CUDA_HOME=${CUDA_HOME}"

# 必需参数
TARGET_WAV=${TARGET_WAV:-"/home/seed/Desktop/target_16k.wav"}
SPK_EMBED_MODEL=${SPK_EMBED_MODEL:-"../../models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}

# ASR模型
ASR_MODEL=${ASR_MODEL:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx"}
TOKENS=${TOKENS:-"../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt"}

# 输出目录
OUTPUT_DIR=${OUTPUT_DIR:-"../../test/streaming_results"}

python3 ./streaming_overlap_3src.py \
  --target-wav "${TARGET_WAV}" \
  --spk-embed-model "${SPK_EMBED_MODEL}" \
  --sv-threshold 0.4 \
  --sense-voice "${ASR_MODEL}" \
  --tokens "${TOKENS}" \
  --provider cuda \
  --num-threads 2 \
  --language auto \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --min-overlap-dur 0.2 \
  --sample-rate 16000 \
  --chunk-size 1024 \
  --process-seconds 5.0 \
  --output-dir "${OUTPUT_DIR}" \
  --save-interval 60.0

echo "Streaming processing completed. Results saved to ${OUTPUT_DIR}"
