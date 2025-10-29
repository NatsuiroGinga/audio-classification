#!/bin/bash 

echo 'speaker-identification-with-vad-non-streaming-asr application starting...'

source ./activate-cuda-11.8.sh

echo "CUDA_HOME=${CUDA_HOME}"

python3 ./version.py

if [ ! -d "../test" ]; then
  mkdir ../test
fi

python3 ./speaker-identification-with-vad-non-streaming-asr.py \
  --silero-vad-model ../models/vad/silero_vad.onnx \
  --speaker-file ../dataset/train-speaker.txt \
  --test-list ../dataset/test-speaker.txt \
  --model ../models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
  --sense-voice ../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --provider cuda \
  --tokens ../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --num-threads 2 \
  --out-dir ../test \
  --language zh
