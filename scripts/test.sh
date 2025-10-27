#!/bin/bash
echo 'speaker-identification-with-vad-non-streaming-asr testing...'

source ./activate-cuda-11.8.sh

echo "CUDA_HOME=${CUDA_HOME}"

python3 ./version.py

if [ ! -d "../test" ]; then
  mkdir ../test
fi

if [ ! -d "../cache" ]; then
    mkdir ../cache
fi

python3 ./benchmark_pipeline.py \
  --silero-vad-model ../models/vad/silero_vad.onnx \
  --speaker-file ../dataset/train-speaker.txt \
  --test-list ../dataset/test-speaker.txt \
  --model ../models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
  --sense-voice ../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --provider cuda \
  --tokens ../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --num-threads 2 \
  --out-dir ../test \
  --language zh \
  --ref-text-list ../dataset/transcription/test_transcription \
  --emb-cache-dir ../cache \
  --cpu-normalize \
  --load-speaker-embeds ../cache/speakers.npz \
  --plot-cpu \
  # 初次运行时, 注释掉--load-speaker-embeds, 启用--save-speaker-embeds, 先生成说话人embeddings缓存
  # 后续使用相同数据集时, 注释掉--save-speaker-embeds, 启用--load-speaker-embeds, 直接加载缓存
  #--save-speaker-embeds ../cache/speakers.npz
