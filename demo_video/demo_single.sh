#!/bin/bash
# 单样本实时演示脚本
# 用法: bash demo_single.sh [样本名称]
# 默认: s1

SAMPLE=${1:-s1}

cd /data/workspace/llm/audio-classification

# 获取目标说话人文件（第一个非mix.wav的wav文件）
TARGET_WAV=$(ls dataset/cn/${SAMPLE}/*.wav | grep -v mix.wav | head -1)

python3 demo_video/demo_single.py \
  --mix-wav dataset/cn/${SAMPLE}/mix.wav \
  --target-wav ${TARGET_WAV} \
  --spk-embed-model models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
  --sense-voice models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --tokens models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --provider cuda \
  --sv-threshold 0.4 2>&1 | grep -v "Warning\|FutureWarning\|UserWarning\|pytorch_lightning\|TensorFloat\|weights_only\|Model was trained"
