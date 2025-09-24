#!/bin/bash
echo 'offline_overlap_mvp testing (OSD+Separation MVP)...'

source ../activate-cuda-11.8.sh

echo "CUDA_HOME=${CUDA_HOME}"

python3 ../version.py

if [ ! -d "../../test_overlap" ]; then
  mkdir ../../test_overlap
fi

if [ ! -d "../../cache" ]; then
  mkdir -p ../../cache
fi

echo "HF_TOKEN=${HF_TOKEN}"

echo 'OSD/Separation backends are REQUIRED (pyannote.audio, asteroid).'

python3 ./offline_overlap_mvp.py \
  --speaker-file ../../dataset/scene-2-speaker.txt \
  --test-list ../../dataset/scene-2-speaker.txt \
  --model ../../models/speaker-recongition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx \
  --sense-voice ../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx \
  --tokens ../../models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt \
  --provider cuda \
  --num-threads 2 \
  --language zh \
  --threshold 0.5 \
  --ref-text-list ../../dataset/transcription/scene-2_transcription \
  --osd-backend pyannote \
  --sep-backend asteroid \
  --min-overlap-dur 0.4 \
  --out-dir ../../test_overlap

echo 'Done running offline_overlap_mvp. See ../../test_overlap for outputs.'
