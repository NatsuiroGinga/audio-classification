#!/bin/bash
# 流式演示脚本 (支持 VAD 分段模式)
# 使用 VAD 进行语音分段，保持语义完整性
#
# 用法: bash demo_streaming.sh [sample] [process_seconds] [sv_threshold] [--vad] [--debug]
# 示例: 
#   bash demo_streaming.sh s1              # 使用s1样本，固定间隔模式
#   bash demo_streaming.sh s1 2.0 0.4 --vad  # 使用 VAD 分段模式
#   bash demo_streaming.sh s1 3.0 0.3 --debug  # 启用调试模式

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# 样本选择 (s1-s10)
SAMPLE=${1:-s1}
# 流式处理间隔秒数（仅固定间隔模式使用）
PROCESS_SECONDS=${2:-3.0}
# SV 阈值 (0-1，越低越容易匹配)
SV_THRESHOLD=${3:-0.4}

# 检查额外参数
USE_VAD=""
DEBUG_MODE=""
for arg in "$@"; do
    if [ "$arg" == "--vad" ]; then
        USE_VAD="yes"
    fi
    if [ "$arg" == "--debug" ]; then
        DEBUG_MODE="--debug"
    fi
done

PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# 获取目标说话人文件（第一个非mix.wav的wav文件）
TARGET_WAV=$(ls "${PROJECT_DIR}/dataset/cn/${SAMPLE}/"*.wav | grep -v mix.wav | head -1)

echo "=================================================="
echo "  🎤 流式声纹识别演示"
if [ -n "$USE_VAD" ]; then
    echo "  📊 VAD 分段模式 - 基于语音边界分段"
else
    echo "  📊 固定间隔模式 - ${PROCESS_SECONDS}秒间隔"
fi
echo "  样本: $SAMPLE"
echo "  SV阈值: ${SV_THRESHOLD}"
echo "  目标: $(basename "$TARGET_WAV")"
echo "=================================================="
echo ""

# 构建 VAD 参数
VAD_ARGS=""
if [ -n "$USE_VAD" ]; then
    VAD_ARGS="--vad-model ${PROJECT_DIR}/models/vad/silero_vad.onnx"
fi

python demo_streaming.py \
    --mix-wav "${PROJECT_DIR}/dataset/cn/${SAMPLE}/mix.wav" \
    --target-wav "${TARGET_WAV}" \
    --spk-embed-model "${PROJECT_DIR}/models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx" \
    --sense-voice "${PROJECT_DIR}/models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx" \
    --tokens "${PROJECT_DIR}/models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt" \
    --provider cuda \
    --sv-threshold "$SV_THRESHOLD" \
    --chunk-size 1024 \
    --process-seconds "$PROCESS_SECONDS" \
    $VAD_ARGS \
    $DEBUG_MODE 2>&1 | grep -v "Warning\|FutureWarning\|UserWarning\|pytorch_lightning\|TensorFloat\|weights_only\|Model was trained\|Lightning automatically\|Using HF token"
