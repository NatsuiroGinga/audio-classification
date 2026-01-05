#!/bin/bash
# 流式 ASR 演示脚本 - 支持 Partial/Final 输出
#
# 用法:
#   bash demo_streaming_asr.sh <sample_name> [chunk_duration] [partial_interval]
#
# 例子:
#   bash demo_streaming_asr.sh s1
#   bash demo_streaming_asr.sh s1 0.3 0.5
#   bash demo_streaming_asr.sh s1 0.3 0.5 --debug

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
SAMPLE=${1:-s1}
CHUNK_DURATION=${2:-0.3}
PARTIAL_INTERVAL=${3:-0.5}
MAX_SEGMENT=3.0
SV_THRESHOLD=0.5

# 从 $4 开始收集额外参数
shift 3 2>/dev/null || true
EXTRA_ARGS="$@"

# 模型路径
VAD_MODEL="$PROJECT_ROOT/models/vad/silero_vad.onnx"
SPK_MODEL="$PROJECT_ROOT/models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
SENSE_VOICE_DIR="$PROJECT_ROOT/models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSE_VOICE="$SENSE_VOICE_DIR/model.onnx"
TOKENS="$SENSE_VOICE_DIR/tokens.txt"

# 音频路径
SAMPLE_DIR="$PROJECT_ROOT/dataset/cn/${SAMPLE}"
if [ -d "$SAMPLE_DIR" ]; then
    INPUT_WAV="$SAMPLE_DIR/mix.wav"
    # 获取目标说话人文件（第一个非mix.wav的wav文件）
    TARGET_WAV=$(ls "${SAMPLE_DIR}/"*.wav | grep -v mix.wav | head -1)
else
    echo "Sample directory not found: $SAMPLE_DIR"
    exit 1
fi

# 检查文件
for f in "$VAD_MODEL" "$SPK_MODEL" "$SENSE_VOICE" "$INPUT_WAV" "$TARGET_WAV"; do
    if [ ! -e "$f" ]; then
        echo "Error: File not found: $f"
        exit 1
    fi
done

echo "========================================"
echo "Streaming ASR Demo (Partial/Final)"
echo "========================================"
echo "Sample:           $SAMPLE"
echo "Input:            $INPUT_WAV"
echo "Target:           $TARGET_WAV"
echo "Chunk duration:   ${CHUNK_DURATION}s"
echo "Partial interval: ${PARTIAL_INTERVAL}s"
echo "Max segment:      ${MAX_SEGMENT}s"
echo "SV threshold:     $SV_THRESHOLD"
echo "Extra args:       $EXTRA_ARGS"
echo "========================================"
echo ""

cd "$SCRIPT_DIR/osd"

python demo_streaming_asr.py \
    --input-wav "$INPUT_WAV" \
    --target-wav "$TARGET_WAV" \
    --vad-model "$VAD_MODEL" \
    --spk-embed-model "$SPK_MODEL" \
    --sense-voice "$SENSE_VOICE" \
    --tokens "$TOKENS" \
    --chunk-duration "$CHUNK_DURATION" \
    --partial-interval "$PARTIAL_INTERVAL" \
    --max-segment-duration "$MAX_SEGMENT" \
    --sv-threshold "$SV_THRESHOLD" \
    --provider cpu \
    $EXTRA_ARGS
