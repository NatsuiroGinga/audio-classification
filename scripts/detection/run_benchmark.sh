#!/bin/bash
# KWS 基准测试运行脚本
# 测试 "你好真真" 唤醒词在多个模型上的表现

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
MODELS_DIR="${ROOT_DIR}/models"

# 唤醒词配置
KEYWORD="n ǐ h ǎo zh ēn zh ēn @你好真真"

# 测试数据目录（请根据实际情况修改）
POSITIVE_DIR="${1:-}"  # 正样本目录（包含唤醒词的音频）
NEGATIVE_DIR="${2:-}"  # 负样本目录（不包含唤醒词的音频）
CUSTOM_OUTPUT_DIR="${3:-}"  # 自定义输出目录（可选）

# 输出目录
if [ -n "$CUSTOM_OUTPUT_DIR" ]; then
    OUTPUT_DIR="$CUSTOM_OUTPUT_DIR"
else
    OUTPUT_DIR="${ROOT_DIR}/benchmark_results/$(date +%Y%m%d_%H%M%S)"
fi

# 模型目录
MODEL_WENETSPEECH="${MODELS_DIR}/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
MODEL_ZH_EN="${MODELS_DIR}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"

echo "======================================"
echo "KWS 基准测试 - 你好真真"
echo "======================================"
echo "唤醒词: ${KEYWORD}"
echo "正样本目录: ${POSITIVE_DIR:-未指定}"
echo "负样本目录: ${NEGATIVE_DIR:-未指定}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

# 检查模型是否存在
check_model() {
    local model_dir="$1"
    local model_name="$2"
    if [ -d "$model_dir" ]; then
        echo "[✓] ${model_name}"
    else
        echo "[✗] ${model_name} (未找到)"
    fi
}

echo "可用模型:"
check_model "$MODEL_WENETSPEECH" "wenetspeech-3.3M"
check_model "$MODEL_ZH_EN" "zh-en-3M"
echo ""

# 构建模型列表
MODELS=""
[ -d "$MODEL_WENETSPEECH" ] && MODELS="${MODEL_WENETSPEECH}"
[ -d "$MODEL_ZH_EN" ] && MODELS="${MODELS:+${MODELS},}${MODEL_ZH_EN}"

if [ -z "$MODELS" ]; then
    echo "错误: 没有找到可用的模型"
    exit 1
fi

# 运行基准测试
cd "$SCRIPT_DIR"

ARGS=(
    --model-dir "$MODELS"
    --keyword "$KEYWORD"
    --output-dir "$OUTPUT_DIR"
    --num-threads 2
    --provider cpu
)

[ -n "$POSITIVE_DIR" ] && ARGS+=(--positive-dir "$POSITIVE_DIR")
[ -n "$NEGATIVE_DIR" ] && ARGS+=(--negative-dir "$NEGATIVE_DIR")

echo "运行命令:"
echo "python benchmark_kws.py ${ARGS[*]}"
echo ""

python benchmark_kws.py "${ARGS[@]}"

echo ""
echo "测试完成！结果保存在: ${OUTPUT_DIR}"
