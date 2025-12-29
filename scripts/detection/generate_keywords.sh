#!/bin/bash
# 使用 sherpa-onnx-cli text2token 将 keywords_raw.txt 转换为 keywords.txt
# 
# 工作原理：
# - 输入：keywords_raw.txt（中文/英文短语，每行一个）
# - 输出：keywords.txt（拼音/BPE token 格式，用于 KWS 模型）
#
# tokens-type 选项：
# - ppinyin: 部分拼音（声母+韵母分开），适合中文模型
# - fpinyin: 完整拼音（带声调），适合中文模型  
# - bpe: BPE 分词，适合英文模型
# - cjkchar+bpe: 中英混合，适合中英双语模型

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
MODELS_DIR="${ROOT_DIR}/models"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "用法: $0 <model_dir> [tokens_type]"
    echo ""
    echo "参数:"
    echo "  model_dir    模型目录路径（需包含 tokens.txt 和 keywords_raw.txt）"
    echo "  tokens_type  分词类型（可选，默认自动检测）"
    echo "               - ppinyin: 部分拼音（中文）"
    echo "               - fpinyin: 完整拼音（中文）"
    echo "               - bpe: BPE 分词（英文）"
    echo "               - cjkchar+bpe: 中英混合"
    echo ""
    echo "示例:"
    echo "  $0 ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    echo "  $0 ../../models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 cjkchar+bpe"
    echo ""
    echo "唤醒词文件格式 (keywords_raw.txt):"
    echo "  小爱同学"
    echo "  你好问问 :2.0 #0.6 @你好问问"
    echo "  HELLO WORLD :1.5 #0.4"
    echo ""
    echo "  可选参数说明:"
    echo "    :score   - boosting score（分数加成）"
    echo "    #thresh  - 触发阈值"
    echo "    @text    - 输出显示文本"
}

# 检测模型类型
detect_tokens_type() {
    local model_dir="$1"
    local model_name=$(basename "$model_dir")
    
    # 根据模型名称推断类型
    if [[ "$model_name" == *"zh-en"* ]]; then
        echo "cjkchar+bpe"
    elif [[ "$model_name" == *"wenetspeech"* ]]; then
        echo "ppinyin"
    elif [[ "$model_name" == *"gigaspeech"* ]]; then
        echo "bpe"
    else
        # 默认使用 ppinyin（中文）
        echo "ppinyin"
    fi
}

# 检查 BPE 模型文件
find_bpe_model() {
    local model_dir="$1"
    local bpe_file=""
    
    # 查找 bpe.model 文件（标准 SentencePiece 格式）
    if [ -f "${model_dir}/bpe.model" ]; then
        bpe_file="${model_dir}/bpe.model"
    fi
    # 注意：en.phone 是 Lexicon 词典，不是 BPE 模型
    
    echo "$bpe_file"
}

# 主逻辑
main() {
    if [ $# -lt 1 ]; then
        print_usage
        exit 1
    fi
    
    local MODEL_DIR="$1"
    local TOKENS_TYPE="${2:-}"
    
    # 转换为绝对路径
    if [[ ! "$MODEL_DIR" = /* ]]; then
        MODEL_DIR="$(cd "$SCRIPT_DIR" && cd "$MODEL_DIR" && pwd)"
    fi
    
    echo -e "${GREEN}======================================"
    echo "sherpa-onnx keywords.txt 生成工具"
    echo -e "======================================${NC}"
    
    # 检查模型目录
    if [ ! -d "$MODEL_DIR" ]; then
        echo -e "${RED}错误: 模型目录不存在: $MODEL_DIR${NC}"
        exit 1
    fi
    
    # 查找 tokens.txt
    local TOKENS_FILE="${MODEL_DIR}/tokens.txt"
    if [ ! -f "$TOKENS_FILE" ]; then
        echo -e "${RED}错误: tokens.txt 不存在: $TOKENS_FILE${NC}"
        exit 1
    fi
    
    # 查找 keywords_raw.txt
    local KEYWORDS_RAW=""
    if [ -f "${MODEL_DIR}/keywords_raw.txt" ]; then
        KEYWORDS_RAW="${MODEL_DIR}/keywords_raw.txt"
    elif [ -f "${MODEL_DIR}/test_wavs/keywords_raw.txt" ]; then
        KEYWORDS_RAW="${MODEL_DIR}/test_wavs/keywords_raw.txt"
    else
        echo -e "${YELLOW}警告: 未找到 keywords_raw.txt，将创建示例文件${NC}"
        KEYWORDS_RAW="${MODEL_DIR}/keywords_raw.txt"
        echo "你好真真 @你好真真" > "$KEYWORDS_RAW"
        echo "已创建示例文件: $KEYWORDS_RAW"
    fi
    
    # 确定输出路径
    local KEYWORDS_OUT=""
    if [ -f "${MODEL_DIR}/test_wavs/keywords_raw.txt" ]; then
        KEYWORDS_OUT="${MODEL_DIR}/test_wavs/keywords.txt"
    else
        KEYWORDS_OUT="${MODEL_DIR}/keywords.txt"
    fi
    
    # 自动检测 tokens_type
    if [ -z "$TOKENS_TYPE" ]; then
        TOKENS_TYPE=$(detect_tokens_type "$MODEL_DIR")
        echo -e "${YELLOW}自动检测 tokens_type: $TOKENS_TYPE${NC}"
    fi
    
    # 构建命令
    local CMD="sherpa-onnx-cli text2token --tokens \"$TOKENS_FILE\" --tokens-type $TOKENS_TYPE"
    
    # 如果是 BPE 或混合模式，需要 bpe-model
    if [[ "$TOKENS_TYPE" == *"bpe"* ]]; then
        local BPE_MODEL=$(find_bpe_model "$MODEL_DIR")
        if [ -n "$BPE_MODEL" ] && [ -f "$BPE_MODEL" ]; then
            CMD="$CMD --bpe-model \"$BPE_MODEL\""
            echo "使用 BPE 模型: $BPE_MODEL"
        else
            echo -e "${YELLOW}警告: 未找到 BPE 模型文件，可能会失败${NC}"
        fi
    fi
    
    CMD="$CMD \"$KEYWORDS_RAW\" \"$KEYWORDS_OUT\""
    
    echo ""
    echo "模型目录: $MODEL_DIR"
    echo "输入文件: $KEYWORDS_RAW"
    echo "输出文件: $KEYWORDS_OUT"
    echo "分词类型: $TOKENS_TYPE"
    echo ""
    echo "执行命令:"
    echo "$CMD"
    echo ""
    
    # 执行命令
    if eval $CMD; then
        echo ""
        echo -e "${GREEN}✓ 成功生成 keywords.txt${NC}"
        echo ""
        echo "生成内容预览:"
        echo "----------------------------------------"
        head -10 "$KEYWORDS_OUT"
        echo "----------------------------------------"
        echo ""
        echo "输出文件: $KEYWORDS_OUT"
    else
        echo -e "${RED}✗ 生成失败${NC}"
        exit 1
    fi
}

main "$@"
