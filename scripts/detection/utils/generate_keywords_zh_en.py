#!/usr/bin/env python3
"""中英混合 Keywords 生成工具。

对于使用 Lexicon 词典的 zh-en 模型，需要：
1. 英文通过 Lexicon (en.phone) 查找音素
2. 中文通过 pypinyin 转换为拼音

使用示例:
    python generate_keywords_zh_en.py \
        --model-dir ../../models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \
        --keyword "你好真真"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pypinyin import pinyin, Style
    from pypinyin.contrib.tone_convert import to_initials, to_finals_tone
except ImportError:
    print("请安装 pypinyin: pip install pypinyin")
    sys.exit(1)


def load_lexicon(lexicon_path: Path) -> Dict[str, str]:
    """加载 Lexicon 词典。

    Args:
        lexicon_path: 词典文件路径（如 en.phone）

    Returns:
        词典映射 {WORD: "phonemes"}
    """
    lexicon: Dict[str, str] = {}
    with open(lexicon_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                word = parts[0].upper()
                phones = parts[1]
                if word not in lexicon:  # 只取第一个发音
                    lexicon[word] = phones
    return lexicon


def is_chinese(char: str) -> bool:
    """判断是否为中文字符。"""
    return "\u4e00" <= char <= "\u9fff"


def is_english_word(text: str) -> bool:
    """判断是否为英文词。"""
    return bool(re.match(r"^[A-Za-z]+$", text))


def chinese_to_pinyin(text: str) -> str:
    """将中文转换为拼音（ppinyin 格式）。

    Args:
        text: 中文文本

    Returns:
        拼音字符串，如 "n ǐ h ǎo"
    """
    result = []
    for char in text:
        if is_chinese(char):
            # 获取带声调的拼音
            py = pinyin(char, style=Style.TONE3, heteronym=False)[0][0]
            # 分离声母和韵母
            try:
                initial = to_initials(py, strict=False)
                final = to_finals_tone(py, strict=False)
                if initial:
                    result.append(initial)
                if final:
                    result.append(final)
            except Exception:
                # 回退：直接使用完整拼音
                result.append(py)
        else:
            # 非中文字符直接保留
            result.append(char)

    return " ".join(result)


def english_to_phones(text: str, lexicon: Dict[str, str]) -> Optional[str]:
    """将英文词转换为音素。

    Args:
        text: 英文文本（可包含多个词，空格分隔）
        lexicon: Lexicon 词典

    Returns:
        音素字符串，如 "L AY1 T AH1 P"
    """
    words = text.upper().split()
    phones_list = []

    for word in words:
        if word in lexicon:
            phones_list.append(lexicon[word])
        else:
            print(f"警告: 词 '{word}' 不在词典中")
            return None

    return " ".join(phones_list)


def convert_keyword(
    keyword: str,
    lexicon: Optional[Dict[str, str]] = None,
    display_text: Optional[str] = None,
) -> str:
    """转换单个唤醒词。

    Args:
        keyword: 原始唤醒词
        lexicon: Lexicon 词典（英文需要）
        display_text: 显示文本

    Returns:
        转换后的格式，如 "n ǐ h ǎo zh ēn zh ēn @你好真真"
    """
    display = display_text or keyword

    # 检测语言
    has_chinese = any(is_chinese(c) for c in keyword)
    has_english = bool(re.search(r"[A-Za-z]", keyword))

    if has_chinese and not has_english:
        # 纯中文
        tokens = chinese_to_pinyin(keyword)
    elif has_english and not has_chinese:
        # 纯英文
        if lexicon:
            tokens = english_to_phones(keyword, lexicon)
            if tokens is None:
                print(f"错误: 无法转换英文词 '{keyword}'")
                return ""
        else:
            print(f"警告: 需要 Lexicon 来处理英文词 '{keyword}'")
            tokens = keyword.upper()
    else:
        # 中英混合 - 逐词处理
        parts = []
        current_word = ""
        current_type = None  # 'zh' or 'en'

        for char in keyword:
            if char == " ":
                if current_word:
                    if current_type == "zh":
                        parts.append(chinese_to_pinyin(current_word))
                    elif current_type == "en" and lexicon:
                        phones = english_to_phones(current_word, lexicon)
                        if phones:
                            parts.append(phones)
                    current_word = ""
                    current_type = None
            elif is_chinese(char):
                if current_type == "en" and current_word:
                    if lexicon:
                        phones = english_to_phones(current_word, lexicon)
                        if phones:
                            parts.append(phones)
                    current_word = ""
                current_word += char
                current_type = "zh"
            elif char.isalpha():
                if current_type == "zh" and current_word:
                    parts.append(chinese_to_pinyin(current_word))
                    current_word = ""
                current_word += char
                current_type = "en"

        # 处理最后一个词
        if current_word:
            if current_type == "zh":
                parts.append(chinese_to_pinyin(current_word))
            elif current_type == "en" and lexicon:
                phones = english_to_phones(current_word, lexicon)
                if phones:
                    parts.append(phones)

        tokens = " ".join(parts)

    return f"{tokens} @{display}"


def process_keywords_file(
    input_file: Path,
    output_file: Path,
    lexicon: Optional[Dict[str, str]] = None,
) -> bool:
    """处理 keywords_raw.txt 文件。

    Args:
        input_file: 输入文件
        output_file: 输出文件
        lexicon: Lexicon 词典

    Returns:
        是否成功
    """
    results = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 解析行格式：keyword [score] [threshold] [@display]
            parts = line.split("@")
            keyword_part = parts[0].strip()
            display_text = parts[1].strip() if len(parts) > 1 else None

            # 提取可选参数
            extra_params = ""
            for token in keyword_part.split():
                if token.startswith(":") or token.startswith("#"):
                    extra_params += " " + token
                    keyword_part = keyword_part.replace(token, "").strip()

            keyword = (
                keyword_part.split()[0] if " " not in keyword_part else keyword_part
            )

            # 转换
            converted = convert_keyword(keyword, lexicon, display_text)
            if converted:
                if extra_params:
                    # 在 @ 前插入额外参数
                    at_idx = converted.find("@")
                    if at_idx > 0:
                        converted = (
                            converted[:at_idx].rstrip()
                            + extra_params
                            + " "
                            + converted[at_idx:]
                        )
                results.append(converted)

    # 写入输出
    with open(output_file, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"✓ 成功生成: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="中英混合 Keywords 生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        help="模型目录路径",
    )
    parser.add_argument(
        "--lexicon",
        type=str,
        help="Lexicon 文件路径（如 en.phone）",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="输入文件路径",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="输出文件路径",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="单个唤醒词（快速测试）",
    )
    parser.add_argument(
        "--display-text",
        type=str,
        help="显示文本",
    )

    args = parser.parse_args()

    # 加载 Lexicon
    lexicon = None
    if args.lexicon:
        lexicon = load_lexicon(Path(args.lexicon))
        print(f"已加载 Lexicon: {len(lexicon)} 词条")
    elif args.model_dir:
        lexicon_path = Path(args.model_dir) / "en.phone"
        if lexicon_path.exists():
            lexicon = load_lexicon(lexicon_path)
            print(f"已加载 Lexicon: {len(lexicon)} 词条")

    # 单词测试模式
    if args.keyword:
        result = convert_keyword(args.keyword, lexicon, args.display_text)
        print(f"\n转换结果:")
        print(result)
        return

    # 批量模式
    if args.model_dir:
        model_dir = Path(args.model_dir)

        # 查找输入文件
        input_file = None
        for candidate in [
            model_dir / "keywords_raw.txt",
            model_dir / "test_wavs" / "keywords_raw.txt",
        ]:
            if candidate.exists():
                input_file = candidate
                break

        if not input_file:
            print(f"错误: 未找到 keywords_raw.txt")
            sys.exit(1)

        # 确定输出路径
        if input_file.parent.name == "test_wavs":
            output_file = input_file.parent / "keywords.txt"
        else:
            output_file = model_dir / "keywords.txt"

    elif args.input_file:
        input_file = Path(args.input_file)
        output_file = (
            Path(args.output_file)
            if args.output_file
            else input_file.with_name("keywords.txt")
        )
    else:
        print("错误: 需要指定 --model-dir 或 --input-file")
        sys.exit(1)

    if args.output_file:
        output_file = Path(args.output_file)

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print()

    success = process_keywords_file(input_file, output_file, lexicon)

    if success:
        print("\n生成内容预览:")
        print("-" * 50)
        with open(output_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    print("...")
                    break
                print(line.rstrip())
        print("-" * 50)


if __name__ == "__main__":
    main()
