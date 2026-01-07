#!/usr/bin/env python3
"""Keywords 生成工具。

使用 sherpa-onnx-cli text2token 将 keywords_raw.txt 转换为 keywords.txt。

支持的格式：
- ppinyin: 部分拼音（声母+韵母分开），适合纯中文模型
- cjkchar+bpe: 中英混合（需要 bpe.model），适合中英双语模型

对于没有 bpe.model 的 zh-en 模型（如使用 Lexicon 的模型），
会回退使用 ppinyin 处理中文部分，英文需要手动处理。

使用示例:
    python generate_keywords.py --model-dir ../../models/xxx --keyword "你好真真"
    python generate_keywords.py --raw-file keywords_raw.txt --tokens-file tokens.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


def detect_tokens_type(model_dir: Path) -> str:
    """根据模型目录推断 tokens 类型。"""
    model_name = model_dir.name.lower()

    # 检查是否有 bpe.model（真正的 BPE 模型）
    if (model_dir / "bpe.model").exists():
        return "cjkchar+bpe"

    # 根据模型名称推断
    if "zh-en" in model_name or "zh_en" in model_name:
        # zh-en 模型如果没有 bpe.model，使用 ppinyin 处理中文
        return "ppinyin"
    elif "wenetspeech" in model_name:
        return "ppinyin"
    elif "gigaspeech" in model_name:
        return "bpe"

    return "ppinyin"


def find_keywords_raw(model_dir: Path) -> Optional[Path]:
    """查找 keywords_raw.txt 文件。"""
    candidates = [
        model_dir / "keywords_raw.txt",
        model_dir / "test_wavs" / "keywords_raw.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def generate_keywords_file(
    tokens_file: Path,
    input_file: Path,
    output_file: Path,
    tokens_type: str = "ppinyin",
    bpe_model: Optional[Path] = None,
) -> bool:
    """使用 sherpa-onnx-cli 生成 keywords.txt。

    Args:
        tokens_file: tokens.txt 路径
        input_file: 输入的 keywords_raw.txt
        output_file: 输出的 keywords.txt
        tokens_type: 分词类型
        bpe_model: BPE 模型路径（可选）

    Returns:
        是否成功
    """
    cmd = [
        "sherpa-onnx-cli",
        "text2token",
        "--tokens",
        str(tokens_file),
        "--tokens-type",
        tokens_type,
    ]

    if bpe_model and bpe_model.exists():
        cmd.extend(["--bpe-model", str(bpe_model)])

    cmd.extend([str(input_file), str(output_file)])

    print(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ 成功生成: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 生成失败: {e.stderr}")
        return False


def generate_single_keyword(
    keyword: str,
    tokens_file: Path,
    tokens_type: str = "ppinyin",
    display_text: Optional[str] = None,
) -> Optional[str]:
    """生成单个唤醒词的拼音表示。

    Args:
        keyword: 唤醒词文本（如 "你好真真"）
        tokens_file: tokens.txt 路径
        tokens_type: 分词类型
        display_text: 显示文本（可选，默认与 keyword 相同）

    Returns:
        拼音格式的唤醒词，如 "n ǐ h ǎo zh ēn zh ēn @你好真真"
    """
    import tempfile

    display = display_text or keyword

    # 创建临时输入文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"{keyword} @{display}\n")
        input_path = Path(f.name)

    # 创建临时输出文件
    output_path = input_path.with_suffix(".out.txt")

    try:
        cmd = [
            "sherpa-onnx-cli",
            "text2token",
            "--tokens",
            str(tokens_file),
            "--tokens-type",
            tokens_type,
            str(input_path),
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with open(output_path, "r", encoding="utf-8") as f:
            result = f.read().strip()

        return result

    except subprocess.CalledProcessError as e:
        print(f"错误: {e.stderr}")
        return None
    finally:
        # 清理临时文件
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Keywords 生成工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        help="模型目录路径",
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        help="keywords_raw.txt 文件路径（直接指定，覆盖 model-dir）",
    )
    parser.add_argument(
        "--tokens-file",
        type=str,
        help="tokens.txt 文件路径（直接指定，覆盖 model-dir）",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="输出文件路径（可选）",
    )
    parser.add_argument(
        "--tokens-type",
        type=str,
        choices=["ppinyin", "fpinyin", "bpe", "cjkchar", "cjkchar+bpe"],
        help="分词类型（可选，默认自动检测）",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        help="单个唤醒词（快速测试模式）",
    )
    parser.add_argument(
        "--display-text",
        type=str,
        help="显示文本（与 --keyword 配合使用）",
    )

    args = parser.parse_args()

    # 快速测试模式：生成单个唤醒词
    if args.keyword:
        tokens_file = None
        if args.tokens_file:
            tokens_file = Path(args.tokens_file)
        elif args.model_dir:
            tokens_file = Path(args.model_dir) / "tokens.txt"

        if not tokens_file or not tokens_file.exists():
            print("错误: 需要指定 --tokens-file 或 --model-dir")
            sys.exit(1)

        tokens_type = args.tokens_type or "ppinyin"
        result = generate_single_keyword(
            args.keyword,
            tokens_file,
            tokens_type,
            args.display_text,
        )

        if result:
            print(f"\n生成结果:")
            print(result)
        sys.exit(0 if result else 1)

    # 批量模式
    if not args.model_dir and not (args.raw_file and args.tokens_file):
        print("错误: 需要指定 --model-dir 或同时指定 --raw-file 和 --tokens-file")
        sys.exit(1)

    if args.model_dir:
        model_dir = Path(args.model_dir)
        tokens_file = model_dir / "tokens.txt"
        raw_file = find_keywords_raw(model_dir)

        if not raw_file:
            print(f"错误: 未找到 keywords_raw.txt in {model_dir}")
            sys.exit(1)

        # 确定输出路径
        if raw_file.parent.name == "test_wavs":
            output_file = raw_file.parent / "keywords.txt"
        else:
            output_file = model_dir / "keywords.txt"

        tokens_type = args.tokens_type or detect_tokens_type(model_dir)
        bpe_model = model_dir / "bpe.model" if "bpe" in tokens_type else None

    else:
        tokens_file = Path(args.tokens_file)
        raw_file = Path(args.raw_file)
        output_file = (
            Path(args.output_file)
            if args.output_file
            else raw_file.with_name("keywords.txt")
        )
        tokens_type = args.tokens_type or "ppinyin"
        bpe_model = None

    if args.output_file:
        output_file = Path(args.output_file)

    print(f"输入文件: {raw_file}")
    print(f"Tokens 文件: {tokens_file}")
    print(f"输出文件: {output_file}")
    print(f"分词类型: {tokens_type}")
    print()

    success = generate_keywords_file(
        tokens_file,
        raw_file,
        output_file,
        tokens_type,
        bpe_model,
    )

    if success:
        print("\n生成内容预览:")
        print("-" * 40)
        with open(output_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 10:
                    print("...")
                    break
                print(line.rstrip())
        print("-" * 40)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
