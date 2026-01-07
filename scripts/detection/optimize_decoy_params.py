#!/usr/bin/env python3
"""
优化 Decoy Keywords 参数的网格搜索脚本

支持两种模式：
1. uniform: 统一参数搜索 (所有关键词使用相同的 boost/threshold)
2. differential: 差异化参数搜索 (目标关键词和诱饵关键词使用不同的 boost)

目标：
- FRR ≤ 2%
- Decoy intercept rate ≥ 95%
- FA_true = 0
"""

import os
import sys
import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.model import create_kws_model
from src.detection.decoy_filter import DecoyFilter, DECOY_KEYWORDS_NIHAO_ZHENZHEN

# 使用 "你好真真" 的诱饵关键词集合
DECOY_KEYWORDS = DECOY_KEYWORDS_NIHAO_ZHENZHEN

# 默认路径
DEFAULT_MODEL_DIR = "../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
DEFAULT_RAW_FILE = "../../test/detection/decoy_keywords_clean.txt"
DEFAULT_TOKENS_FILE = (
    "../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
)
DEFAULT_POSITIVE_DIR = "../../test/detection/positive_samples"
DEFAULT_HOMOPHONE_DIR = "../../test/detection/homophone_samples"
DEFAULT_TRUE_NEGATIVE_DIR = "../../test/detection/true_negative_samples"


def generate_keywords_file(
    raw_file: str,
    tokens_file: str,
    output_file: str,
    target_boost: float,
    target_threshold: float,
    decoy_boost: float = None,
    decoy_threshold: float = None,
) -> bool:
    """
    生成带有指定参数的 keywords.txt 文件

    Args:
        raw_file: 原始关键词文件 (中文)
        tokens_file: tokens.txt 文件路径
        output_file: 输出的 keywords.txt 文件路径
        target_boost: 目标关键词的 boost 值
        target_threshold: 目标关键词的 threshold 值
        decoy_boost: 诱饵关键词的 boost 值 (None 则使用 target_boost)
        decoy_threshold: 诱饵关键词的 threshold 值 (None 则使用 target_threshold)
    """
    if decoy_boost is None:
        decoy_boost = target_boost
    if decoy_threshold is None:
        decoy_threshold = target_threshold

    # 读取原始关键词文件
    with open(raw_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # 创建临时文件，添加参数
    temp_raw = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".txt"
    )
    try:
        for line in lines:
            # 解析原始行，移除旧的参数
            parts = line.split("@")
            if len(parts) == 2:
                keyword_text = parts[0].strip().split(":")[0].split("#")[0].strip()
                display_text = parts[1].strip()
            else:
                keyword_text = line.split(":")[0].split("#")[0].strip()
                display_text = keyword_text

            # 判断是目标还是诱饵
            if "真真" in keyword_text:
                # 目标关键词
                temp_raw.write(
                    f"{keyword_text} :{target_boost} #{target_threshold} @{display_text}\n"
                )
            else:
                # 诱饵关键词
                temp_raw.write(
                    f"{keyword_text} :{decoy_boost} #{decoy_threshold} @{display_text}\n"
                )

        temp_raw.close()

        # 使用 sherpa-onnx-cli 生成 token 格式
        cmd = [
            "sherpa-onnx-cli",
            "text2token",
            "--tokens",
            tokens_file,
            "--tokens-type",
            "ppinyin",
            temp_raw.name,
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating keywords file: {result.stderr}")
            return False

        return True

    finally:
        # 清理临时文件
        if os.path.exists(temp_raw.name):
            os.unlink(temp_raw.name)


def load_audio_files(directory: str) -> List[str]:
    """加载目录中的所有 WAV 文件"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return sorted(wav_files)


def test_config(
    keywords_file: str,
    model_dir: str,
    positive_files: List[str],
    homophone_files: List[str],
    true_negative_files: List[str],
    provider: str = "cpu",
) -> Dict:
    """
    测试单个配置的性能

    Returns:
        包含各项指标的字典
    """
    # 创建 KWS 模型
    model = create_kws_model(
        model_dir=model_dir,
        keywords_file=keywords_file,
        provider=provider,
        num_threads=4,
    )

    # 创建 Decoy Filter
    decoy_filter = DecoyFilter(DECOY_KEYWORDS)

    # 统计指标
    tp = 0  # True Positive: 正样本检测到目标关键词
    fp_decoy = 0  # False Positive (Decoy): 正样本误检测为诱饵关键词
    fn = 0  # False Negative: 正样本未检测到任何关键词

    fa_decoy_intercepted = 0  # 谐音样本被诱饵拦截 (成功)
    fa_decoy_leak = 0  # 谐音样本误识别为目标 (失败)

    fa_true = 0  # 真实负样本误识别为目标 (严重错误)

    # 测试正样本
    for audio_file in positive_files:
        stream = model.create_stream()
        import sherpa_onnx

        samples, sample_rate = sherpa_onnx.read_wave(audio_file)
        stream.accept_waveform(sample_rate, samples)

        detected = []
        while model.is_ready(stream):
            model.decode_stream(stream)
            keyword = model.get_result(stream)
            if keyword:
                detected.append(keyword)

        # 应用 Decoy Filter
        filtered = decoy_filter.filter(detected)

        if filtered:
            tp += 1
        elif detected:
            fp_decoy += 1
        else:
            fn += 1

    # 测试谐音负样本
    for audio_file in homophone_files:
        stream = model.create_stream()
        import sherpa_onnx

        samples, sample_rate = sherpa_onnx.read_wave(audio_file)
        stream.accept_waveform(sample_rate, samples)

        detected = []
        while model.is_ready(stream):
            model.decode_stream(stream)
            keyword = model.get_result(stream)
            if keyword:
                detected.append(keyword)

        # 应用 Decoy Filter
        filtered = decoy_filter.filter(detected)

        if detected and not filtered:
            fa_decoy_leak += 1  # 误识别为目标
        elif detected:
            fa_decoy_intercepted += 1  # 成功拦截

    # 测试真实负样本
    for audio_file in true_negative_files:
        stream = model.create_stream()
        import sherpa_onnx

        samples, sample_rate = sherpa_onnx.read_wave(audio_file)
        stream.accept_waveform(sample_rate, samples)

        detected = []
        while model.is_ready(stream):
            model.decode_stream(stream)
            keyword = model.get_result(stream)
            if keyword:
                detected.append(keyword)

        # 应用 Decoy Filter
        filtered = decoy_filter.filter(detected)

        if filtered:
            fa_true += 1

    # 计算指标
    total_positive = len(positive_files)
    total_homophone = len(homophone_files)
    total_true_negative = len(true_negative_files)

    frr = (fn / total_positive * 100) if total_positive > 0 else 0

    total_decoy = fa_decoy_intercepted + fa_decoy_leak
    decoy_intercept_rate = (
        (fa_decoy_intercepted / total_decoy * 100) if total_decoy > 0 else 0
    )

    fa_true_rate = (
        (fa_true / total_true_negative * 100) if total_true_negative > 0 else 0
    )

    return {
        "tp": tp,
        "fp_decoy": fp_decoy,
        "fn": fn,
        "fa_decoy_intercepted": fa_decoy_intercepted,
        "fa_decoy_leak": fa_decoy_leak,
        "fa_true": fa_true,
        "frr": frr,
        "decoy_intercept_rate": decoy_intercept_rate,
        "fa_true_rate": fa_true_rate,
        "total_positive": total_positive,
        "total_homophone": total_homophone,
        "total_true_negative": total_true_negative,
    }


def uniform_search(
    raw_file: str,
    tokens_file: str,
    model_dir: str,
    positive_files: List[str],
    homophone_files: List[str],
    true_negative_files: List[str],
    provider: str = "cpu",
) -> List[Dict]:
    """
    统一参数搜索：所有关键词使用相同的 boost 和 threshold

    搜索范围：
    - boost: [1.5, 1.8, 2.0, 2.2, 2.5]
    - threshold: [0.40, 0.42, 0.45, 0.47, 0.50]

    共 25 种配置
    """
    boost_values = [1.5, 1.8, 2.0, 2.2, 2.5]
    threshold_values = [0.40, 0.42, 0.45, 0.47, 0.50]

    results = []
    total_configs = len(boost_values) * len(threshold_values)
    current = 0

    print(f"开始统一参数搜索，共 {total_configs} 种配置...")
    print()

    for boost in boost_values:
        for threshold in threshold_values:
            current += 1
            print(
                f"[{current}/{total_configs}] 测试配置: boost={boost}, threshold={threshold}"
            )

            # 生成 keywords 文件
            temp_keywords = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            temp_keywords.close()

            try:
                success = generate_keywords_file(
                    raw_file=raw_file,
                    tokens_file=tokens_file,
                    output_file=temp_keywords.name,
                    target_boost=boost,
                    target_threshold=threshold,
                )

                if not success:
                    print(f"  ⚠️  生成 keywords 文件失败，跳过")
                    continue

                # 测试配置
                metrics = test_config(
                    keywords_file=temp_keywords.name,
                    model_dir=model_dir,
                    positive_files=positive_files,
                    homophone_files=homophone_files,
                    true_negative_files=true_negative_files,
                    provider=provider,
                )

                config_result = {
                    "mode": "uniform",
                    "boost": boost,
                    "threshold": threshold,
                    "target_boost": boost,
                    "target_threshold": threshold,
                    "decoy_boost": boost,
                    "decoy_threshold": threshold,
                    **metrics,
                }

                results.append(config_result)

                # 打印结果
                print(
                    f"  FRR: {metrics['frr']:.2f}%, "
                    f"Intercept: {metrics['decoy_intercept_rate']:.1f}%, "
                    f"FA_true: {metrics['fa_true']}"
                )

                # 检查是否满足目标
                meets_target = (
                    metrics["frr"] <= 2.0
                    and metrics["decoy_intercept_rate"] >= 95.0
                    and metrics["fa_true"] == 0
                )
                if meets_target:
                    print(f"  ✅ 满足目标！")

                print()

            finally:
                # 清理临时文件
                if os.path.exists(temp_keywords.name):
                    os.unlink(temp_keywords.name)

    return results


def differential_search(
    raw_file: str,
    tokens_file: str,
    model_dir: str,
    positive_files: List[str],
    homophone_files: List[str],
    true_negative_files: List[str],
    provider: str = "cpu",
) -> List[Dict]:
    """
    差异化参数搜索：目标关键词和诱饵关键词使用不同的 boost

    搜索范围：
    - target_boost: [2.2, 2.5, 2.8, 3.0]
    - decoy_boost: [1.5, 1.8, 2.0]
    - threshold: 固定 0.40 (基于假设：降低 threshold 可提高 intercept rate)

    共 12 种配置
    """
    target_boost_values = [2.2, 2.5, 2.8, 3.0]
    decoy_boost_values = [1.5, 1.8, 2.0]
    threshold = 0.40

    results = []
    total_configs = len(target_boost_values) * len(decoy_boost_values)
    current = 0

    print(f"开始差异化参数搜索，共 {total_configs} 种配置...")
    print(f"固定 threshold={threshold}")
    print()

    for target_boost in target_boost_values:
        for decoy_boost in decoy_boost_values:
            current += 1
            print(
                f"[{current}/{total_configs}] 测试配置: target_boost={target_boost}, decoy_boost={decoy_boost}"
            )

            # 生成 keywords 文件
            temp_keywords = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            temp_keywords.close()

            try:
                success = generate_keywords_file(
                    raw_file=raw_file,
                    tokens_file=tokens_file,
                    output_file=temp_keywords.name,
                    target_boost=target_boost,
                    target_threshold=threshold,
                    decoy_boost=decoy_boost,
                    decoy_threshold=threshold,
                )

                if not success:
                    print(f"  ⚠️  生成 keywords 文件失败，跳过")
                    continue

                # 测试配置
                metrics = test_config(
                    keywords_file=temp_keywords.name,
                    model_dir=model_dir,
                    positive_files=positive_files,
                    homophone_files=homophone_files,
                    true_negative_files=true_negative_files,
                    provider=provider,
                )

                config_result = {
                    "mode": "differential",
                    "target_boost": target_boost,
                    "target_threshold": threshold,
                    "decoy_boost": decoy_boost,
                    "decoy_threshold": threshold,
                    **metrics,
                }

                results.append(config_result)

                # 打印结果
                print(
                    f"  FRR: {metrics['frr']:.2f}%, "
                    f"Intercept: {metrics['decoy_intercept_rate']:.1f}%, "
                    f"FA_true: {metrics['fa_true']}"
                )

                # 检查是否满足目标
                meets_target = (
                    metrics["frr"] <= 2.0
                    and metrics["decoy_intercept_rate"] >= 95.0
                    and metrics["fa_true"] == 0
                )
                if meets_target:
                    print(f"  ✅ 满足目标！")

                print()

            finally:
                # 清理临时文件
                if os.path.exists(temp_keywords.name):
                    os.unlink(temp_keywords.name)

    return results


def main():
    parser = argparse.ArgumentParser(description="优化 Decoy Keywords 参数")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["uniform", "differential"],
        required=True,
        help="搜索模式: uniform (统一参数) 或 differential (差异化参数)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="KWS 模型目录"
    )
    parser.add_argument(
        "--raw-file", type=str, default=DEFAULT_RAW_FILE, help="原始关键词文件 (中文)"
    )
    parser.add_argument(
        "--tokens-file",
        type=str,
        default=DEFAULT_TOKENS_FILE,
        help="tokens.txt 文件路径",
    )
    parser.add_argument(
        "--positive-dir", type=str, default=DEFAULT_POSITIVE_DIR, help="正样本目录"
    )
    parser.add_argument(
        "--homophone-dir",
        type=str,
        default=DEFAULT_HOMOPHONE_DIR,
        help="谐音负样本目录",
    )
    parser.add_argument(
        "--true-negative-dir",
        type=str,
        default=DEFAULT_TRUE_NEGATIVE_DIR,
        help="真实负样本目录",
    )
    parser.add_argument(
        "--provider", type=str, default="cpu", choices=["cpu", "cuda"], help="推理设备"
    )
    parser.add_argument(
        "--output", type=str, default="optimization_results.json", help="输出结果文件"
    )

    args = parser.parse_args()

    # 加载音频文件
    print("加载音频文件...")
    positive_files = load_audio_files(args.positive_dir)
    homophone_files = load_audio_files(args.homophone_dir)
    true_negative_files = load_audio_files(args.true_negative_dir)

    print(f"正样本: {len(positive_files)} 个文件")
    print(f"谐音负样本: {len(homophone_files)} 个文件")
    print(f"真实负样本: {len(true_negative_files)} 个文件")
    print()

    # 执行搜索
    if args.mode == "uniform":
        results = uniform_search(
            raw_file=args.raw_file,
            tokens_file=args.tokens_file,
            model_dir=args.model_dir,
            positive_files=positive_files,
            homophone_files=homophone_files,
            true_negative_files=true_negative_files,
            provider=args.provider,
        )
    else:
        results = differential_search(
            raw_file=args.raw_file,
            tokens_file=args.tokens_file,
            model_dir=args.model_dir,
            positive_files=positive_files,
            homophone_files=homophone_files,
            true_negative_files=true_negative_files,
            provider=args.provider,
        )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {args.output}")
    print()

    # 筛选满足目标的配置
    meeting_target = [
        r
        for r in results
        if r["frr"] <= 2.0 and r["decoy_intercept_rate"] >= 95.0 and r["fa_true"] == 0
    ]

    if meeting_target:
        print(f"✅ 找到 {len(meeting_target)} 个满足目标的配置：")
        print()
        for i, config in enumerate(meeting_target, 1):
            if config["mode"] == "uniform":
                print(f"{i}. boost={config['boost']}, threshold={config['threshold']}")
            else:
                print(
                    f"{i}. target_boost={config['target_boost']}, decoy_boost={config['decoy_boost']}, threshold={config['target_threshold']}"
                )
            print(
                f"   FRR: {config['frr']:.2f}%, Intercept: {config['decoy_intercept_rate']:.1f}%, FA_true: {config['fa_true']}"
            )
            print()
    else:
        print("⚠️  未找到完全满足目标的配置")
        print()
        print("最接近的配置：")
        # 按综合得分排序 (优先 FRR, 其次 intercept rate)
        sorted_results = sorted(
            results, key=lambda x: (x["frr"], -x["decoy_intercept_rate"], x["fa_true"])
        )
        for i, config in enumerate(sorted_results[:5], 1):
            if config["mode"] == "uniform":
                print(f"{i}. boost={config['boost']}, threshold={config['threshold']}")
            else:
                print(
                    f"{i}. target_boost={config['target_boost']}, decoy_boost={config['decoy_boost']}, threshold={config['target_threshold']}"
                )
            print(
                f"   FRR: {config['frr']:.2f}%, Intercept: {config['decoy_intercept_rate']:.1f}%, FA_true: {config['fa_true']}"
            )
            print()


if __name__ == "__main__":
    main()
