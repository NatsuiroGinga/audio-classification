#!/usr/bin/env python3
"""
分析误分类样本

分析两类误分类：
1. 正样本被误识别为诱饵关键词 (FP_decoy)
2. 谐音样本漏过诱饵拦截 (FA_decoy_leak)

输出：
- 按检测到的诱饵关键词分组
- 音频特征分析 (SNR, duration, 可能的音色特征)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.detection.model import create_kws_model
from src.detection.decoy_filter import DecoyFilter, DECOY_KEYWORDS_NIHAO_ZHENZHEN
import sherpa_onnx

# 使用 "你好真真" 的诱饵关键词集合
DECOY_KEYWORDS = DECOY_KEYWORDS_NIHAO_ZHENZHEN

# 默认路径
DEFAULT_MODEL_DIR = "../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
DEFAULT_KEYWORDS_FILE = "../../test/detection/decoy_keywords.txt"
DEFAULT_METADATA_FILE = "../../dataset/kws_test_data/metadata.json"


def analyze_audio(audio_file: str) -> Dict:
    """
    分析音频文件的基本特征

    Returns:
        包含 duration, sample_rate, num_samples 等信息的字典
    """
    samples, sample_rate = sherpa_onnx.read_wave(audio_file)
    duration = len(samples) / sample_rate

    # 简单的 SNR 估计 (基于能量)
    energy = sum(s**2 for s in samples) / len(samples)

    return {
        "filename": os.path.basename(audio_file),
        "filepath": audio_file,
        "duration": duration,
        "sample_rate": sample_rate,
        "num_samples": len(samples),
        "energy": energy,
    }


def detect_keywords(model, audio_file: str) -> List[str]:
    """
    检测音频中的关键词

    Returns:
        检测到的关键词列表
    """
    stream = model.create_stream()
    samples, sample_rate = sherpa_onnx.read_wave(audio_file)
    stream.accept_waveform(sample_rate, samples)

    detected = []
    while model.is_ready(stream):
        model.decode_stream(stream)
        keyword = model.get_result(stream)
        if keyword:
            detected.append(keyword)

    return detected


def analyze_positive_misclassified(
    model, decoy_filter: DecoyFilter, positive_files: List[str]
) -> Dict:
    """
    分析正样本中被误识别为诱饵关键词的样本

    Returns:
        {
            'total': 总数,
            'fp_decoy': 误识别为诱饵的数量,
            'by_decoy': {诱饵关键词: [样本信息]},
            'samples': [详细样本信息]
        }
    """
    fp_decoy_samples = []
    by_decoy = defaultdict(list)

    for audio_file in positive_files:
        detected = detect_keywords(model, audio_file)
        filtered = decoy_filter.filter(detected)

        # 如果检测到关键词但被过滤掉，说明被误识别为诱饵
        if detected and not filtered:
            audio_info = analyze_audio(audio_file)
            sample_info = {
                **audio_info,
                "detected": detected,
                "detected_decoy": [k for k in detected if k in DECOY_KEYWORDS],
            }
            fp_decoy_samples.append(sample_info)

            # 按检测到的诱饵关键词分组
            for decoy in sample_info["detected_decoy"]:
                by_decoy[decoy].append(sample_info)

    return {
        "total": len(positive_files),
        "fp_decoy": len(fp_decoy_samples),
        "fp_decoy_rate": (
            len(fp_decoy_samples) / len(positive_files) * 100 if positive_files else 0
        ),
        "by_decoy": dict(by_decoy),
        "samples": fp_decoy_samples,
    }


def analyze_homophone_leaked(
    model, decoy_filter: DecoyFilter, homophone_files: List[str]
) -> Dict:
    """
    分析谐音样本中漏过诱饵拦截的样本

    Returns:
        {
            'total': 总数,
            'leaked': 漏过的数量,
            'intercepted': 被拦截的数量,
            'leaked_samples': [详细样本信息],
            'intercepted_samples': [详细样本信息]
        }
    """
    leaked_samples = []
    intercepted_samples = []

    for audio_file in homophone_files:
        detected = detect_keywords(model, audio_file)
        filtered = decoy_filter.filter(detected)

        audio_info = analyze_audio(audio_file)

        if detected:
            if filtered:
                # 误识别为目标关键词 (漏过)
                sample_info = {**audio_info, "detected": detected, "filtered": filtered}
                leaked_samples.append(sample_info)
            else:
                # 成功拦截
                sample_info = {
                    **audio_info,
                    "detected": detected,
                    "detected_decoy": [k for k in detected if k in DECOY_KEYWORDS],
                }
                intercepted_samples.append(sample_info)

    total_detected = len(leaked_samples) + len(intercepted_samples)

    return {
        "total": len(homophone_files),
        "leaked": len(leaked_samples),
        "intercepted": len(intercepted_samples),
        "leak_rate": (
            len(leaked_samples) / total_detected * 100 if total_detected > 0 else 0
        ),
        "intercept_rate": (
            len(intercepted_samples) / total_detected * 100 if total_detected > 0 else 0
        ),
        "leaked_samples": leaked_samples,
        "intercepted_samples": intercepted_samples,
    }


def print_analysis_summary(analysis: Dict):
    """打印分析摘要"""
    print("=" * 80)
    print("正样本误分类分析")
    print("=" * 80)
    print()
    print(f"总样本数: {analysis['positive']['total']}")
    print(
        f"误识别为诱饵: {analysis['positive']['fp_decoy']} ({analysis['positive']['fp_decoy_rate']:.1f}%)"
    )
    print()

    if analysis["positive"]["by_decoy"]:
        print("按诱饵关键词分组:")
        for decoy, samples in sorted(
            analysis["positive"]["by_decoy"].items(),
            key=lambda x: len(x[1]),
            reverse=True,
        ):
            print(f"  {decoy}: {len(samples)} 个样本")
        print()

    if analysis["positive"]["samples"]:
        print("详细样本信息 (前 10 个):")
        for i, sample in enumerate(analysis["positive"]["samples"][:10], 1):
            print(f"  {i}. {sample['filename']}")
            print(f"     时长: {sample['duration']:.2f}s, 能量: {sample['energy']:.6f}")
            print(f"     检测到: {', '.join(sample['detected'])}")
            print(f"     诱饵: {', '.join(sample['detected_decoy'])}")
            print()

    print("=" * 80)
    print("谐音样本漏过分析")
    print("=" * 80)
    print()
    print(f"总样本数: {analysis['homophone']['total']}")
    print(
        f"检测到的样本: {analysis['homophone']['leaked'] + analysis['homophone']['intercepted']}"
    )
    print(
        f"漏过 (误识别为目标): {analysis['homophone']['leaked']} ({analysis['homophone']['leak_rate']:.1f}%)"
    )
    print(
        f"拦截 (成功): {analysis['homophone']['intercepted']} ({analysis['homophone']['intercept_rate']:.1f}%)"
    )
    print()

    if analysis["homophone"]["leaked_samples"]:
        print("漏过的样本 (前 10 个):")
        for i, sample in enumerate(analysis["homophone"]["leaked_samples"][:10], 1):
            print(f"  {i}. {sample['filename']}")
            print(f"     时长: {sample['duration']:.2f}s, 能量: {sample['energy']:.6f}")
            print(f"     检测到: {', '.join(sample['detected'])}")
            print()

    if analysis["homophone"]["intercepted_samples"]:
        print("成功拦截的样本 (前 5 个):")
        for i, sample in enumerate(analysis["homophone"]["intercepted_samples"][:5], 1):
            print(f"  {i}. {sample['filename']}")
            print(f"     时长: {sample['duration']:.2f}s, 能量: {sample['energy']:.6f}")
            print(f"     检测到诱饵: {', '.join(sample['detected_decoy'])}")
            print()


def load_audio_files(directory: str) -> List[str]:
    """加载目录中的所有 WAV 文件"""
    wav_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return sorted(wav_files)


def main():
    parser = argparse.ArgumentParser(description="分析误分类样本")
    parser.add_argument(
        "--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="KWS 模型目录"
    )
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=DEFAULT_KEYWORDS_FILE,
        help="关键词文件 (Token 格式)",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default=DEFAULT_METADATA_FILE,
        help="测试数据元数据文件",
    )
    parser.add_argument(
        "--provider", type=str, default="cpu", choices=["cpu", "cuda"], help="推理设备"
    )
    parser.add_argument(
        "--output-json", type=str, default=None, help="输出 JSON 文件 (可选)"
    )

    args = parser.parse_args()

    # 加载测试数据
    print("加载测试数据...")
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 数据集分类
    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]
    homophone_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]
    positive_samples = meta["positive_samples"]

    # 提取文件路径
    positive_files = [s["file"] for s in positive_samples]
    homophone_files = [s["file"] for s in homophone_samples]

    print(f"正样本: {len(positive_files)} 个文件")
    print(f"谐音负样本: {len(homophone_files)} 个文件")
    print()

    # 创建 KWS 模型
    print("初始化 KWS 模型...")
    model = create_kws_model(
        model_dir=args.model_dir,
        keywords_file=args.keywords_file,
        provider=args.provider,
        num_threads=4,
    )

    # 创建 Decoy Filter
    decoy_filter = DecoyFilter(DECOY_KEYWORDS)

    print("开始分析...")
    print()

    # 分析正样本误分类
    positive_analysis = analyze_positive_misclassified(
        model=model, decoy_filter=decoy_filter, positive_files=positive_files
    )

    # 分析谐音样本漏过
    homophone_analysis = analyze_homophone_leaked(
        model=model, decoy_filter=decoy_filter, homophone_files=homophone_files
    )

    # 汇总分析结果
    analysis = {"positive": positive_analysis, "homophone": homophone_analysis}

    # 打印摘要
    print_analysis_summary(analysis)

    # 保存到 JSON (可选)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"详细分析已保存到: {args.output_json}")


if __name__ == "__main__":
    main()
