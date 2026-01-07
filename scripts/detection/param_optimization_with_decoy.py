#!/usr/bin/env python3
"""KWS 参数优化脚本 - 诱导词防御矩阵版本。

使用诱导词 (Decoy Keywords) 机制，通过 Beam Search 竞争让模型区分声调。
评估指标：
- FRR (False Rejection Rate): 漏报率，最小化
- FA_true: 真实负样本误报，最小化
- Decoy拦截率: 谐音词被诱导词拦截的比例，期望100%
"""

from __future__ import annotations

import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "DejaVu Sans",
    "Arial Unicode MS",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 添加 src 到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import create_kws_model

# 诱导词集合（排除目标词）
# 注意: "你好甄甄" 与目标词拼音完全相同，已移除
DECOY_KEYWORDS = {
    "你好镇镇",
    "你好诊诊",
    "你好振振",
    "你好正正",
    "你好争争",
    "你好整整",
    "你好征征",
    "你好认认",
    "你好曾曾",
    "你好怎怎",
}

TARGET_KEYWORD = "你好真真"


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def test_config_with_decoy(
    model_dir: str,
    keywords_file: str,
    positive_samples: list,
    homophone_samples: list,
    true_negative_samples: list,
) -> dict:
    """测试单个配置（使用诱导词文件）。

    Args:
        model_dir: 模型目录
        keywords_file: keywords.txt 文件路径（包含目标词+诱导词）
        positive_samples: 正样本列表
        homophone_samples: 谐音负样本列表
        true_negative_samples: 真实负样本列表

    Returns:
        结果字典，包含 tp, fp, fn, fa_decoy, fa_true, frr, decoy_intercept_rate
    """
    kws = create_kws_model(
        model_dir,
        keywords_file=keywords_file,
        use_int8=True,
        epoch=12,
        avg=2,
        keywords_threshold=0.25,
    )

    # 正样本测试
    tp = 0
    fp = 0  # 正样本中误识别为诱导词
    for s in positive_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                tp += 1
            elif detected_keyword in DECOY_KEYWORDS:
                fp += 1  # 正样本被误判为诱导词

    # 谐音负样本测试（期望被诱导词拦截）
    fa_decoy_intercepted = 0  # 被诱导词成功拦截
    fa_decoy_leak = 0  # 漏过诱导词，被识别为目标词
    for s in homophone_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword in DECOY_KEYWORDS:
                fa_decoy_intercepted += 1  # 成功拦截
            elif detected_keyword == TARGET_KEYWORD:
                fa_decoy_leak += 1  # 拦截失败

    # 真实负样本测试
    fa_true = 0
    for s in true_negative_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                fa_true += 1

    fn = len(positive_samples) - tp - fp
    frr = fn / len(positive_samples) * 100 if positive_samples else 0

    # 诱导词拦截率（谐音样本中被诱导词拦截的比例）
    total_homophone_detections = fa_decoy_intercepted + fa_decoy_leak
    decoy_intercept_rate = (
        fa_decoy_intercepted / total_homophone_detections * 100
        if total_homophone_detections > 0
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fa_decoy_intercepted": fa_decoy_intercepted,
        "fa_decoy_leak": fa_decoy_leak,
        "fa_true": fa_true,
        "frr": frr,
        "decoy_intercept_rate": decoy_intercept_rate,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KWS 参数优化（诱导词版本）")
    parser.add_argument(
        "--keywords-file",
        type=str,
        default=str(_ROOT_DIR / "test/detection/decoy_keywords.txt"),
        help="诱导词 keywords.txt 文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT_DIR / "test/detection"),
        help="输出目录",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="运行基线测试（仅目标词，无诱导词）",
    )
    args = parser.parse_args()

    model_dir = str(
        _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    )

    # 加载测试数据（优先使用合并数据集）
    merged_meta_path = _ROOT_DIR / "dataset/kws_test_data_merged/metadata.json"
    original_meta_path = _ROOT_DIR / "dataset/kws_test_data/metadata.json"

    if merged_meta_path.exists():
        print("使用合并数据集 (kws_test_data_merged)")
        with open(merged_meta_path, "r") as f:
            meta = json.load(f)
    else:
        print("使用原始数据集 (kws_test_data)")
        with open(original_meta_path, "r") as f:
            meta = json.load(f)

    # 数据集分类 - 更新谐音词列表
    homophones = [
        "你好镇镇",
        "你好诊诊",
        "你好正正",
        "你好争争",
        "你好整整",
        "你好认认",
        "你好曾曾",
        "你好怎怎",
        "李浩真真",
        "泥豪真真",
        "你好珍珍",  # 一声同音词（与目标词音频相同）
    ]
    homophone_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]
    true_negative_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") not in homophones
    ]
    positive_samples = meta["positive_samples"]

    print("=" * 70)
    print("KWS 参数优化 - 诱导词防御矩阵版本")
    print("=" * 70)
    print(f"正样本: {len(positive_samples)}")
    print(f"谐音负样本: {len(homophone_samples)}")
    print(f"真实负样本: {len(true_negative_samples)}")
    print(f"诱导词文件: {args.keywords_file}")
    print()

    # 测试单个配置
    print("开始测试...")
    result = test_config_with_decoy(
        model_dir,
        args.keywords_file,
        positive_samples,
        homophone_samples,
        true_negative_samples,
    )

    print()
    print("=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"正样本 (Total: {len(positive_samples)})")
    print(f"  ✓ 正确识别 (TP): {result['tp']}")
    print(f"  ✗ 误判为诱导词 (FP): {result['fp']}")
    print(f"  ✗ 漏检 (FN): {result['fn']}")
    print(f"  漏报率 (FRR): {result['frr']:.2f}%")
    print()
    print(f"谐音负样本 (Total: {len(homophone_samples)})")
    print(f"  ✓ 被诱导词拦截: {result['fa_decoy_intercepted']}")
    print(f"  ✗ 漏过拦截: {result['fa_decoy_leak']}")
    print(f"  诱导词拦截率: {result['decoy_intercept_rate']:.1f}%")
    print()
    print(f"真实负样本 (Total: {len(true_negative_samples)})")
    print(f"  ✗ 误报 (FA_true): {result['fa_true']}")
    print(f"  误报率: {result['fa_true'] / len(true_negative_samples) * 100:.2f}%")
    print()

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"decoy_test_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "keywords_file": args.keywords_file,
                "result": result,
                "dataset_info": {
                    "positive_samples": len(positive_samples),
                    "homophone_samples": len(homophone_samples),
                    "true_negative_samples": len(true_negative_samples),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"结果已保存到: {json_path}")


if __name__ == "__main__":
    main()
