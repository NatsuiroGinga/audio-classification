#!/usr/bin/env python3
"""全局 keywords_threshold 参数优化 - 测试不同的阈值对 FRR 和 Intercept 的影响"""

import json
import wave
import numpy as np
from pathlib import Path
import sys

_ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import create_kws_model

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


def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    return samples, sr


def test_threshold(
    threshold: float,
    model_dir: str,
    keywords_file: str,
    positive_samples,
    homophone_samples,
    true_negative_samples,
):
    """测试特定 keywords_threshold 值的性能"""
    kws = create_kws_model(
        model_dir,
        keywords_file=keywords_file,
        use_int8=True,
        epoch=12,
        avg=2,
        keywords_threshold=threshold,
    )

    tp = fp = fn = 0
    fa_decoy_intercepted = fa_decoy_leak = fa_true = 0

    for s in positive_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                tp += 1
            elif detected_keyword in DECOY_KEYWORDS:
                fp += 1

    for s in homophone_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword in DECOY_KEYWORDS:
                fa_decoy_intercepted += 1
            elif detected_keyword == TARGET_KEYWORD:
                fa_decoy_leak += 1

    for s in true_negative_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                fa_true += 1

    fn = len(positive_samples) - tp - fp
    frr = fn / len(positive_samples) * 100 if positive_samples else 0

    total_homophone = fa_decoy_intercepted + fa_decoy_leak
    intercept_rate = (
        (fa_decoy_intercepted / total_homophone * 100) if total_homophone > 0 else 0
    )

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fa_decoy_intercepted": fa_decoy_intercepted,
        "fa_decoy_leak": fa_decoy_leak,
        "fa_true": fa_true,
        "frr": frr,
        "intercept_rate": intercept_rate,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="threshold_optimization.json")
    args = parser.parse_args()

    model_dir = str(
        _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    )
    keywords_file = str(_ROOT_DIR / "test/detection/decoy_keywords.txt")

    with open(_ROOT_DIR / "dataset/kws_test_data/metadata.json", "r") as f:
        meta = json.load(f)

    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]
    homophone_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]
    true_negative_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") not in homophones
    ]
    positive_samples = meta["positive_samples"]

    print(
        f"正样本: {len(positive_samples)}, 谐音负样本: {len(homophone_samples)}, 真实负样本: {len(true_negative_samples)}\n"
    )

    # 测试不同的 keywords_threshold 值
    # 较低的 threshold = 更容易触发 = 更低的 FRR，但可能更低的 Intercept
    # 较高的 threshold = 更难触发 = 更高的 FRR，但可能更高的 Intercept
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    results = []
    print("keywords_threshold 优化搜索\n")
    print(
        f"{'Threshold':>10} | {'FRR':>8} | {'Intercept':>10} | {'FA_true':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4}"
    )
    print("-" * 70)

    for threshold in thresholds:
        r = test_threshold(
            threshold,
            model_dir,
            keywords_file,
            positive_samples,
            homophone_samples,
            true_negative_samples,
        )
        results.append(r)

        status = ""
        if r["frr"] <= 3.0 and r["intercept_rate"] >= 80.0 and r["fa_true"] == 0:
            status = " ✅"

        print(
            f"{threshold:>10.2f} | {r['frr']:>7.2f}% | {r['intercept_rate']:>9.1f}% | {r['fa_true']:>8} | {r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4}{status}"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {args.output}\n")

    # 筛选最佳配置
    best = min(results, key=lambda x: (x["frr"], -x["intercept_rate"], x["fa_true"]))
    print(f"推荐配置: keywords_threshold={best['threshold']}")
    print(
        f"  FRR: {best['frr']:.2f}%, Intercept: {best['intercept_rate']:.1f}%, FA_true: {best['fa_true']}"
    )


if __name__ == "__main__":
    main()
