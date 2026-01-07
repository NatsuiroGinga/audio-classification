#!/usr/bin/env python3
"""简化版参数优化脚本 - 直接使用现有数据加载方式"""

import json
import sys
import tempfile
import subprocess
import wave
import numpy as np
from pathlib import Path
from typing import Dict, List

# 添加项目根目录到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import create_kws_model
from src.detection.decoy_filter import DecoyFilter, DECOY_KEYWORDS_NIHAO_ZHENZHEN

DECOY_KEYWORDS = DECOY_KEYWORDS_NIHAO_ZHENZHEN
TARGET_KEYWORD = "你好真真"


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        samples /= 32768.0
    return samples, sr


def generate_keywords_file(
    raw_file: str,
    tokens_file: str,
    output_file: str,
    target_boost: float,
    target_threshold: float,
    decoy_boost: float = None,
    decoy_threshold: float = None,
) -> bool:
    """生成带有指定参数的 keywords.txt 文件"""
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
                temp_raw.write(
                    f"{keyword_text} :{target_boost} #{target_threshold} @{display_text}\n"
                )
            else:
                temp_raw.write(
                    f"{keyword_text} :{decoy_boost} #{decoy_threshold} @{display_text}\n"
                )

        temp_raw.close()

        # 使用 sherpa-onnx-cli 生成 token 格式
        import shutil

        cli_path = (
            shutil.which("sherpa-onnx-cli")
            or "/data/workspace/llm/anaconda3/envs/default/bin/sherpa-onnx-cli"
        )
        cmd = [
            cli_path,
            "text2token",
            "--tokens",
            tokens_file,
            "--tokens-type",
            "ppinyin",
            temp_raw.name,
            output_file,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    finally:
        import os

        if os.path.exists(temp_raw.name):
            os.unlink(temp_raw.name)


def test_config(
    keywords_file: str,
    model_dir: str,
    positive_samples: List[Dict],
    homophone_samples: List[Dict],
    true_negative_samples: List[Dict],
) -> Dict:
    """测试单个配置的性能"""
    model = create_kws_model(
        model_dir=model_dir, keywords_file=keywords_file, provider="cpu", num_threads=4
    )

    tp = fp = fn = 0
    fa_decoy_intercepted = fa_decoy_leak = fa_true = 0

    # 测试正样本
    for sample in positive_samples:
        samples, sr = read_wav(sample["file"])
        detections, _ = model.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                tp += 1
            elif detected_keyword in DECOY_KEYWORDS:
                fp += 1

    # 测试谐音负样本
    for sample in homophone_samples:
        samples, sr = read_wav(sample["file"])
        detections, _ = model.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword in DECOY_KEYWORDS:
                fa_decoy_intercepted += 1
            elif detected_keyword == TARGET_KEYWORD:
                fa_decoy_leak += 1

    # 测试真实负样本
    for sample in true_negative_samples:
        samples, sr = read_wav(sample["file"])
        detections, _ = model.detect(samples, sr)
        if detections:
            detected_keyword = detections[0].keyword
            if detected_keyword == TARGET_KEYWORD:
                fa_true += 1

    # 计算指标
    fn = len(positive_samples) - tp - fp
    total_positive = len(positive_samples)
    frr = (fn / total_positive * 100) if total_positive > 0 else 0

    total_decoy = fa_decoy_intercepted + fa_decoy_leak
    decoy_intercept_rate = (
        (fa_decoy_intercepted / total_decoy * 100) if total_decoy > 0 else 0
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


def differential_search(
    model_dir: str,
    raw_file: str,
    tokens_file: str,
    positive_samples: List[Dict],
    homophone_samples: List[Dict],
    true_negative_samples: List[Dict],
) -> List[Dict]:
    """差异化参数搜索"""
    target_boost_values = [2.2, 2.5, 2.8, 3.0]
    decoy_boost_values = [1.5, 1.8, 2.0]
    threshold = 0.40

    results = []
    total = len(target_boost_values) * len(decoy_boost_values)
    current = 0

    print(f"开始差异化参数搜索 (threshold={threshold})，共 {total} 种配置...\n")

    for target_boost in target_boost_values:
        for decoy_boost in decoy_boost_values:
            current += 1
            print(
                f"[{current}/{total}] target_boost={target_boost}, decoy_boost={decoy_boost}"
            )

            temp_kw = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
            temp_kw.close()

            try:
                success = generate_keywords_file(
                    raw_file,
                    tokens_file,
                    temp_kw.name,
                    target_boost,
                    threshold,
                    decoy_boost,
                    threshold,
                )

                if not success:
                    print("  ⚠️  生成失败，跳过\n")
                    continue

                metrics = test_config(
                    temp_kw.name,
                    model_dir,
                    positive_samples,
                    homophone_samples,
                    true_negative_samples,
                )

                config = {
                    "target_boost": target_boost,
                    "decoy_boost": decoy_boost,
                    "threshold": threshold,
                    **metrics,
                }
                results.append(config)

                print(
                    f"  FRR: {metrics['frr']:.2f}%, Intercept: {metrics['decoy_intercept_rate']:.1f}%, FA_true: {metrics['fa_true']}"
                )

                if (
                    metrics["frr"] <= 2.0
                    and metrics["decoy_intercept_rate"] >= 95.0
                    and metrics["fa_true"] == 0
                ):
                    print("  ✅ 满足目标！")
                print()

            finally:
                import os

                if os.path.exists(temp_kw.name):
                    os.unlink(temp_kw.name)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="简化版参数优化")
    parser.add_argument("--output", type=str, default="optimization_results.json")
    args = parser.parse_args()

    model_dir = str(
        _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    )
    raw_file = str(_ROOT_DIR / "test/detection/decoy_keywords_clean.txt")
    tokens_file = str(
        _ROOT_DIR
        / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt"
    )

    # 加载测试数据
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

    print(f"正样本: {len(positive_samples)}")
    print(f"谐音负样本: {len(homophone_samples)}")
    print(f"真实负样本: {len(true_negative_samples)}\n")

    # 执行差异化搜索
    results = differential_search(
        model_dir,
        raw_file,
        tokens_file,
        positive_samples,
        homophone_samples,
        true_negative_samples,
    )

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到: {args.output}\n")

    # 筛选满足目标的配置
    meeting_target = [
        r
        for r in results
        if r["frr"] <= 2.0 and r["decoy_intercept_rate"] >= 95.0 and r["fa_true"] == 0
    ]

    if meeting_target:
        print(f"✅ 找到 {len(meeting_target)} 个满足目标的配置：\n")
        for i, cfg in enumerate(meeting_target, 1):
            print(
                f"{i}. target_boost={cfg['target_boost']}, decoy_boost={cfg['decoy_boost']}, threshold={cfg['threshold']}"
            )
            print(
                f"   FRR: {cfg['frr']:.2f}%, Intercept: {cfg['decoy_intercept_rate']:.1f}%, FA_true: {cfg['fa_true']}\n"
            )
    else:
        print("⚠️  未找到完全满足目标的配置\n最接近的 Top 5：\n")
        sorted_results = sorted(
            results, key=lambda x: (x["frr"], -x["decoy_intercept_rate"], x["fa_true"])
        )
        for i, cfg in enumerate(sorted_results[:5], 1):
            print(
                f"{i}. target_boost={cfg['target_boost']}, decoy_boost={cfg['decoy_boost']}"
            )
            print(
                f"   FRR: {cfg['frr']:.2f}%, Intercept: {cfg['decoy_intercept_rate']:.1f}%, FA_true: {cfg['fa_true']}\n"
            )


if __name__ == "__main__":
    main()
