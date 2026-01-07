#!/usr/bin/env python3
"""验证脚本 - 使用 keywords.txt 中的 threshold 值（不覆盖）"""

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


def main():
    model_dir = str(
        _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    )
    keywords_file = str(_ROOT_DIR / "test/detection/decoy_keywords.txt")

    # 关键：使用 keywords_threshold=0 让文件中的阈值生效
    kws = create_kws_model(
        model_dir,
        keywords_file=keywords_file,
        use_int8=True,
        epoch=12,
        avg=2,
        keywords_threshold=0,  # 不覆盖文件中的 threshold
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

    # 测试
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
    frr = fn / len(positive_samples) * 100

    total_homophone_detections = fa_decoy_intercepted + fa_decoy_leak
    decoy_intercept_rate = (
        (fa_decoy_intercepted / total_homophone_detections * 100)
        if total_homophone_detections > 0
        else 0
    )

    print("=" * 70)
    print("验证结果（使用 keywords.txt 中的 threshold）")
    print("=" * 70)
    print(f"正样本 (Total: {len(positive_samples)})")
    print(f"  ✓ 正确识别 (TP): {tp}")
    print(f"  ✗ 误判为诱导词 (FP): {fp}")
    print(f"  ✗ 漏检 (FN): {fn}")
    print(f"  漏报率 (FRR): {frr:.2f}%")
    print()
    print(f"谐音负样本 (Total: {len(homophone_samples)})")
    print(f"  ✓ 被诱导词拦截: {fa_decoy_intercepted}")
    print(f"  ✗ 漏过拦截: {fa_decoy_leak}")
    print(f"  诱导词拦截率: {decoy_intercept_rate:.1f}%")
    print()
    print(f"真实负样本 (Total: {len(true_negative_samples)})")
    print(f"  ✗ 误报 (FA_true): {fa_true}")
    print(f"  误报率: {fa_true / len(true_negative_samples) * 100:.2f}%")


if __name__ == "__main__":
    main()
