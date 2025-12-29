#!/usr/bin/env python3
"""流式模拟测试：对比 detect vs detect_streaming 的指标差异。

测试场景：
1. 离线批处理模式（detect）
2. 流式模拟模式（detect_streaming with simulated delay）

对比指标：
- FRR (False Rejection Rate)
- FA (False Alarm)
- RTF (Real-Time Factor)
- 处理延迟差异
"""

from __future__ import annotations

import json
import sys
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np

# 添加 src 到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KeywordDetection,
    create_kws_model,
)


@dataclass
class TestResult:
    """单个方法的测试结果。"""

    method_name: str
    total_positive: int = 0
    true_positive: int = 0
    false_negative: int = 0
    total_negative_normal: int = 0
    false_positive_normal: int = 0
    total_negative_homophone: int = 0
    false_positive_homophone: int = 0
    total_process_time: float = 0.0
    total_audio_duration: float = 0.0
    detection_latencies: List[float] = None  # 每次检测的延迟

    def __post_init__(self):
        if self.detection_latencies is None:
            self.detection_latencies = []

    @property
    def frr(self) -> float:
        """错误拒绝率 (%)。"""
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100

    @property
    def rtf(self) -> float:
        """实时率。"""
        if self.total_audio_duration == 0:
            return 0.0
        return self.total_process_time / self.total_audio_duration

    @property
    def avg_latency(self) -> float:
        """平均检测延迟（毫秒）。"""
        if not self.detection_latencies:
            return 0.0
        return sum(self.detection_latencies) / len(self.detection_latencies) * 1000


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def audio_chunk_iterator(
    samples: np.ndarray, sr: int, chunk_ms: int = 50, simulate_delay: bool = True
) -> Iterator[np.ndarray]:
    """音频块迭代器，模拟实时流式场景。

    Args:
        samples: 音频数组
        sr: 采样率
        chunk_ms: 每个 chunk 的毫秒数
        simulate_delay: 是否模拟实时采集延迟
    """
    chunk_samples = int(sr * chunk_ms / 1000)
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i : i + chunk_samples]
        if simulate_delay:
            time.sleep(chunk_ms / 1000)  # 模拟实时采集延迟
        yield chunk


def test_with_detect(
    model,
    samples: np.ndarray,
    sr: int,
    chunk_ms: int = 100,
) -> tuple[List[KeywordDetection], float]:
    """使用 detect 方法测试（离线批处理）。"""
    start_time = time.perf_counter()
    detections, _ = model.detect(samples, sr, chunk_size_ms=chunk_ms)
    process_time = time.perf_counter() - start_time
    return detections, process_time


def test_with_detect_streaming(
    model,
    samples: np.ndarray,
    sr: int,
    chunk_ms: int = 50,
    simulate_delay: bool = True,
) -> tuple[List[KeywordDetection], float]:
    """使用 detect_streaming 方法测试（流式模拟）。"""
    start_time = time.perf_counter()
    chunks = audio_chunk_iterator(samples, sr, chunk_ms, simulate_delay)
    detections = list(model.detect_streaming(chunks, sr))
    process_time = time.perf_counter() - start_time
    return detections, process_time


def run_comparison_test(
    model,
    positive_samples: List[Dict[str, Any]],
    negative_normal: List[Dict[str, Any]],
    negative_homophone: List[Dict[str, Any]],
    chunk_ms: int = 50,
    simulate_delay: bool = True,
) -> tuple[TestResult, TestResult]:
    """运行对比测试。

    Returns:
        (detect_result, detect_streaming_result)
    """
    # 方法 1: detect (离线批处理)
    print("\n测试方法 1: detect (离线批处理)...")
    result_detect = TestResult(method_name="detect (offline)")

    # 正样本
    result_detect.total_positive = len(positive_samples)
    for sample in positive_samples:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect(model, samples, sr, chunk_ms=100)
        result_detect.total_process_time += proc_time
        result_detect.total_audio_duration += len(samples) / sr

        if detections:
            result_detect.true_positive += 1
            result_detect.detection_latencies.append(detections[0].start_time)
        else:
            result_detect.false_negative += 1

    # 负样本（正常）
    result_detect.total_negative_normal = len(negative_normal)
    for sample in negative_normal:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect(model, samples, sr)
        result_detect.total_process_time += proc_time
        result_detect.total_audio_duration += len(samples) / sr

        if detections:
            result_detect.false_positive_normal += 1

    # 负样本（谐音）
    result_detect.total_negative_homophone = len(negative_homophone)
    for sample in negative_homophone:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect(model, samples, sr)
        result_detect.total_process_time += proc_time
        result_detect.total_audio_duration += len(samples) / sr

        if detections:
            result_detect.false_positive_homophone += 1

    print(
        f"  FRR={result_detect.frr:.2f}%, FA_normal={result_detect.false_positive_normal}, "
        f"FA_homophone={result_detect.false_positive_homophone}, RTF={result_detect.rtf:.4f}"
    )

    # 方法 2: detect_streaming (流式模拟)
    print(f"\n测试方法 2: detect_streaming (流式模拟, chunk={chunk_ms}ms)...")
    result_streaming = TestResult(method_name="detect_streaming (simulated)")

    # 正样本
    result_streaming.total_positive = len(positive_samples)
    for sample in positive_samples:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect_streaming(
            model, samples, sr, chunk_ms, simulate_delay
        )
        result_streaming.total_process_time += proc_time
        result_streaming.total_audio_duration += len(samples) / sr

        if detections:
            result_streaming.true_positive += 1
            result_streaming.detection_latencies.append(detections[0].start_time)
        else:
            result_streaming.false_negative += 1

    # 负样本（正常）
    result_streaming.total_negative_normal = len(negative_normal)
    for sample in negative_normal:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect_streaming(
            model, samples, sr, chunk_ms, simulate_delay
        )
        result_streaming.total_process_time += proc_time
        result_streaming.total_audio_duration += len(samples) / sr

        if detections:
            result_streaming.false_positive_normal += 1

    # 负样本（谐音）
    result_streaming.total_negative_homophone = len(negative_homophone)
    for sample in negative_homophone:
        samples, sr = read_wav(sample["file"])
        detections, proc_time = test_with_detect_streaming(
            model, samples, sr, chunk_ms, simulate_delay
        )
        result_streaming.total_process_time += proc_time
        result_streaming.total_audio_duration += len(samples) / sr

        if detections:
            result_streaming.false_positive_homophone += 1

    print(
        f"  FRR={result_streaming.frr:.2f}%, FA_normal={result_streaming.false_positive_normal}, "
        f"FA_homophone={result_streaming.false_positive_homophone}, RTF={result_streaming.rtf:.4f}"
    )

    return result_detect, result_streaming


def print_comparison_report(
    result_detect: TestResult, result_streaming: TestResult, chunk_ms: int
):
    """打印详细对比报告。"""
    print("\n" + "=" * 100)
    print("流式模拟测试对比报告")
    print("=" * 100)

    print(f"\n【测试配置】")
    print(f"  流式 chunk 大小: {chunk_ms} ms")
    print(f"  离线 chunk 大小: 100 ms")
    print(f"  模拟实时延迟: 是")

    print(f"\n【准确率指标对比】")
    print(f"  {'指标':<20} | {'离线批处理':<15} | {'流式模拟':<15} | {'差异':<15}")
    print("-" * 70)

    # FRR
    frr_diff = result_streaming.frr - result_detect.frr
    print(
        f"  {'FRR (%)':<20} | {result_detect.frr:<15.2f} | {result_streaming.frr:<15.2f} | "
        f"{frr_diff:+.2f} pp"
    )

    # FA_normal
    fa_normal_diff = (
        result_streaming.false_positive_normal - result_detect.false_positive_normal
    )
    print(
        f"  {'FA_正常':<20} | {result_detect.false_positive_normal:<15} | "
        f"{result_streaming.false_positive_normal:<15} | {fa_normal_diff:+d}"
    )

    # FA_homophone
    fa_homo_diff = (
        result_streaming.false_positive_homophone
        - result_detect.false_positive_homophone
    )
    print(
        f"  {'FA_谐音':<20} | {result_detect.false_positive_homophone:<15} | "
        f"{result_streaming.false_positive_homophone:<15} | {fa_homo_diff:+d}"
    )

    print(f"\n【性能指标对比】")
    print(f"  {'指标':<20} | {'离线批处理':<15} | {'流式模拟':<15} | {'差异':<15}")
    print("-" * 70)

    # RTF
    rtf_diff_pct = (
        (result_streaming.rtf / result_detect.rtf - 1) * 100
        if result_detect.rtf > 0
        else 0
    )
    print(
        f"  {'RTF':<20} | {result_detect.rtf:<15.4f} | {result_streaming.rtf:<15.4f} | "
        f"{rtf_diff_pct:+.1f}%"
    )

    # 总处理时间
    time_diff_pct = (
        (result_streaming.total_process_time / result_detect.total_process_time - 1)
        * 100
        if result_detect.total_process_time > 0
        else 0
    )
    print(
        f"  {'总处理时间 (s)':<20} | {result_detect.total_process_time:<15.2f} | "
        f"{result_streaming.total_process_time:<15.2f} | {time_diff_pct:+.1f}%"
    )

    # 平均检测延迟
    latency_diff = result_streaming.avg_latency - result_detect.avg_latency
    print(
        f"  {'平均检测延迟 (ms)':<20} | {result_detect.avg_latency:<15.1f} | "
        f"{result_streaming.avg_latency:<15.1f} | {latency_diff:+.1f}"
    )

    print(f"\n【结论分析】")
    if abs(frr_diff) < 1.0:
        print(f"  ✅ FRR 差异 < 1%，准确率基本一致")
    else:
        print(f"  ⚠️  FRR 差异 {abs(frr_diff):.1f}%，需关注")

    if abs(rtf_diff_pct) < 50:
        print(f"  ✅ RTF 差异 < 50%，性能影响可控")
    else:
        print(f"  ⚠️  RTF 差异 {abs(rtf_diff_pct):.1f}%，流式开销较大")

    if fa_normal_diff == 0:
        print(f"  ✅ 正常负样本误报数一致")
    else:
        print(f"  ⚠️  正常负样本误报数差异: {fa_normal_diff:+d}")

    print("=" * 100)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="流式模拟测试对比")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_ROOT_DIR / "dataset/kws_test_data"),
        help="测试数据目录",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(
            _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        ),
        help="KWS 模型目录",
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="使用 int8 量化模型",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=50,
        help="流式模拟的 chunk 大小（毫秒）",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="不模拟实时延迟（仅测试检测逻辑差异）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT_DIR / "test/detection"),
        help="输出目录",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大测试样本数（用于快速测试）",
    )

    args = parser.parse_args()

    # 加载测试数据
    metadata_path = Path(args.data_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"错误: 找不到测试数据 {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]
    positive_samples = meta["positive_samples"]
    negative_normal = [
        s for s in meta["negative_samples"] if s.get("phrase") not in homophones
    ]
    negative_homophone = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]

    # 限制样本数（用于快速测试）
    if args.max_samples:
        positive_samples = positive_samples[: args.max_samples]
        negative_normal = negative_normal[: args.max_samples]
        negative_homophone = negative_homophone[
            : min(args.max_samples // 4, len(negative_homophone))
        ]

    print("=" * 100)
    print("流式模拟测试：detect vs detect_streaming")
    print("=" * 100)
    print(f"模型目录: {args.model_dir}")
    print(f"量化: {'int8' if args.use_int8 else 'fp32'}")
    print(f"流式 chunk 大小: {args.chunk_ms} ms")
    print(f"模拟实时延迟: {'否' if args.no_delay else '是'}")
    print(f"正样本数: {len(positive_samples)}")
    print(f"正常负样本数: {len(negative_normal)}")
    print(f"谐音负样本数: {len(negative_homophone)}")
    print()

    # 创建模型
    print("加载模型...")
    model = create_kws_model(
        args.model_dir,
        keywords=KEYWORD_NIHAO_ZHENZHEN,
        use_int8=args.use_int8,
        provider="cpu",
        num_threads=2,
    )

    # 运行对比测试
    result_detect, result_streaming = run_comparison_test(
        model=model,
        positive_samples=positive_samples,
        negative_normal=negative_normal,
        negative_homophone=negative_homophone,
        chunk_ms=args.chunk_ms,
        simulate_delay=not args.no_delay,
    )

    # 打印对比报告
    print_comparison_report(result_detect, result_streaming, args.chunk_ms)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"streaming_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model_dir": args.model_dir,
            "use_int8": args.use_int8,
            "chunk_ms": args.chunk_ms,
            "simulate_delay": not args.no_delay,
        },
        "detect_offline": {
            "frr": result_detect.frr,
            "fa_normal": result_detect.false_positive_normal,
            "fa_homophone": result_detect.false_positive_homophone,
            "rtf": result_detect.rtf,
            "total_process_time": result_detect.total_process_time,
            "avg_latency_ms": result_detect.avg_latency,
        },
        "detect_streaming": {
            "frr": result_streaming.frr,
            "fa_normal": result_streaming.false_positive_normal,
            "fa_homophone": result_streaming.false_positive_homophone,
            "rtf": result_streaming.rtf,
            "total_process_time": result_streaming.total_process_time,
            "avg_latency_ms": result_streaming.avg_latency,
        },
        "differences": {
            "frr_diff_pp": result_streaming.frr - result_detect.frr,
            "fa_normal_diff": result_streaming.false_positive_normal
            - result_detect.false_positive_normal,
            "fa_homophone_diff": result_streaming.false_positive_homophone
            - result_detect.false_positive_homophone,
            "rtf_diff_pct": (
                (result_streaming.rtf / result_detect.rtf - 1) * 100
                if result_detect.rtf > 0
                else 0
            ),
            "time_diff_pct": (
                (
                    result_streaming.total_process_time
                    / result_detect.total_process_time
                    - 1
                )
                * 100
                if result_detect.total_process_time > 0
                else 0
            ),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
