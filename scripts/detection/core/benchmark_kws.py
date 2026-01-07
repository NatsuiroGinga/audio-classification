#!/usr/bin/env python3
"""KWS（关键词唤醒）模型基准测试脚本。

评价指标：
1. FRR (False Rejection Rate) - 错误拒绝率/漏报率
2. FAR (False Alarm Rate) - 错误虚警率/误报率，以 FA/Hr 表示
3. RTF (Real-Time Factor) - 实时率

使用示例:
    python benchmark_kws.py \
        --model-dir ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
        --positive-dir /path/to/positive/wavs \
        --negative-dir /path/to/negative/wavs \
        --keyword "n ǐ h ǎo zh ēn zh ēn @你好真真"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 添加 src 目录到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KeywordDetection,
    KeywordSpotterModel,
    KWSModelConfig,
    create_kws_model,
)


@dataclass
class BenchmarkResult:
    """基准测试结果。

    Attributes:
        model_name: 模型名称
        keyword: 测试的唤醒词
        total_positive: 正样本总数
        true_positive: 正确识别的正样本数
        false_negative: 漏报数（正样本未识别）
        total_negative_hours: 负样本总时长（小时）
        false_positive: 误报数（负样本被错误触发）
        total_process_time: 总处理时间（秒）
        total_audio_duration: 总音频时长（秒）
        frr: 错误拒绝率 (%)
        fa_per_hour: 每小时误报数
        rtf: 实时率
    """

    model_name: str
    keyword: str
    total_positive: int = 0
    true_positive: int = 0
    false_negative: int = 0
    total_negative_hours: float = 0.0
    false_positive: int = 0
    total_process_time: float = 0.0
    total_audio_duration: float = 0.0
    positive_details: List[Dict[str, Any]] = field(default_factory=list)
    negative_details: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def frr(self) -> float:
        """错误拒绝率 (%)。"""
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100

    @property
    def fa_per_hour(self) -> float:
        """每小时误报数。"""
        if self.total_negative_hours == 0:
            return 0.0
        return self.false_positive / self.total_negative_hours

    @property
    def rtf(self) -> float:
        """实时率。"""
        if self.total_audio_duration == 0:
            return 0.0
        return self.total_process_time / self.total_audio_duration

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "model_name": self.model_name,
            "keyword": self.keyword,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "frr_percent": round(self.frr, 4),
            "total_negative_hours": round(self.total_negative_hours, 4),
            "false_positive": self.false_positive,
            "fa_per_hour": round(self.fa_per_hour, 4),
            "total_process_time_sec": round(self.total_process_time, 4),
            "total_audio_duration_sec": round(self.total_audio_duration, 4),
            "rtf": round(self.rtf, 6),
            "positive_details": self.positive_details,
            "negative_details": self.negative_details,
        }

    def summary(self) -> str:
        """生成摘要字符串。"""
        lines = [
            f"{'='*60}",
            f"模型: {self.model_name}",
            f"唤醒词: {self.keyword}",
            f"{'='*60}",
            f"",
            f"【正样本评估】",
            f"  总数: {self.total_positive}",
            f"  正确识别: {self.true_positive}",
            f"  漏报: {self.false_negative}",
            f"  FRR (错误拒绝率): {self.frr:.2f}%",
            f"  目标: FRR < 5% (SNR > 10dB)",
            f"",
            f"【负样本评估】",
            f"  总时长: {self.total_negative_hours:.4f} 小时",
            f"  误报次数: {self.false_positive}",
            f"  FA/Hr (每小时误报): {self.fa_per_hour:.4f}",
            f"  目标: < 1 FA/Hr (一般) 或 < 1 FA/24Hr (严格)",
            f"",
            f"【性能评估】",
            f"  总处理时间: {self.total_process_time:.4f} 秒",
            f"  总音频时长: {self.total_audio_duration:.4f} 秒",
            f"  RTF (实时率): {self.rtf:.6f}",
            f"  目标: RTF < 0.1 (边缘端), RTF < 1.0 (实时)",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def read_wave_file(wav_path: str) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件并返回 float32 采样和采样率。

    Args:
        wav_path: WAV 文件路径

    Returns:
        (samples, sample_rate): float32 音频数组和采样率
    """
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()

        data = wf.readframes(num_frames)

    # 根据采样宽度转换
    if sample_width == 2:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # 如果是多通道，取第一个通道
    if num_channels > 1:
        samples = samples[::num_channels]

    return samples, sample_rate


def resample_if_needed(
    samples: np.ndarray, src_sr: int, target_sr: int = 16000
) -> np.ndarray:
    """如有必要，重采样到目标采样率。

    Args:
        samples: 原始音频
        src_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频
    """
    if src_sr == target_sr:
        return samples

    # 使用简单的线性插值重采样
    duration = len(samples) / src_sr
    num_samples = int(duration * target_sr)
    indices = np.linspace(0, len(samples) - 1, num_samples)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def get_wav_duration(wav_path: str) -> float:
    """获取 WAV 文件时长（秒）。"""
    with wave.open(wav_path, "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def find_wav_files(directory: str) -> List[Path]:
    """递归查找目录下所有 WAV 文件。"""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    return sorted(dir_path.rglob("*.wav"))


def benchmark_positive(
    model: KeywordSpotterModel,
    positive_dir: str,
    keyword: str,
) -> Tuple[int, int, int, float, float, List[Dict[str, Any]]]:
    """对正样本进行基准测试。

    Args:
        model: KWS 模型
        positive_dir: 正样本目录
        keyword: 唤醒词

    Returns:
        (total, true_positive, false_negative, process_time, audio_duration, details)
    """
    wav_files = find_wav_files(positive_dir)
    total = len(wav_files)
    true_positive = 0
    false_negative = 0
    total_process_time = 0.0
    total_audio_duration = 0.0
    details: List[Dict[str, Any]] = []

    print(f"\n正样本测试 ({total} 个文件):")

    for i, wav_path in enumerate(wav_files):
        try:
            samples, sr = read_wave_file(str(wav_path))
            samples = resample_if_needed(samples, sr, 16000)
            audio_duration = len(samples) / 16000

            detections, process_time = model.detect(samples, 16000, keyword)

            detected = len(detections) > 0
            if detected:
                true_positive += 1
                status = "✓"
            else:
                false_negative += 1
                status = "✗"

            total_process_time += process_time
            total_audio_duration += audio_duration

            detail = {
                "file": str(wav_path),
                "detected": detected,
                "num_detections": len(detections),
                "audio_duration_sec": round(audio_duration, 4),
                "process_time_sec": round(process_time, 4),
                "rtf": (
                    round(process_time / audio_duration, 6) if audio_duration > 0 else 0
                ),
            }
            details.append(detail)

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(
                    f"  进度: {i + 1}/{total}, TP: {true_positive}, FN: {false_negative}"
                )

        except Exception as e:
            print(f"  跳过 {wav_path}: {e}")
            details.append({"file": str(wav_path), "error": str(e)})

    return (
        total,
        true_positive,
        false_negative,
        total_process_time,
        total_audio_duration,
        details,
    )


def benchmark_negative(
    model: KeywordSpotterModel,
    negative_dir: str,
    keyword: str,
) -> Tuple[float, int, float, float, List[Dict[str, Any]]]:
    """对负样本进行基准测试。

    Args:
        model: KWS 模型
        negative_dir: 负样本目录
        keyword: 唤醒词

    Returns:
        (total_hours, false_positive, process_time, audio_duration, details)
    """
    wav_files = find_wav_files(negative_dir)
    total_duration_sec = 0.0
    false_positive = 0
    total_process_time = 0.0
    details: List[Dict[str, Any]] = []

    print(f"\n负样本测试 ({len(wav_files)} 个文件):")

    for i, wav_path in enumerate(wav_files):
        try:
            samples, sr = read_wave_file(str(wav_path))
            samples = resample_if_needed(samples, sr, 16000)
            audio_duration = len(samples) / 16000

            detections, process_time = model.detect(samples, 16000, keyword)

            num_false_alarms = len(detections)
            false_positive += num_false_alarms

            total_duration_sec += audio_duration
            total_process_time += process_time

            detail = {
                "file": str(wav_path),
                "false_alarms": num_false_alarms,
                "audio_duration_sec": round(audio_duration, 4),
                "process_time_sec": round(process_time, 4),
            }
            if num_false_alarms > 0:
                detail["detections"] = [d.keyword for d in detections]
            details.append(detail)

            if (i + 1) % 10 == 0 or (i + 1) == len(wav_files):
                hours = total_duration_sec / 3600
                print(
                    f"  进度: {i + 1}/{len(wav_files)}, FA: {false_positive}, 累计时长: {hours:.4f}h"
                )

        except Exception as e:
            print(f"  跳过 {wav_path}: {e}")
            details.append({"file": str(wav_path), "error": str(e)})

    total_hours = total_duration_sec / 3600
    return total_hours, false_positive, total_process_time, total_duration_sec, details


def run_benchmark(
    model_dir: str,
    positive_dir: Optional[str],
    negative_dir: Optional[str],
    keyword: str,
    output_dir: str,
    use_int8: bool = False,
    provider: str = "cpu",
    num_threads: int = 2,
    keywords_threshold: float = 0.25,
) -> BenchmarkResult:
    """运行完整基准测试。

    Args:
        model_dir: 模型目录
        positive_dir: 正样本目录（可选）
        negative_dir: 负样本目录（可选）
        keyword: 唤醒词
        output_dir: 输出目录
        use_int8: 是否使用 int8 模型
        provider: 推理后端
        num_threads: 线程数
        keywords_threshold: 唤醒阈值

    Returns:
        BenchmarkResult 实例
    """
    model_name = Path(model_dir).name
    print(f"\n{'='*60}")
    print(f"KWS 基准测试")
    print(f"模型: {model_name}")
    print(f"唤醒词: {keyword}")
    print(f"INT8: {use_int8}, Provider: {provider}, Threads: {num_threads}")
    print(f"阈值: {keywords_threshold}")
    print(f"{'='*60}")

    # 创建模型
    model = create_kws_model(
        model_dir=model_dir,
        keywords=keyword,
        provider=provider,
        num_threads=num_threads,
        keywords_threshold=keywords_threshold,
        use_int8=use_int8,
    )

    result = BenchmarkResult(model_name=model_name, keyword=keyword)

    # 正样本测试
    if positive_dir and Path(positive_dir).exists():
        (
            total,
            tp,
            fn,
            proc_time,
            audio_dur,
            details,
        ) = benchmark_positive(model, positive_dir, keyword)
        result.total_positive = total
        result.true_positive = tp
        result.false_negative = fn
        result.total_process_time += proc_time
        result.total_audio_duration += audio_dur
        result.positive_details = details

    # 负样本测试
    if negative_dir and Path(negative_dir).exists():
        hours, fp, proc_time, audio_dur, details = benchmark_negative(
            model, negative_dir, keyword
        )
        result.total_negative_hours = hours
        result.false_positive = fp
        result.total_process_time += proc_time
        result.total_audio_duration += audio_dur
        result.negative_details = details

    # 输出结果
    print(result.summary())

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_path / f"benchmark_{model_name}_{timestamp}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {result_file}")

    return result


def compare_models(
    model_dirs: List[str],
    positive_dir: Optional[str],
    negative_dir: Optional[str],
    keyword: str,
    output_dir: str,
    **kwargs: Any,
) -> List[BenchmarkResult]:
    """对比多个模型的性能。

    Args:
        model_dirs: 模型目录列表
        positive_dir: 正样本目录
        negative_dir: 负样本目录
        keyword: 唤醒词
        output_dir: 输出目录
        **kwargs: 其他参数

    Returns:
        各模型的测试结果列表
    """
    results: List[BenchmarkResult] = []

    for model_dir in model_dirs:
        result = run_benchmark(
            model_dir=model_dir,
            positive_dir=positive_dir,
            negative_dir=negative_dir,
            keyword=keyword,
            output_dir=output_dir,
            **kwargs,
        )
        results.append(result)

    # 生成对比报告
    print("\n" + "=" * 80)
    print("模型对比结果")
    print("=" * 80)

    headers = ["模型", "FRR(%)", "FA/Hr", "RTF"]
    col_widths = [40, 10, 10, 12]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for r in results:
        row = [
            r.model_name[:40],
            f"{r.frr:.2f}",
            f"{r.fa_per_hour:.4f}",
            f"{r.rtf:.6f}",
        ]
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, col_widths)))

    # 保存对比结果
    output_path = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = output_path / f"comparison_{timestamp}.json"

    comparison_data = {
        "keyword": keyword,
        "timestamp": timestamp,
        "results": [r.to_dict() for r in results],
    }

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)

    print(f"\n对比结果已保存到: {comparison_file}")

    return results


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="KWS 模型基准测试",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型配置
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="模型目录路径，或逗号分隔的多个模型目录（用于对比）",
    )

    # 测试数据
    parser.add_argument(
        "--positive-dir",
        type=str,
        default="",
        help="正样本目录（包含唤醒词的音频）",
    )
    parser.add_argument(
        "--negative-dir",
        type=str,
        default="",
        help="负样本目录（不包含唤醒词的音频）",
    )

    # 唤醒词配置
    parser.add_argument(
        "--keyword",
        type=str,
        default=KEYWORD_NIHAO_ZHENZHEN,
        help="唤醒词（拼音格式，如 'n ǐ h ǎo zh ēn zh ēn @你好真真'）",
    )

    # 模型参数
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="使用 INT8 量化模型",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理后端",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=2,
        help="推理线程数",
    )
    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="唤醒词触发阈值（越大越难触发）",
    )

    # 输出配置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="结果输出目录",
    )

    return parser.parse_args()


def main() -> None:
    """主入口。"""
    args = parse_args()

    # 支持多模型对比
    model_dirs = [d.strip() for d in args.model_dir.split(",")]

    if len(model_dirs) == 1:
        run_benchmark(
            model_dir=model_dirs[0],
            positive_dir=args.positive_dir or None,
            negative_dir=args.negative_dir or None,
            keyword=args.keyword,
            output_dir=args.output_dir,
            use_int8=args.use_int8,
            provider=args.provider,
            num_threads=args.num_threads,
            keywords_threshold=args.keywords_threshold,
        )
    else:
        compare_models(
            model_dirs=model_dirs,
            positive_dir=args.positive_dir or None,
            negative_dir=args.negative_dir or None,
            keyword=args.keyword,
            output_dir=args.output_dir,
            use_int8=args.use_int8,
            provider=args.provider,
            num_threads=args.num_threads,
            keywords_threshold=args.keywords_threshold,
        )


if __name__ == "__main__":
    main()
