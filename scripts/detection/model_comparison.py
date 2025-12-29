#!/usr/bin/env python3
"""KWS 模型对比测试脚本（流式模式）。

测试多个模型在不同量化配置下的性能：
1. sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
2. sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20

测试指标：
- FRR (False Rejection Rate) - 漏报率
- FA_normal - 正常负样本误报数
- FA_homophone - 谐音负样本误报数
- RTF (Real-Time Factor) - 实时率
- CPU负载 - 推理时CPU使用率
- 内存占用 - 推理时内存使用

输出：对比结果表格和 JSON 报告
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# 资源监控
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil 未安装，无法监控 CPU/内存。安装: pip install psutil")

# 添加 src 到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KeywordSpotterModel,
    KWSModelConfig,
    create_kws_model,
)


# ============================================================================
# 资源监控器
# ============================================================================


class ResourceMonitor:
    """CPU 和内存资源监控器。"""

    def __init__(self, interval: float = 0.1, num_cores: int = 1) -> None:
        self.interval = interval
        self.num_cores = num_cores
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cpu_samples: List[float] = []
        self._mem_samples: List[float] = []
        self._process: Optional[Any] = None

    def start(self) -> None:
        """开始监控。"""
        if not PSUTIL_AVAILABLE:
            return
        self._running = True
        self._cpu_samples = []
        self._mem_samples = []
        self._process = psutil.Process()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Tuple[float, float, float, float, float, float]:
        """停止监控并返回统计结果。

        Returns:
            (avg_cpu, max_cpu, avg_mem_mb, max_mem_mb, avg_cpu_per_core, max_cpu_per_core)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        if not self._cpu_samples:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        avg_cpu = sum(self._cpu_samples) / len(self._cpu_samples)
        max_cpu = max(self._cpu_samples)
        avg_mem = sum(self._mem_samples) / len(self._mem_samples)
        max_mem = max(self._mem_samples)
        # 计算单核百分比
        avg_cpu_per_core = avg_cpu / self.num_cores if self.num_cores > 0 else 0.0
        max_cpu_per_core = max_cpu / self.num_cores if self.num_cores > 0 else 0.0

        return avg_cpu, max_cpu, avg_mem, max_mem, avg_cpu_per_core, max_cpu_per_core

    def _monitor_loop(self) -> None:
        """监控循环。"""
        while self._running:
            try:
                # CPU 使用率（百分比）
                cpu = self._process.cpu_percent(interval=None)
                self._cpu_samples.append(cpu)

                # 内存使用（MB）
                mem_info = self._process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                self._mem_samples.append(mem_mb)
            except Exception:
                pass
            time.sleep(self.interval)


# ============================================================================
# 测试配置
# ============================================================================


@dataclass
class ModelTestConfig:
    """模型测试配置。"""

    model_dir: str  # 模型目录
    model_name: str  # 模型名称（显示用）
    use_int8: bool = False  # 是否使用 int8 量化
    epoch: int = 12  # epoch 编号
    avg: int = 2  # 平均数
    chunk: int = 16  # chunk 大小
    left: int = 64  # left 大小


# 流式测试配置
@dataclass
class StreamingConfig:
    """流式测试配置。"""

    chunk_ms: int = 50  # 每个 chunk 的毫秒数
    simulate_delay: bool = True  # 是否模拟实时延迟


@dataclass
class TestResult:
    """测试结果。"""

    model_name: str
    use_int8: bool
    total_positive: int = 0
    true_positive: int = 0
    false_negative: int = 0
    total_negative_normal: int = 0
    false_positive_normal: int = 0
    total_negative_homophone: int = 0
    false_positive_homophone: int = 0
    total_process_time: float = 0.0
    total_audio_duration: float = 0.0
    # 资源指标（多核 CPU %）
    avg_cpu: float = 0.0
    max_cpu: float = 0.0
    # 资源指标（单核 CPU %）
    avg_cpu_per_core: float = 0.0
    max_cpu_per_core: float = 0.0
    # 内存指标
    avg_mem_mb: float = 0.0
    max_mem_mb: float = 0.0
    # 模型信息
    model_size_mb: float = 0.0

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "use_int8": self.use_int8,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "frr_percent": round(self.frr, 2),
            "total_negative_normal": self.total_negative_normal,
            "fa_normal": self.false_positive_normal,
            "total_negative_homophone": self.total_negative_homophone,
            "fa_homophone": self.false_positive_homophone,
            "rtf": round(self.rtf, 4),
            "avg_cpu_percent_total": round(self.avg_cpu, 1),
            "max_cpu_percent_total": round(self.max_cpu, 1),
            "avg_cpu_percent_per_core": round(self.avg_cpu_per_core, 1),
            "max_cpu_percent_per_core": round(self.max_cpu_per_core, 1),
            "avg_mem_mb": round(self.avg_mem_mb, 1),
            "max_mem_mb": round(self.max_mem_mb, 1),
            "model_size_mb": round(self.model_size_mb, 2),
        }


# ============================================================================
# 工具函数
# ============================================================================


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """加载测试数据元信息。"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_size_mb(model_dir: str, use_int8: bool) -> float:
    """计算模型文件大小（MB）。"""
    model_path = Path(model_dir)
    total_size = 0

    suffix = ".int8.onnx" if use_int8 else ".onnx"

    for onnx_file in model_path.glob(f"*{suffix}"):
        # 排除非 int8 文件（当查找 int8 时）或 int8 文件（当查找非 int8 时）
        if use_int8 and ".int8.onnx" not in onnx_file.name:
            continue
        if not use_int8 and ".int8.onnx" in onnx_file.name:
            continue
        total_size += onnx_file.stat().st_size

    return total_size / (1024 * 1024)


def create_model_with_config(
    config: ModelTestConfig, keyword: str
) -> KeywordSpotterModel:
    """根据配置创建 KWS 模型。"""
    return create_kws_model(
        config.model_dir,
        keywords=keyword,
        use_int8=config.use_int8,
        epoch=config.epoch,
        avg=config.avg,
        chunk=config.chunk,
        left=config.left,
    )


def audio_chunk_iterator(
    samples: np.ndarray,
    sr: int,
    chunk_ms: int = 50,
    simulate_delay: bool = True,
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


# ============================================================================
# 测试主逻辑
# ============================================================================


def run_model_test(
    config: ModelTestConfig,
    positive_samples: List[Dict[str, Any]],
    negative_normal_samples: List[Dict[str, Any]],
    negative_homophone_samples: List[Dict[str, Any]],
    num_cores: int = 1,
    streaming_config: Optional[StreamingConfig] = None,
    keyword: str = KEYWORD_NIHAO_ZHENZHEN,
    keywords_threshold: float = 0.25,
) -> TestResult:
    """运行单个模型配置的测试（流式模式）。"""

    if streaming_config is None:
        streaming_config = StreamingConfig()

    quant_str = "int8" if config.use_int8 else "fp32"
    result = TestResult(
        model_name=config.model_name,
        use_int8=config.use_int8,
    )

    # 计算模型大小
    result.model_size_mb = get_model_size_mb(config.model_dir, config.use_int8)

    # 强制 GC 以获得更准确的内存测量
    gc.collect()

    # 初始化资源监控器
    monitor = ResourceMonitor(interval=0.05, num_cores=num_cores)

    # 创建模型
    try:
        # 使用指定的阈值创建模型
        kws_model = create_kws_model(
            config.model_dir,
            keywords=keyword,
            use_int8=config.use_int8,
            epoch=config.epoch,
            avg=config.avg,
            chunk=config.chunk,
            left=config.left,
            keywords_threshold=keywords_threshold,
        )
    except Exception as e:
        print(f"  模型创建失败: {e}")
        return result

    # 开始监控
    monitor.start()

    # 1. 测试正样本（流式）
    result.total_positive = len(positive_samples)
    for sample in positive_samples:
        samples, sr = read_wav(sample["file"])
        audio_duration = len(samples) / sr
        result.total_audio_duration += audio_duration

        start_time = time.perf_counter()
        chunks = audio_chunk_iterator(
            samples, sr, streaming_config.chunk_ms, streaming_config.simulate_delay
        )
        detections = list(kws_model.detect_streaming(chunks, sr))
        result.total_process_time += time.perf_counter() - start_time

        if detections:
            result.true_positive += 1
        else:
            result.false_negative += 1

    # 2. 测试正常负样本（流式）
    result.total_negative_normal = len(negative_normal_samples)
    for sample in negative_normal_samples:
        samples, sr = read_wav(sample["file"])
        audio_duration = len(samples) / sr
        result.total_audio_duration += audio_duration

        start_time = time.perf_counter()
        chunks = audio_chunk_iterator(
            samples, sr, streaming_config.chunk_ms, streaming_config.simulate_delay
        )
        detections = list(kws_model.detect_streaming(chunks, sr))
        result.total_process_time += time.perf_counter() - start_time

        if detections:
            result.false_positive_normal += 1

    # 3. 测试谐音负样本（流式）
    result.total_negative_homophone = len(negative_homophone_samples)
    for sample in negative_homophone_samples:
        samples, sr = read_wav(sample["file"])
        audio_duration = len(samples) / sr
        result.total_audio_duration += audio_duration

        start_time = time.perf_counter()
        chunks = audio_chunk_iterator(
            samples, sr, streaming_config.chunk_ms, streaming_config.simulate_delay
        )
        detections = list(kws_model.detect_streaming(chunks, sr))
        result.total_process_time += time.perf_counter() - start_time

        if detections:
            result.false_positive_homophone += 1

    # 停止监控并获取资源使用统计
    avg_cpu, max_cpu, avg_mem, max_mem, avg_cpu_per_core, max_cpu_per_core = (
        monitor.stop()
    )
    result.avg_cpu = avg_cpu
    result.max_cpu = max_cpu
    result.avg_cpu_per_core = avg_cpu_per_core
    result.max_cpu_per_core = max_cpu_per_core
    result.avg_mem_mb = avg_mem
    result.max_mem_mb = max_mem

    # 释放模型
    del kws_model
    gc.collect()

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KWS 模型对比测试")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_ROOT_DIR / "dataset/kws_test_data"),
        help="测试数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT_DIR / "test/detection"),
        help="输出目录",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(_ROOT_DIR / "models"),
        help="模型根目录",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=50,
        help="流式测试的 chunk 大小（毫秒）",
    )
    parser.add_argument(
        "--no-delay",
        action="store_true",
        help="不模拟实时延迟（仅测试检测逻辑）",
    )
    parser.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.25,
        help="全局唤醒阈值（越大越难触发）",
    )
    parser.add_argument(
        "--per-word-threshold",
        type=float,
        default=None,
        help="逐词阈值（仅对目标关键词生效），例如 0.40",
    )

    args = parser.parse_args()

    # 加载测试数据
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"错误: 找不到测试数据 {metadata_path}")
        sys.exit(1)

    meta = load_metadata(metadata_path)

    # 谐音短语列表
    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]

    positive_samples = meta["positive_samples"]
    negative_normal = [
        s for s in meta["negative_samples"] if s.get("phrase") not in homophones
    ]
    negative_homophone = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]

    # 获取 CPU 核心数
    num_cores = psutil.cpu_count(logical=True) if PSUTIL_AVAILABLE else 1

    # 流式测试配置
    streaming_config = StreamingConfig(
        chunk_ms=args.chunk_ms,
        simulate_delay=not args.no_delay,
    )

    print("=" * 80)
    print("KWS 模型对比测试（流式模式）")
    print("=" * 80)
    print(f"CPU 核心数: {num_cores}")
    print(f"流式 chunk 大小: {streaming_config.chunk_ms} ms")
    print(f"模拟实时延迟: {'否' if args.no_delay else '是'}")
    print(f"正样本: {len(positive_samples)}")
    print(f"正常负样本: {len(negative_normal)}")
    print(f"谐音负样本: {len(negative_homophone)}")
    print(f"psutil 可用: {PSUTIL_AVAILABLE}")
    print(f"全局阈值: {args.keywords_threshold}")
    if args.per_word_threshold is not None:
        print(f"逐词阈值: {args.per_word_threshold}（针对 你好真真）")
    print()

    # 定义测试配置
    models_dir = Path(args.models_dir)

    test_configs = [
        # wenetspeech-3.3M 模型
        ModelTestConfig(
            model_dir=str(
                models_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
            ),
            model_name="wenetspeech-3.3M",
            use_int8=False,
            epoch=12,
            avg=2,
        ),
        ModelTestConfig(
            model_dir=str(
                models_dir / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
            ),
            model_name="wenetspeech-3.3M",
            use_int8=True,
            epoch=12,
            avg=2,
        ),
        # zh-en-3M 模型
        ModelTestConfig(
            model_dir=str(models_dir / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"),
            model_name="zh-en-3M",
            use_int8=False,
            epoch=13,
            avg=2,
        ),
        ModelTestConfig(
            model_dir=str(models_dir / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"),
            model_name="zh-en-3M",
            use_int8=True,
            epoch=13,
            avg=2,
        ),
    ]

    results: List[TestResult] = []

    # 构造关键词字符串（逐词阈值：在关键词行加入 #threshold）
    keyword_line = KEYWORD_NIHAO_ZHENZHEN
    if args.per_word_threshold is not None:
        # 将 '@' 前插入 '#阈值'
        # 例如: "... zh ēn zh ēn #0.40 @你好真真"
        parts = keyword_line.split("@")
        if len(parts) == 2:
            keyword_line = (
                parts[0].strip() + f" #{args.per_word_threshold} @" + parts[1].strip()
            )

    for config in test_configs:
        quant_str = "int8" if config.use_int8 else "fp32"
        print(f"测试: {config.model_name} ({quant_str})...")

        # 检查模型目录是否存在
        if not os.path.exists(config.model_dir):
            print(f"  跳过: 模型目录不存在 {config.model_dir}")
            continue

        result = run_model_test(
            config=config,
            positive_samples=positive_samples,
            negative_normal_samples=negative_normal,
            negative_homophone_samples=negative_homophone,
            num_cores=num_cores,
            streaming_config=streaming_config,
            keyword=keyword_line,
            keywords_threshold=args.keywords_threshold,
        )
        results.append(result)

        print(
            f"  FRR={result.frr:.1f}%, FA_normal={result.false_positive_normal}, "
            f"FA_homophone={result.false_positive_homophone}, RTF={result.rtf:.4f}"
        )
        print(
            f"  CPU (多核): avg={result.avg_cpu:.1f}%, max={result.max_cpu:.1f}% | "
            f"CPU (单核): avg={result.avg_cpu_per_core:.1f}%, max={result.max_cpu_per_core:.1f}%"
        )
        print(f"  内存: avg={result.avg_mem_mb:.1f}MB, max={result.max_mem_mb:.1f}MB")
        print(f"  模型大小: {result.model_size_mb:.2f}MB")
        print()

    # 输出结果表格
    print("=" * 130)
    print("模型对比结果汇总（流式模式）")
    print("=" * 130)
    header = (
        f"{'模型':<20} | {'量化':<5} | {'FRR(%)':<7} | {'FA_正常':<7} | {'FA_谐音':<7} | "
        f"{'RTF':<8} | {'单核CPU(%)':<12} | {'多核CPU(%)':<12} | {'内存(MB)':<9} | {'大小(MB)':<9}"
    )
    print(header)
    print("-" * 130)

    for r in results:
        quant = "int8" if r.use_int8 else "fp32"
        row = (
            f"{r.model_name:<20} | {quant:<5} | {r.frr:<7.1f} | {r.false_positive_normal:<7} | "
            f"{r.false_positive_homophone:<7} | {r.rtf:<8.4f} | {r.avg_cpu_per_core:<12.1f} | "
            f"{r.avg_cpu:<12.1f} | {r.avg_mem_mb:<9.1f} | {r.model_size_mb:<9.2f}"
        )
        print(row)

    print("=" * 130)

    # 保存 JSON 结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "psutil_available": PSUTIL_AVAILABLE,
            "cpu_cores": num_cores,
        },
        "streaming_config": {
            "chunk_ms": streaming_config.chunk_ms,
            "simulate_delay": streaming_config.simulate_delay,
        },
        "data_summary": {
            "positive_samples": len(positive_samples),
            "negative_normal_samples": len(negative_normal),
            "negative_homophone_samples": len(negative_homophone),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
