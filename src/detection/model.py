"""Keyword Spotting (KWS) 模型定义模块。

本模块提供基于 sherpa-onnx 的关键词唤醒能力：
- KeywordSpotterModel: 通用唤醒词检测器封装
- 支持自定义唤醒词（拼音格式）
- 支持多种 Zipformer Transducer 预训练模型

典型用法:
    model = KeywordSpotterModel(
        encoder="models/xxx/encoder.onnx",
        decoder="models/xxx/decoder.onnx",
        joiner="models/xxx/joiner.onnx",
        tokens="models/xxx/tokens.txt",
        keywords="n ǐ h ǎo zh ēn zh ēn @你好真真",
    )
    detections = model.detect(samples, sample_rate)
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import sherpa_onnx

# 全局采样率常量
G_SAMPLE_RATE = 16000


@dataclass
class KeywordDetection:
    """单次唤醒词检测结果。

    Attributes:
        keyword: 检测到的唤醒词文本
        start_time: 检测时刻（秒），相对于音频起点
        tokens: 分词后的 token 列表
        timestamps: 各 token 对应的时间戳列表
    """

    keyword: str
    start_time: float
    tokens: List[str] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


@dataclass
class KWSModelConfig:
    """KWS 模型配置。

    Attributes:
        encoder: encoder ONNX 模型路径
        decoder: decoder ONNX 模型路径
        joiner: joiner ONNX 模型路径
        tokens: tokens.txt 文件路径
        keywords_file: 预定义唤醒词文件路径（可选）
        keywords: 运行时动态唤醒词字符串，格式如 "n ǐ h ǎo @你好"
        num_threads: 推理线程数
        sample_rate: 采样率
        feature_dim: 特征维度
        max_active_paths: 最大激活路径数
        keywords_score: 唤醒词得分权重
        keywords_threshold: 唤醒词触发阈值（越大越难触发）
        num_trailing_blanks: 尾部空白帧数
        provider: 推理后端 ("cpu" / "cuda")
    """

    encoder: str
    decoder: str
    joiner: str
    tokens: str
    keywords_file: str = ""
    keywords: str = ""
    num_threads: int = 2
    sample_rate: int = G_SAMPLE_RATE
    feature_dim: int = 80
    max_active_paths: int = 4
    keywords_score: float = 1.0
    keywords_threshold: float = 0.25
    num_trailing_blanks: int = 1
    provider: str = "cpu"

    def validate(self) -> bool:
        """验证配置有效性。"""
        required_files = [self.encoder, self.decoder, self.joiner, self.tokens]
        for f in required_files:
            if not Path(f).exists():
                raise FileNotFoundError(f"Required file not found: {f}")
        if self.keywords_file and not Path(self.keywords_file).exists():
            raise FileNotFoundError(f"Keywords file not found: {self.keywords_file}")
        return True


class KeywordSpotterModel:
    """关键词唤醒模型封装类。

    基于 sherpa-onnx KeywordSpotter 实现，支持：
    - 文件级别批量检测
    - 流式检测（模拟流式）
    - 自定义唤醒词
    """

    def __init__(self, config: KWSModelConfig) -> None:
        """初始化唤醒词检测器。

        Args:
            config: KWS 模型配置
        """
        config.validate()
        self.config = config
        self._spotter: Optional[sherpa_onnx.KeywordSpotter] = None
        self._temp_keywords_file: Optional[str] = None
        self._init_spotter()

    def __del__(self) -> None:
        """清理临时文件。"""
        if self._temp_keywords_file and os.path.exists(self._temp_keywords_file):
            try:
                os.unlink(self._temp_keywords_file)
            except OSError:
                pass

    def _get_keywords_file(self) -> str:
        """获取有效的 keywords 文件路径。

        如果配置了 keywords_file 则直接使用，
        否则从 keywords 字符串创建临时文件。
        """
        if self.config.keywords_file:
            return self.config.keywords_file

        if self.config.keywords:
            # 从 keywords 字符串创建临时文件
            fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="kws_keywords_")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                # 支持多个关键词用 / 分隔
                keywords_list = self.config.keywords.split("/")
                for kw in keywords_list:
                    f.write(kw.strip() + "\n")
            self._temp_keywords_file = temp_path
            return temp_path

        raise ValueError("Either keywords_file or keywords must be provided")

    def _init_spotter(self) -> None:
        """初始化 sherpa-onnx KeywordSpotter。"""
        keywords_file = self._get_keywords_file()
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=self.config.tokens,
            encoder=self.config.encoder,
            decoder=self.config.decoder,
            joiner=self.config.joiner,
            keywords_file=keywords_file,
            num_threads=self.config.num_threads,
            sample_rate=self.config.sample_rate,
            feature_dim=self.config.feature_dim,
            max_active_paths=self.config.max_active_paths,
            keywords_score=self.config.keywords_score,
            keywords_threshold=self.config.keywords_threshold,
            num_trailing_blanks=self.config.num_trailing_blanks,
            provider=self.config.provider,
        )

    def create_stream(self, keywords: Optional[str] = None) -> sherpa_onnx.OnlineStream:
        """创建检测流。

        Args:
            keywords: 可选的运行时唤醒词（覆盖配置中的 keywords_file）
                      格式: "n ǐ h ǎo @你好/x iǎo m ǐ @小米"

        Returns:
            OnlineStream 实例
        """
        if self._spotter is None:
            raise RuntimeError("KeywordSpotter not initialized")
        kw = keywords or self.config.keywords
        if kw:
            return self._spotter.create_stream(kw)
        return self._spotter.create_stream()

    def detect(
        self,
        samples: np.ndarray,
        sample_rate: int = G_SAMPLE_RATE,
        keywords: Optional[str] = None,
        chunk_size_ms: int = 100,
    ) -> Tuple[List[KeywordDetection], float]:
        """对音频进行唤醒词检测。

        Args:
            samples: float32 音频波形，形状 (N,)
            sample_rate: 采样率
            keywords: 可选的运行时唤醒词
            chunk_size_ms: 模拟流式的 chunk 大小（毫秒）

        Returns:
            (detections, process_time_sec): 检测结果列表和处理耗时
        """
        if self._spotter is None:
            raise RuntimeError("KeywordSpotter not initialized")

        stream = self.create_stream(keywords)
        detections: List[KeywordDetection] = []

        # 模拟流式处理
        chunk_samples = int(sample_rate * chunk_size_ms / 1000)
        num_chunks = (len(samples) + chunk_samples - 1) // chunk_samples

        start_time = time.perf_counter()

        for i in range(num_chunks):
            start_idx = i * chunk_samples
            end_idx = min(start_idx + chunk_samples, len(samples))
            chunk = samples[start_idx:end_idx]

            stream.accept_waveform(sample_rate, chunk)

            while self._spotter.is_ready(stream):
                self._spotter.decode_stream(stream)
                result = self._spotter.get_result(stream)
                if result:
                    detection = KeywordDetection(
                        keyword=result,
                        start_time=start_idx / sample_rate,
                        tokens=self._spotter.tokens(stream),
                        timestamps=self._spotter.timestamps(stream),
                    )
                    detections.append(detection)
                    # 重置流以继续检测后续唤醒词
                    self._spotter.reset_stream(stream)

        # 添加尾部填充以确保检测完整
        tail_padding = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
        stream.accept_waveform(sample_rate, tail_padding)

        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
            result = self._spotter.get_result(stream)
            if result:
                detection = KeywordDetection(
                    keyword=result,
                    start_time=len(samples) / sample_rate,
                    tokens=self._spotter.tokens(stream),
                    timestamps=self._spotter.timestamps(stream),
                )
                detections.append(detection)
                self._spotter.reset_stream(stream)

        process_time = time.perf_counter() - start_time
        return detections, process_time

    def detect_streaming(
        self,
        audio_chunks: Iterator[np.ndarray],
        sample_rate: int = G_SAMPLE_RATE,
        keywords: Optional[str] = None,
    ) -> Iterator[KeywordDetection]:
        """流式检测生成器。

        Args:
            audio_chunks: 音频块迭代器，每个块为 float32 数组
            sample_rate: 采样率
            keywords: 可选的运行时唤醒词

        Yields:
            KeywordDetection 实例
        """
        if self._spotter is None:
            raise RuntimeError("KeywordSpotter not initialized")

        stream = self.create_stream(keywords)
        current_time = 0.0

        for chunk in audio_chunks:
            stream.accept_waveform(sample_rate, chunk)
            chunk_duration = len(chunk) / sample_rate

            while self._spotter.is_ready(stream):
                self._spotter.decode_stream(stream)
                result = self._spotter.get_result(stream)
                if result:
                    detection = KeywordDetection(
                        keyword=result,
                        start_time=current_time,
                        tokens=self._spotter.tokens(stream),
                        timestamps=self._spotter.timestamps(stream),
                    )
                    yield detection
                    self._spotter.reset_stream(stream)

            current_time += chunk_duration


def create_kws_model(
    model_dir: str,
    keywords: str = "",
    keywords_file: str = "",
    provider: str = "cpu",
    num_threads: int = 2,
    keywords_threshold: float = 0.25,
    use_int8: bool = False,
    epoch: int = 12,
    avg: int = 2,
    chunk: int = 16,
    left: int = 64,
) -> KeywordSpotterModel:
    """工厂函数：从模型目录创建 KWS 模型。

    Args:
        model_dir: 模型目录路径（包含 encoder/decoder/joiner/tokens）
        keywords: 运行时唤醒词
        keywords_file: 唤醒词文件路径
        provider: 推理后端
        num_threads: 推理线程数
        keywords_threshold: 唤醒阈值
        use_int8: 是否使用 int8 量化模型
        epoch: 模型 epoch 编号（用于定位文件名）
        avg: 模型平均数
        chunk: chunk 大小
        left: left 大小

    Returns:
        KeywordSpotterModel 实例
    """
    model_dir_path = Path(model_dir)

    # 构建文件名后缀
    suffix = f"epoch-{epoch}-avg-{avg}-chunk-{chunk}-left-{left}"
    int8_suffix = ".int8" if use_int8 else ""

    # 尝试查找匹配的模型文件
    encoder_candidates = list(
        model_dir_path.glob(f"encoder-{suffix}{int8_suffix}.onnx")
    )
    if not encoder_candidates:
        # 尝试其他 epoch
        encoder_candidates = list(
            model_dir_path.glob(f"encoder-epoch-*{int8_suffix}.onnx")
        )

    if not encoder_candidates:
        raise FileNotFoundError(f"No encoder found in {model_dir}")

    encoder_path = encoder_candidates[0]

    # 从 encoder 文件名推断其他文件
    encoder_name = encoder_path.name
    base_suffix = encoder_name.replace("encoder-", "").replace(".onnx", "")

    decoder_path = model_dir_path / f"decoder-{base_suffix}.onnx"
    joiner_path = model_dir_path / f"joiner-{base_suffix}.onnx"
    tokens_path = model_dir_path / "tokens.txt"

    # 检查 decoder 是否存在（可能没有 int8 版本）
    if not decoder_path.exists():
        base_suffix_no_int8 = base_suffix.replace(".int8", "")
        decoder_path = model_dir_path / f"decoder-{base_suffix_no_int8}.onnx"

    if not joiner_path.exists():
        base_suffix_no_int8 = base_suffix.replace(".int8", "")
        joiner_path = model_dir_path / f"joiner-{base_suffix_no_int8}.onnx"

    config = KWSModelConfig(
        encoder=str(encoder_path),
        decoder=str(decoder_path),
        joiner=str(joiner_path),
        tokens=str(tokens_path),
        keywords=keywords,
        keywords_file=keywords_file,
        num_threads=num_threads,
        provider=provider,
        keywords_threshold=keywords_threshold,
    )

    return KeywordSpotterModel(config)


# 预定义的中文唤醒词拼音格式
KEYWORD_NIHAO_ZHENZHEN = "n ǐ h ǎo zh ēn zh ēn @你好真真"
KEYWORD_XIAOAI_TONGXUE = "x iǎo ài t óng x ué @小爱同学"
KEYWORD_XIAOMI_XIAOMI = "x iǎo m ǐ x iǎo m ǐ @小米小米"
KEYWORD_NIHAO_WENWEN = "n ǐ h ǎo w èn w èn @你好问问"
