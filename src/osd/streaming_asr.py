"""流式 ASR 识别器 - 支持中间结果 (Partial) 和最终结果 (Final)。

本模块提供 StreamingASR 类，用于实时语音识别场景：
- 维护上下文状态，持续接收音频数据
- 输出中间识别结果（Partial Result）
- 在检测到句尾时输出最终结果（Final Result）

支持的 ASR 后端：
- sherpa_onnx 的 OnlineRecognizer（真正的流式模型）
- 或基于 VAD 的伪流式（使用 OfflineRecognizer 分段处理）

Usage:
    asr = StreamingASR(
        model_type="sense_voice",
        model_path="path/to/model.onnx",
        tokens_path="path/to/tokens.txt",
    )

    # 流式处理
    for audio_chunk in audio_stream:
        result = asr.process_chunk(audio_chunk)
        if result.is_final:
            print(f"Final: {result.text}")
        else:
            print(f"Partial: {result.text}")

    # 结束时刷新
    final = asr.flush()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
from enum import Enum
import numpy as np

G_SAMPLE_RATE = 16000


class ResultType(Enum):
    """识别结果类型"""

    PARTIAL = "partial"  # 中间结果（可能会被后续结果修正）
    FINAL = "final"  # 最终结果（句子结束，不会再修改）


@dataclass
class ASRResult:
    """ASR 识别结果"""

    text: str  # 识别文本
    result_type: ResultType  # 结果类型（partial/final）
    start_time: float = 0.0  # 开始时间（相对于整个流）
    end_time: float = 0.0  # 结束时间
    confidence: float = 1.0  # 置信度（如果模型支持）

    @property
    def is_final(self) -> bool:
        return self.result_type == ResultType.FINAL

    @property
    def is_partial(self) -> bool:
        return self.result_type == ResultType.PARTIAL


@dataclass
class StreamingASR:
    """流式 ASR 识别器

    支持两种模式：
    1. 真流式模式（use_online=True）：使用 sherpa_onnx OnlineRecognizer
    2. 伪流式模式（use_online=False）：基于 VAD 分段 + OfflineRecognizer

    参数：
    - model_type: 模型类型 ("sense_voice", "paraformer", "transducer", "zipformer")
    - model_path: 模型文件路径
    - tokens_path: tokens.txt 路径
    - use_online: 是否使用在线流式模型
    - provider: 推理后端 ("cpu", "cuda")
    - endpoint_rule: 端点检测规则配置
    """

    # 模型配置
    model_type: str = "sense_voice"
    model_path: str = ""
    tokens_path: str = ""

    # 在线模型额外参数（用于 transducer/zipformer）
    encoder_path: str = ""
    decoder_path: str = ""
    joiner_path: str = ""

    # 运行配置
    use_online: bool = False  # 是否使用真正的在线流式识别
    provider: str = "cpu"
    num_threads: int = 2
    sample_rate: int = G_SAMPLE_RATE

    # 端点检测配置
    endpoint_rule1_min_trailing_silence: float = 2.4  # 句尾静音阈值
    endpoint_rule2_min_trailing_silence: float = 1.2  # 中间停顿阈值
    endpoint_rule3_min_utterance_length: float = 20.0  # 最大句子长度

    # 内部状态
    _recognizer: any = field(default=None, repr=False)
    _stream: any = field(default=None, repr=False)
    _audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32), repr=False
    )
    _current_time: float = field(default=0.0, repr=False)
    _last_partial_text: str = field(default="", repr=False)
    _segment_start_time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        """初始化 ASR 识别器"""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._current_time = 0.0
        self._last_partial_text = ""
        self._segment_start_time = 0.0

        if self.use_online:
            self._init_online_recognizer()
        else:
            self._init_offline_recognizer()

    def _init_online_recognizer(self):
        """初始化在线流式识别器（sherpa_onnx OnlineRecognizer）"""
        import sherpa_onnx

        # 端点检测规则配置
        endpoint_config = sherpa_onnx.EndpointConfig(
            rule1=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence=self.endpoint_rule1_min_trailing_silence,
                min_utterance_length=0,
            ),
            rule2=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=True,
                min_trailing_silence=self.endpoint_rule2_min_trailing_silence,
                min_utterance_length=0,
            ),
            rule3=sherpa_onnx.EndpointRule(
                must_contain_nonsilence=False,
                min_trailing_silence=0,
                min_utterance_length=self.endpoint_rule3_min_utterance_length,
            ),
        )

        if self.model_type == "transducer" or self.encoder_path:
            # Transducer 模型（Zipformer 等）
            config = sherpa_onnx.OnlineRecognizerConfig(
                tokens=self.tokens_path,
                model_config=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder=self.encoder_path or self.model_path,
                        decoder=self.decoder_path,
                        joiner=self.joiner_path,
                    ),
                    num_threads=self.num_threads,
                    provider=self.provider,
                ),
                endpoint_config=endpoint_config,
                enable_endpoint_detection=True,
            )
        else:
            # 其他在线模型（如有）
            raise ValueError(
                f"Online streaming not directly supported for model_type={self.model_type}. "
                "Please use transducer/zipformer models for true streaming, "
                "or set use_online=False for VAD-based pseudo-streaming."
            )

        self._recognizer = sherpa_onnx.OnlineRecognizer(config)
        self._stream = self._recognizer.create_stream()
        print(f"Initialized online streaming ASR: {self.model_type}")

    def _init_offline_recognizer(self):
        """初始化离线识别器（用于伪流式模式）"""
        import sherpa_onnx

        if self.model_type == "sense_voice" or "sense" in self.model_path.lower():
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self.model_path,
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                use_itn=True,
                debug=False,
            )
        elif self.model_type == "paraformer" or "paraformer" in self.model_path.lower():
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=self.model_path,
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
            )
        elif self.encoder_path:  # Transducer offline
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=self.encoder_path,
                decoder=self.decoder_path,
                joiner=self.joiner_path,
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                sample_rate=self.sample_rate,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        print(f"Initialized offline ASR for pseudo-streaming: {self.model_type}")

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """处理一个音频块，返回识别结果（可能是 Partial 或 Final）

        Args:
            audio_chunk: 音频数据 (float32, 16kHz)

        Returns:
            ASRResult 或 None（如果没有新结果）
        """
        chunk_duration = len(audio_chunk) / self.sample_rate
        self._current_time += chunk_duration

        if self.use_online:
            return self._process_chunk_online(audio_chunk)
        else:
            return self._process_chunk_offline(audio_chunk)

    def _process_chunk_online(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """在线流式处理"""
        self._stream.accept_waveform(self.sample_rate, audio_chunk)

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        # 检查是否到达端点（句子结束）
        is_endpoint = self._recognizer.is_endpoint(self._stream)
        text = self._recognizer.get_result(self._stream)

        if not text:
            return None

        if is_endpoint:
            # 最终结果
            result = ASRResult(
                text=text,
                result_type=ResultType.FINAL,
                start_time=self._segment_start_time,
                end_time=self._current_time,
            )
            # 重置流以开始新句子
            self._recognizer.reset(self._stream)
            self._segment_start_time = self._current_time
            self._last_partial_text = ""
            return result
        else:
            # 中间结果（仅当文本有变化时返回）
            if text != self._last_partial_text:
                self._last_partial_text = text
                return ASRResult(
                    text=text,
                    result_type=ResultType.PARTIAL,
                    start_time=self._segment_start_time,
                    end_time=self._current_time,
                )

        return None

    def _process_chunk_offline(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """伪流式处理（累积音频，定期输出中间结果）"""
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        # 每累积一定量音频输出一次中间结果
        buffer_duration = len(self._audio_buffer) / self.sample_rate

        if buffer_duration >= 1.0:  # 每 1 秒输出一次中间结果
            text = self._transcribe_buffer()
            if text and text != self._last_partial_text:
                self._last_partial_text = text
                return ASRResult(
                    text=text,
                    result_type=ResultType.PARTIAL,
                    start_time=self._segment_start_time,
                    end_time=self._current_time,
                )

        return None

    def _transcribe_buffer(self) -> str:
        """使用离线识别器转录缓冲区中的音频"""
        if len(self._audio_buffer) < 0.1 * self.sample_rate:
            return ""

        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, self._audio_buffer)
        self._recognizer.decode_stream(stream)
        return stream.result.text or ""

    def finalize_segment(self) -> Optional[ASRResult]:
        """标记当前段落结束，输出最终结果

        在 VAD 检测到静音或强制分段时调用
        """
        if self.use_online:
            # 获取当前结果并重置
            text = self._recognizer.get_result(self._stream)
            if text:
                result = ASRResult(
                    text=text,
                    result_type=ResultType.FINAL,
                    start_time=self._segment_start_time,
                    end_time=self._current_time,
                )
                self._recognizer.reset(self._stream)
                self._segment_start_time = self._current_time
                self._last_partial_text = ""
                return result
        else:
            # 离线模式：转录缓冲区并清空
            if len(self._audio_buffer) >= 0.2 * self.sample_rate:
                text = self._transcribe_buffer()
                if text:
                    result = ASRResult(
                        text=text,
                        result_type=ResultType.FINAL,
                        start_time=self._segment_start_time,
                        end_time=self._current_time,
                    )
                    self._audio_buffer = np.array([], dtype=np.float32)
                    self._segment_start_time = self._current_time
                    self._last_partial_text = ""
                    return result

        # 清空缓冲区
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_partial_text = ""
        return None

    def flush(self) -> Optional[ASRResult]:
        """刷新所有剩余数据，返回最终结果

        在流结束时调用
        """
        if self.use_online:
            self._stream.input_finished()
            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)
            text = self._recognizer.get_result(self._stream)
            if text:
                return ASRResult(
                    text=text,
                    result_type=ResultType.FINAL,
                    start_time=self._segment_start_time,
                    end_time=self._current_time,
                )
        else:
            return self.finalize_segment()

        return None

    def reset(self):
        """重置识别器状态，开始新的识别会话"""
        if self.use_online and self._stream:
            self._recognizer.reset(self._stream)

        self._audio_buffer = np.array([], dtype=np.float32)
        self._current_time = 0.0
        self._last_partial_text = ""
        self._segment_start_time = 0.0


@dataclass
class VADStreamingASR:
    """结合 VAD 的流式 ASR

    使用 VAD 检测语音段落边界，在静音时输出 Final 结果，
    在语音过程中输出 Partial 结果。
    """

    # ASR 配置
    asr: StreamingASR = field(default_factory=StreamingASR)

    # VAD 配置
    vad_model_path: str = ""
    vad_min_silence_duration: float = 0.25
    vad_min_speech_duration: float = 0.25

    # 最大分段时长
    max_segment_duration: float = 10.0

    # 中间结果输出间隔
    partial_result_interval: float = 0.5

    # 内部状态
    _vad: any = field(default=None, repr=False)
    _vad_window_size: int = field(default=512, repr=False)
    _audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32), repr=False
    )
    _speech_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32), repr=False
    )
    _current_time: float = field(default=0.0, repr=False)
    _last_partial_time: float = field(default=0.0, repr=False)
    _in_speech: bool = field(default=False, repr=False)
    _speech_start_time: float = field(default=0.0, repr=False)

    def __post_init__(self):
        """初始化 VAD"""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._current_time = 0.0
        self._last_partial_time = 0.0
        self._in_speech = False
        self._speech_start_time = 0.0

        if self.vad_model_path:
            self._init_vad()

    def _init_vad(self):
        """初始化 VAD 模型"""
        import sherpa_onnx

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = self.vad_model_path
        vad_config.silero_vad.min_silence_duration = self.vad_min_silence_duration
        vad_config.silero_vad.min_speech_duration = self.vad_min_speech_duration
        vad_config.sample_rate = self.asr.sample_rate

        if not vad_config.validate():
            raise ValueError("Invalid VAD config")

        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=100
        )
        self._vad_window_size = vad_config.silero_vad.window_size
        print(f"Initialized VAD with model: {self.vad_model_path}")

    def process_chunk(self, audio_chunk: np.ndarray) -> List[ASRResult]:
        """处理音频块，返回识别结果列表

        可能返回多个结果（Partial 和/或 Final）
        """
        results = []
        chunk_duration = len(audio_chunk) / self.asr.sample_rate
        self._current_time += chunk_duration

        if self._vad is None:
            # 无 VAD，直接使用 ASR 处理
            result = self.asr.process_chunk(audio_chunk)
            if result:
                results.append(result)
            return results

        # 将音频发送给 VAD
        self._audio_buffer = np.concatenate([self._audio_buffer, audio_chunk])

        while len(self._audio_buffer) >= self._vad_window_size:
            window = self._audio_buffer[: self._vad_window_size]
            self._vad.accept_waveform(window)
            self._audio_buffer = self._audio_buffer[self._vad_window_size :]

            # 累积到语音缓冲
            self._speech_buffer = np.concatenate([self._speech_buffer, window])

        # 检查 VAD 是否检测到语音结束
        vad_detected_speech_end = False
        while not self._vad.empty():
            # VAD 队列非空表示检测到了语音段结束
            vad_detected_speech_end = True
            self._vad.pop()

        # 检查是否需要输出结果
        speech_duration = len(self._speech_buffer) / self.asr.sample_rate
        time_since_partial = self._current_time - self._last_partial_time

        # 条件1: VAD 检测到语音结束 → Final
        # 条件2: 超过最大分段时长 → Final
        # 条件3: 到达中间结果输出间隔 → Partial

        if vad_detected_speech_end and speech_duration >= 0.3:
            # 输出最终结果
            result = self._finalize_current_segment()
            if result:
                results.append(result)
        elif speech_duration >= self.max_segment_duration:
            # 强制分段，输出最终结果
            result = self._finalize_current_segment()
            if result:
                results.append(result)
        elif (
            time_since_partial >= self.partial_result_interval
            and speech_duration >= 0.5
        ):
            # 输出中间结果
            result = self._emit_partial_result()
            if result:
                results.append(result)

        return results

    def _emit_partial_result(self) -> Optional[ASRResult]:
        """输出中间结果"""
        if len(self._speech_buffer) < 0.3 * self.asr.sample_rate:
            return None

        # 使用 ASR 转录当前缓冲
        text = self._transcribe_buffer(self._speech_buffer)
        if text:
            self._last_partial_time = self._current_time
            return ASRResult(
                text=text,
                result_type=ResultType.PARTIAL,
                start_time=self._speech_start_time,
                end_time=self._current_time,
            )
        return None

    def _finalize_current_segment(self) -> Optional[ASRResult]:
        """完成当前语音段，输出最终结果"""
        if len(self._speech_buffer) < 0.2 * self.asr.sample_rate:
            self._speech_buffer = np.array([], dtype=np.float32)
            return None

        text = self._transcribe_buffer(self._speech_buffer)
        end_time = self._current_time

        # 重置状态
        self._speech_buffer = np.array([], dtype=np.float32)
        self._speech_start_time = self._current_time
        self._last_partial_time = self._current_time

        if text:
            return ASRResult(
                text=text,
                result_type=ResultType.FINAL,
                start_time=self._speech_start_time,
                end_time=end_time,
            )
        return None

    def _transcribe_buffer(self, audio: np.ndarray) -> str:
        """转录音频缓冲区"""
        if len(audio) < 0.1 * self.asr.sample_rate:
            return ""

        stream = self.asr._recognizer.create_stream()
        stream.accept_waveform(self.asr.sample_rate, audio)
        self.asr._recognizer.decode_stream(stream)
        return stream.result.text or ""

    def flush(self) -> Optional[ASRResult]:
        """刷新所有剩余数据"""
        # 处理剩余的音频缓冲
        if len(self._audio_buffer) > 0:
            self._speech_buffer = np.concatenate(
                [self._speech_buffer, self._audio_buffer]
            )
            self._audio_buffer = np.array([], dtype=np.float32)

        # 如果 VAD 存在，刷新它
        if self._vad:
            self._vad.flush()
            while not self._vad.empty():
                self._vad.pop()

        # 输出最终结果
        return self._finalize_current_segment()

    def reset(self):
        """重置状态"""
        self._audio_buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._current_time = 0.0
        self._last_partial_time = 0.0
        self._in_speech = False
        self._speech_start_time = 0.0
        self.asr.reset()
