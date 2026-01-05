#!/usr/bin/env python3
"""
流式 ASR Pipeline - 支持中间结果 (Partial) 和最终结果 (Final)

核心功能：
1. VAD 检测语音边界
2. 3 源分离（Conv-TasNet）
3. 说话人验证（筛选目标说话人）
4. 流式 ASR（维护上下文状态，输出 Partial/Final）

流式特性：
- 实时输出中间识别结果（Partial），用户可以看到正在识别的文字
- 在检测到句尾（静音或最大时长）时输出最终结果（Final）
- 支持上下文累积，避免词语断裂
"""
import time
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import sherpa_onnx

from overlap3_core import (
    _build_asr,
    _provider_to_torch_device,
    _ensure_sr_np,
    l2norm,
    G_SAMPLE_RATE,
)
from src.osd.separation import Separator
from src.osd.streaming_asr import StreamingASR, ASRResult, ResultType


class StreamResultType(Enum):
    """流式结果类型"""

    PARTIAL = "partial"  # 中间结果（正在识别，可能会变化）
    FINAL = "final"  # 最终结果（句子完成，不会再变）


@dataclass
class StreamingResult:
    """流式识别结果"""

    seq_id: int  # 分段序号
    stream_id: int  # 分离源 ID
    text: str  # 识别文本
    result_type: StreamResultType  # 结果类型
    sv_score: float  # 说话人验证分数
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def is_final(self) -> bool:
        return self.result_type == StreamResultType.FINAL

    @property
    def is_partial(self) -> bool:
        return self.result_type == StreamResultType.PARTIAL

    def to_dict(self) -> Dict:
        return {
            "seq_id": self.seq_id,
            "stream": self.stream_id,
            "text": self.text,
            "kind": f"vad_{self.result_type.value}",
            "sv_score": round(self.sv_score, 4) if self.sv_score else None,
            "start": round(self.start_time, 3),
            "end": round(self.end_time, 3),
            "is_final": self.is_final,
        }


@dataclass
class SeparatedStreamState:
    """每个分离流的状态（维护 ASR 上下文）"""

    stream_id: int
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    last_partial_text: str = ""
    last_partial_time: float = 0.0
    accumulated_text: str = ""  # 累积的最终文本

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_partial_text = ""


class StreamingASRPipeline:
    """流式 ASR Pipeline

    支持：
    - VAD 语音边界检测
    - 3 源分离
    - 说话人验证
    - 流式 ASR（Partial + Final 输出）
    """

    def __init__(self, args, target_wav_path: str):
        self.args = args
        self.device = _provider_to_torch_device(args.provider)

        # 初始化 ASR
        self.asr = _build_asr(args)

        # 初始化分离器
        self.sep = Separator(
            backend=args.sep_backend or "asteroid",
            checkpoint=(args.sep_checkpoint or None),
            device=self.device,
            n_src=3,
        )

        # 初始化说话人验证
        se_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.spk_embed_model,
            num_threads=getattr(args, "num_threads", 1),
            debug=getattr(args, "debug", False),
            provider=args.provider,
        )
        if not se_config.validate():
            raise ValueError(f"Invalid speaker embedding config: {se_config}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(se_config)

        # 初始化 VAD
        vad_model_path = getattr(args, "vad_model", None)
        if not vad_model_path:
            raise ValueError("VAD model path is required (--vad-model)")

        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.silero_vad.min_silence_duration = getattr(
            args, "vad_min_silence", 0.25
        )
        vad_config.silero_vad.min_speech_duration = getattr(
            args, "vad_min_speech", 0.25
        )
        vad_config.sample_rate = G_SAMPLE_RATE

        if not vad_config.validate():
            raise ValueError("Invalid VAD config")

        self.vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=100
        )
        self.vad_window_size = vad_config.silero_vad.window_size

        # 配置参数
        self.max_segment_duration = getattr(args, "max_segment_duration", 3.0)
        self.partial_interval = getattr(
            args, "partial_interval", 0.5
        )  # 中间结果输出间隔

        # 状态管理
        self.audio_buffer = np.array([], dtype=np.float32)
        self.pending_audio = np.array([], dtype=np.float32)
        self.pending_start_time = 0.0
        self.current_time = 0.0
        self.last_partial_time = 0.0
        self.segment_seq = 0

        # 每个分离流的状态
        self.stream_states: Dict[int, SeparatedStreamState] = {
            i: SeparatedStreamState(stream_id=i) for i in range(3)
        }

        # 结果队列
        self.results_queue = Queue()

        # 工作线程
        self._segment_queue = Queue()
        self._shutdown = False
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="StreamingASRWorker"
        )
        self._worker_thread.start()

        # 加载目标说话人
        self._load_target_speaker(target_wav_path)

    def _load_target_speaker(self, target_wav_path: str):
        """加载目标说话人"""
        import torchaudio

        t_wav, t_sr = torchaudio.load(target_wav_path)
        print(f"Target audio sample rate: {t_sr}Hz")

        if t_sr != G_SAMPLE_RATE:
            t_wav = torchaudio.functional.resample(t_wav, t_sr, G_SAMPLE_RATE)
            t_sr = G_SAMPLE_RATE

        t_np, _ = _ensure_sr_np(t_wav, int(t_sr), G_SAMPLE_RATE)

        # 计算说话人嵌入
        s = self.extractor.create_stream()
        s.accept_waveform(G_SAMPLE_RATE, t_np)
        s.input_finished()
        enrolled_vec = np.array(self.extractor.compute(s), dtype=np.float32)
        self.enrolled_vec_norm = l2norm(enrolled_vec)

        # ASR 获取目标文本
        st_tgt = self.asr.create_stream()
        st_tgt.accept_waveform(G_SAMPLE_RATE, t_np)
        self.asr.decode_stream(st_tgt)
        self.target_src_text = st_tgt.result.text or ""
        print(f"Target speaker enrolled. Text: '{self.target_src_text}'")

    def add_audio_data(self, audio_chunk: np.ndarray) -> List[StreamingResult]:
        """添加音频数据，返回识别结果列表（可能包含 Partial 和 Final）"""
        results = []

        # 累积待处理音频
        if len(self.pending_audio) == 0:
            self.pending_start_time = self.current_time
        self.pending_audio = np.concatenate([self.pending_audio, audio_chunk])
        self.current_time += len(audio_chunk) / G_SAMPLE_RATE

        # 发送给 VAD
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        while len(self.audio_buffer) > self.vad_window_size:
            window = self.audio_buffer[: self.vad_window_size]
            self.vad.accept_waveform(window)
            self.audio_buffer = self.audio_buffer[self.vad_window_size :]

        # 检查 VAD 状态
        vad_detected_end = False
        while not self.vad.empty():
            vad_detected_end = True
            self.vad.pop()

        pending_duration = len(self.pending_audio) / G_SAMPLE_RATE
        time_since_partial = self.current_time - self.last_partial_time

        # 决定输出类型（只依靠 VAD，不使用最大时长强制输出）
        if vad_detected_end and pending_duration >= 0.3:
            # VAD 检测到句尾 → 输出 Final
            results.extend(self._emit_final_results())
        elif time_since_partial >= self.partial_interval and pending_duration >= 0.5:
            # 到达中间结果间隔 → 输出 Partial
            results.extend(self._emit_partial_results())

        return results

    def _emit_partial_results(self) -> List[StreamingResult]:
        """输出中间结果（Partial）

        优化：只处理最近一段音频（增量处理），避免重复处理整段累积音频
        这样可以保持 O(n) 复杂度而不是 O(n²)
        """
        results = []

        pending_duration = len(self.pending_audio) / G_SAMPLE_RATE
        if pending_duration < 0.3:
            return results

        # 只处理最近 partial_window_size 秒的音频（增量处理）
        partial_window_size = min(3.0, pending_duration)  # 最多处理最近 3 秒
        partial_samples = int(partial_window_size * G_SAMPLE_RATE)
        audio_to_process = self.pending_audio[-partial_samples:]

        # 分离音频
        try:
            separated = self.sep.separate(audio_to_process, G_SAMPLE_RATE)
        except Exception as e:
            print(f"Separation error: {e}")
            return results

        # 对每个分离流进行处理
        for stream_id, stream_audio in enumerate(separated):
            # 说话人验证
            sv_score, matched = self._speaker_verification(stream_audio, G_SAMPLE_RATE)

            if matched:
                # ASR 转录
                text = self._transcribe(stream_audio, G_SAMPLE_RATE)

                if text and text != self.stream_states[stream_id].last_partial_text:
                    self.stream_states[stream_id].last_partial_text = text

                    result = StreamingResult(
                        seq_id=self.segment_seq,
                        stream_id=stream_id,
                        text=text,
                        result_type=StreamResultType.PARTIAL,
                        sv_score=sv_score,
                        start_time=self.current_time - partial_window_size,
                        end_time=self.current_time,
                    )
                    results.append(result)
                    self.results_queue.put(result.to_dict())

        self.last_partial_time = self.current_time
        return results

    def _emit_final_results(self) -> List[StreamingResult]:
        """输出最终结果（Final）"""
        results = []

        if len(self.pending_audio) < 0.2 * G_SAMPLE_RATE:
            self.pending_audio = np.array([], dtype=np.float32)
            return results

        self.segment_seq += 1

        # 分离音频
        try:
            separated = self.sep.separate(self.pending_audio, G_SAMPLE_RATE)
        except Exception as e:
            print(f"Separation error: {e}")
            self.pending_audio = np.array([], dtype=np.float32)
            return results

        # 对每个分离流进行处理
        for stream_id, stream_audio in enumerate(separated):
            # 说话人验证
            sv_score, matched = self._speaker_verification(stream_audio, G_SAMPLE_RATE)

            if matched:
                # ASR 转录
                text = self._transcribe(stream_audio, G_SAMPLE_RATE)

                if text:
                    # 累积到该流的总文本
                    self.stream_states[stream_id].accumulated_text += text

                    result = StreamingResult(
                        seq_id=self.segment_seq,
                        stream_id=stream_id,
                        text=text,
                        result_type=StreamResultType.FINAL,
                        sv_score=sv_score,
                        start_time=self.pending_start_time,
                        end_time=self.current_time,
                    )
                    results.append(result)
                    self.results_queue.put(result.to_dict())

        # 重置状态
        self.pending_audio = np.array([], dtype=np.float32)
        self.last_partial_time = self.current_time
        for state in self.stream_states.values():
            state.reset()

        return results

    def _speaker_verification(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[Optional[float], bool]:
        """说话人验证"""
        try:
            sstream = self.extractor.create_stream()
            sstream.accept_waveform(sample_rate, audio)
            sstream.input_finished()

            if self.extractor.is_ready(sstream):
                emb = np.array(self.extractor.compute(sstream), dtype=np.float32)
                emb_norm = l2norm(emb)
                sv_score = float(np.dot(emb_norm, self.enrolled_vec_norm))
                return sv_score, sv_score >= self.args.sv_threshold
        except Exception as e:
            print(f"Speaker verification error: {e}")

        return None, False

    def _transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """ASR 转录"""
        try:
            st = self.asr.create_stream()
            st.accept_waveform(sample_rate, audio)
            self.asr.decode_stream(st)
            return st.result.text or ""
        except Exception as e:
            print(f"ASR error: {e}")
            return ""

    def _worker_loop(self):
        """工作线程（暂时保留，用于未来异步处理）"""
        while not self._shutdown:
            try:
                time.sleep(0.1)
            except Exception:
                pass

    def flush(self) -> List[StreamingResult]:
        """刷新所有剩余数据"""
        # 处理剩余缓冲
        if len(self.audio_buffer) > 0:
            self.pending_audio = np.concatenate([self.pending_audio, self.audio_buffer])
            self.audio_buffer = np.array([], dtype=np.float32)

        # VAD flush
        self.vad.flush()
        while not self.vad.empty():
            self.vad.pop()

        # 输出最终结果
        return self._emit_final_results()

    def get_accumulated_text(self, stream_id: int = None) -> str:
        """获取累积的识别文本"""
        if stream_id is not None:
            return self.stream_states[stream_id].accumulated_text

        # 返回所有流的累积文本
        return "".join(state.accumulated_text for state in self.stream_states.values())

    def shutdown(self):
        """关闭 pipeline"""
        self._shutdown = True
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
