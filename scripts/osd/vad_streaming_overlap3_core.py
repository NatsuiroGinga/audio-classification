#!/usr/bin/env python3
"""
VAD 版 StreamingOverlap3Pipeline - 使用 VAD 进行语音分段

改进点：
1. 使用 silero_vad 进行语音活动检测（VAD），按自然语音边界分段
2. 对每个 VAD 分段进行 3 源分离
3. 说话人验证筛选目标说话人
4. ASR 识别

优势：
- 基于语音边界分段，保持语义完整性
- 避免固定时间间隔导致的词语断裂
- 利用静音段作为自然分隔点
"""
import time
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

import sherpa_onnx

from overlap3_core import (
    _build_asr,
    _provider_to_torch_device,
    _ensure_sr_np,
    l2norm,
    G_SAMPLE_RATE,
)
from src.osd.separation import Separator


@dataclass
class VADSegment:
    """VAD 检测到的语音段"""

    audio_data: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    seq_id: int = 0


class VADStreamingOverlap3Pipeline:
    """VAD 版流式重叠语音处理管道

    使用 VAD 进行语音分段，保持语义完整性
    """

    def __init__(self, args, target_wav_path: str):
        self.args = args
        self.device = _provider_to_torch_device(args.provider)

        # 初始化 ASR 和分离组件
        self.asr = _build_asr(args)
        self.sep = Separator(
            backend=args.sep_backend or "asteroid",
            checkpoint=(args.sep_checkpoint or None),
            device=self.device,
            n_src=3,
        )

        # Speaker embedding extractor
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

        # 最大分段时长限制（秒），超过此时长强制分段，保证流式特性
        self.max_segment_duration = getattr(args, "max_segment_duration", 3.0)

        # 流式处理状态
        self.audio_buffer = np.array([], dtype=np.float32)
        self.segment_queue = Queue()
        self.results_queue = Queue()
        self._segment_seq = 0
        self._shutdown = False
        self._current_time = 0.0  # 当前处理的音频时间位置

        # 累积语音缓冲（用于流式处理）
        self._pending_audio = np.array([], dtype=np.float32)
        self._pending_start_time = 0.0
        self._last_emit_time = 0.0  # 上次发射分段的时间

        # 启动工作线程
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="VADStreamingWorker"
        )
        self._worker_thread.start()

        # 加载目标说话人
        self._load_target_speaker(target_wav_path)

    def _load_target_speaker(self, target_wav_path: str):
        """加载目标说话人语音和文本"""
        import torchaudio

        t_wav, t_sr = torchaudio.load(target_wav_path)
        print(f"Target audio original sample rate: {t_sr}Hz")

        if t_sr != G_SAMPLE_RATE:
            print(f"Resampling target audio from {t_sr}Hz to {G_SAMPLE_RATE}Hz")
            t_wav = torchaudio.functional.resample(t_wav, t_sr, G_SAMPLE_RATE)
            t_sr = G_SAMPLE_RATE

        t_np, _ = _ensure_sr_np(t_wav, int(t_sr), G_SAMPLE_RATE)

        # 计算说话人嵌入
        s = self.extractor.create_stream()
        s.accept_waveform(G_SAMPLE_RATE, t_np)
        s.input_finished()
        enrolled_vec = np.array(self.extractor.compute(s), dtype=np.float32)
        self.enrolled_vec_norm = l2norm(enrolled_vec)

        # 为目标语音创建 ASR 流获取文本
        st_tgt = self.asr.create_stream()
        st_tgt.accept_waveform(G_SAMPLE_RATE, t_np)
        self.asr.decode_stream(st_tgt)
        self.target_src_text = st_tgt.result.text or ""
        print(f"Target speaker enrolled. Text: '{self.target_src_text}'")

    def add_audio_data(self, audio_chunk: np.ndarray):
        """添加音频数据，结合 VAD 和最大时长限制进行流式分段"""
        # 累积待处理音频
        if len(self._pending_audio) == 0:
            self._pending_start_time = self._current_time
        self._pending_audio = np.concatenate([self._pending_audio, audio_chunk])
        self._current_time += len(audio_chunk) / G_SAMPLE_RATE

        # 同时将音频发送给 VAD 进行检测
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        while len(self.audio_buffer) > self.vad_window_size:
            window = self.audio_buffer[: self.vad_window_size]
            self.vad.accept_waveform(window)
            self.audio_buffer = self.audio_buffer[self.vad_window_size :]

        # 策略1: 如果 VAD 检测到静音（队列非空），说明一段语音结束
        vad_detected_end = False
        while not self.vad.empty():
            # VAD 检测到了一段完整的语音
            vad_detected_end = True
            self.vad.pop()

        # 策略2: 检查是否超过最大分段时长
        pending_duration = len(self._pending_audio) / G_SAMPLE_RATE
        should_emit = (
            vad_detected_end  # VAD 检测到语音结束
            or pending_duration >= self.max_segment_duration  # 超过最大时长
        )

        if should_emit and pending_duration >= 0.3:  # 至少 0.3 秒
            self._emit_pending_segment()

    def _emit_pending_segment(self):
        """发射当前累积的语音段到处理队列"""
        if len(self._pending_audio) < 0.3 * G_SAMPLE_RATE:
            self._pending_audio = np.array([], dtype=np.float32)
            return

        duration = len(self._pending_audio) / G_SAMPLE_RATE
        end_time = self._pending_start_time + duration

        self._segment_seq += 1
        segment = VADSegment(
            audio_data=self._pending_audio.copy(),
            start_time=self._pending_start_time,
            end_time=end_time,
            sample_rate=G_SAMPLE_RATE,
            seq_id=self._segment_seq,
        )
        self.segment_queue.put(segment)

        # 重置状态
        self._pending_audio = np.array([], dtype=np.float32)
        self._last_emit_time = self._current_time

    def _worker_loop(self):
        """工作线程循环"""
        while not self._shutdown:
            try:
                segment = self.segment_queue.get(timeout=1.0)
                self._process_segment(segment)
                self.segment_queue.task_done()
            except Exception:
                pass

    def _process_segment(self, segment: VADSegment):
        """处理单个 VAD 语音段"""
        try:
            audio_data = segment.audio_data
            sample_rate = segment.sample_rate

            # 3 源分离
            separation_start = time.time()
            separated_streams = self.sep.separate(audio_data, sample_rate)
            separation_time = time.time() - separation_start

            # 说话人验证和识别
            for stream_id, stream_audio in enumerate(separated_streams):
                sv_score, matched = self._speaker_verification(
                    stream_audio, sample_rate
                )

                if matched:
                    asr_start = time.time()
                    text, _ = self._transcribe_audio(stream_audio, sample_rate)
                    asr_time = time.time() - asr_start

                    result = {
                        "seq_id": segment.seq_id,
                        "start": round(segment.start_time, 3),
                        "end": round(segment.end_time, 3),
                        "duration": round(segment.end_time - segment.start_time, 3),
                        "kind": "vad_separated",
                        "stream": stream_id,
                        "text": text,
                        "sv_score": (
                            round(sv_score, 4) if sv_score is not None else None
                        ),
                        "target_src_text": self.target_src_text,
                        "asr_time": round(asr_time, 3),
                        "separation_time": round(separation_time, 3),
                    }
                    self.results_queue.put(result)

        except Exception as e:
            print(f"Segment processing error: {e}")

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

    def _transcribe_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[str, float]:
        """语音识别"""
        try:
            asr_t0 = time.time()
            st = self.asr.create_stream()
            st.accept_waveform(sample_rate, audio)
            self.asr.decode_stream(st)
            text = st.result.text or ""
            asr_time = time.time() - asr_t0
            return text, asr_time
        except Exception as e:
            print(f"ASR error: {e}")
            return "", 0.0

    def get_results(self):
        """获取处理结果"""
        results = []
        while not self.results_queue.empty():
            results.append(self.results_queue.get())
        return results

    def flush_buffer(self):
        """强制处理缓冲区中剩余的数据"""
        # 将剩余的缓冲区数据发送给 VAD
        if len(self.audio_buffer) > 0:
            self.vad.accept_waveform(self.audio_buffer)
            self.audio_buffer = np.array([], dtype=np.float32)

        # 通知 VAD 输入结束
        self.vad.flush()

        # 清空 VAD 队列
        while not self.vad.empty():
            self.vad.pop()

        # 发射剩余的待处理音频
        if len(self._pending_audio) >= 0.2 * G_SAMPLE_RATE:
            self._emit_pending_segment()

    def wait_completion(self, timeout: float = None):
        """等待所有已提交的任务处理完成"""
        self.segment_queue.join()

    def shutdown(self):
        """关闭管道"""
        self._shutdown = True
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
