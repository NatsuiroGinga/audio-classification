#!/usr/bin/env python3
"""
Streaming version of Overlap3Pipeline for real-time processing
"""
import time
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from overlap3_core import (
    Overlap3Pipeline,
    _build_asr,
    _provider_to_torch_device,
    l2norm,
)
from src.osd import OverlapAnalyzer
from src.osd.separation import Separator


@dataclass
class StreamingSegment:
    """流式处理中的音频段"""

    audio_data: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int
    is_overlap: bool = False
    stream_id: Optional[int] = None


class StreamingOverlap3Pipeline:
    """流式重叠语音处理管道"""

    def __init__(self, args, target_wav_path: str):
        self.args = args
        self.device = _provider_to_torch_device(args.provider)

        # 初始化组件
        self.asr = _build_asr(args)
        self.osd = OverlapAnalyzer(
            threshold=args.osd_thr,
            win_sec=args.osd_win,
            hop_sec=args.osd_hop,
            backend=args.osd_backend or "pyannote",
            device=self.device,
        )
        self.sep = Separator(
            backend=args.sep_backend or "asteroid",
            checkpoint=(args.sep_checkpoint or None),
            device=self.device,
            n_src=3,
        )
        # Speaker embedding extractor（与 offline pipeline 一致）
        try:
            import sherpa_onnx  # type: ignore
        except Exception:
            sherpa_onnx = None
        if sherpa_onnx is None:
            raise RuntimeError(
                "sherpa_onnx is required for speaker embedding in streaming mode"
            )
        se_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.spk_embed_model,
            num_threads=getattr(args, "num_threads", 1),
            debug=getattr(args, "debug", False),
            provider=args.provider,
        )
        if not se_config.validate():
            raise ValueError(f"Invalid speaker embedding config: {se_config}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(se_config)

        # 流式处理状态
        self.audio_buffer = []
        self.processing = False
        self.segment_queue = Queue()
        self.results_queue = Queue()

        # 加载目标说话人
        self._load_target_speaker(target_wav_path)

    def _load_target_speaker(self, target_wav_path: str):
        """加载目标说话人语音"""
        import torchaudio
        from overlap3_core import _ensure_sr_np, G_SAMPLE_RATE, l2norm

        # 加载目标语音文件
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

        # 为目标语音创建ASR流获取文本
        st_tgt = self.asr.create_stream()
        st_tgt.accept_waveform(G_SAMPLE_RATE, t_np)
        self.asr.decode_stream(st_tgt)
        self.target_src_text = st_tgt.result.text or ""
        print(f"Target speaker enrolled. Text: '{self.target_src_text}'")

    def add_audio_data(self, audio_chunk: np.ndarray):
        """添加音频数据到缓冲区"""
        self.audio_buffer.append(audio_chunk)
        self._process_audio_chunk()

    def _process_audio_chunk(self):
        """处理音频块"""
        if not self.audio_buffer:
            return

        # 合并缓冲区数据
        audio_data = np.concatenate(self.audio_buffer)
        current_time = time.time()
        chunk_duration = len(audio_data) / self.args.sample_rate

        # 清空缓冲区
        self.audio_buffer = []

        # 创建处理任务
        segment = StreamingSegment(
            audio_data=audio_data,
            start_time=current_time - chunk_duration,
            end_time=current_time,
            sample_rate=self.args.sample_rate,
        )

        # 异步处理
        threading.Thread(
            target=self._analyze_segment, args=(segment,), daemon=True
        ).start()

    def _analyze_segment(self, segment: StreamingSegment):
        """分析单个音频段"""
        try:
            # OSD检测
            osd_segments = self.osd.analyze(segment.audio_data, segment.sample_rate)

            if not osd_segments:
                # 没有检测到重叠，按清洁段处理
                self._process_clean_segment(segment, 0, len(segment.audio_data))
            else:
                # 处理检测到的段
                for start, end, is_overlap in osd_segments:
                    start_idx = int(start * segment.sample_rate)
                    end_idx = int(end * segment.sample_rate)
                    sub_audio = segment.audio_data[start_idx:end_idx]

                    if is_overlap and (end - start) >= self.args.min_overlap_dur:
                        # 重叠段处理
                        self._process_overlap_segment(
                            segment, start_idx, end_idx, sub_audio
                        )
                    else:
                        # 清洁段处理
                        self._process_clean_segment(
                            segment, start_idx, end_idx, sub_audio
                        )

            # 对整个混合音频进行直接分离处理（不依赖 OSD 检测结果）
            self._process_full_separation(segment)

        except Exception as e:
            print(f"Segment analysis error: {e}")

    def _process_full_separation(self, segment: StreamingSegment):
        """对整个混合音频进行声音分离 → 说话人识别 → ASR（不经过 OSD）"""
        try:
            audio_data = segment.audio_data
            sample_rate = segment.sample_rate

            # 对整个音频进行声音分离
            separated_streams = self.sep.separate(audio_data, sample_rate)

            # 对每个分离的流进行说话人验证和 ASR
            for stream_id, stream_audio in enumerate(separated_streams):
                sv_score, matched = self._speaker_verification(
                    stream_audio, sample_rate
                )

                if matched:
                    text, asr_time = self._transcribe_audio(stream_audio, sample_rate)

                    result = {
                        "start": segment.start_time,
                        "end": segment.end_time,
                        "kind": "full_separation",  # 标记为整体分离结果
                        "stream": stream_id,
                        "text": text,
                        "asr_time": asr_time,
                        "sv_score": sv_score,
                        "target_src_text": self.target_src_text,
                    }
                    self.results_queue.put(result)

        except Exception as e:
            print(f"Full separation error: {e}")

    def _process_clean_segment(
        self, segment: StreamingSegment, start_idx: int, end_idx: int, sub_audio=None
    ):
        """处理清洁段"""
        if sub_audio is None:
            sub_audio = segment.audio_data[start_idx:end_idx]

        # 说话人验证
        sv_score, matched = self._speaker_verification(sub_audio, segment.sample_rate)

        if matched:
            # ASR识别
            text, asr_time = self._transcribe_audio(sub_audio, segment.sample_rate)

            result = {
                "start": segment.start_time + (start_idx / segment.sample_rate),
                "end": segment.start_time + (end_idx / segment.sample_rate),
                "kind": "clean",
                "stream": None,
                "text": text,
                "asr_time": asr_time,
                "sv_score": sv_score,
                "target_src_text": self.target_src_text,
            }
            self.results_queue.put(result)

    def _process_overlap_segment(
        self,
        segment: StreamingSegment,
        start_idx: int,
        end_idx: int,
        sub_audio: np.ndarray,
    ):
        """处理重叠段"""
        # 语音分离
        separated_streams = self.sep.separate(sub_audio, segment.sample_rate)

        # 对每个分离的流进行说话人验证和ASR
        for stream_id, stream_audio in enumerate(separated_streams):
            sv_score, matched = self._speaker_verification(
                stream_audio, segment.sample_rate
            )

            if matched:
                text, asr_time = self._transcribe_audio(
                    stream_audio, segment.sample_rate
                )

                result = {
                    "start": segment.start_time + (start_idx / segment.sample_rate),
                    "end": segment.start_time + (end_idx / segment.sample_rate),
                    "kind": "overlap",
                    "stream": stream_id,
                    "text": text,
                    "asr_time": asr_time,
                    "sv_score": sv_score,
                    "target_src_text": self.target_src_text,
                }
                self.results_queue.put(result)

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
            text = st.result.text
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
        if self.audio_buffer:
            self._process_audio_chunk()
