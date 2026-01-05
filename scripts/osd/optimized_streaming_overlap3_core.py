#!/usr/bin/env python3
"""
优化版 StreamingOverlap3Pipeline - 基于对比分析的改进

改进点：
1. 移除了冗余的 OSD-based 处理路径（_process_overlap_segment 等）
2. 采用单一的 direct separation 处理流程
3. 保留可选的 OSD 分析用于监控和可视化
4. 性能提升和代码简化
"""
import time
import numpy as np
import threading
from queue import Queue
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from overlap3_core import (
    _build_asr,
    _provider_to_torch_device,
    _ensure_sr_np,
    l2norm,
    G_SAMPLE_RATE,
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
    seq_id: int = 0  # 序列号，用于保证顺序


class OptimizedStreamingOverlap3Pipeline:
    """优化版流式重叠语音处理管道

    核心改进：
    - 采用直接分离方法（不依赖OSD）
    - 移除OSD-based的多路径分支
    - 简化为单一清晰的处理流程
    """

    def __init__(self, args, target_wav_path: str):
        self.args = args
        self.device = _provider_to_torch_device(args.provider)

        # 初始化ASR和分离组件
        self.asr = _build_asr(args)
        self.sep = Separator(
            backend=args.sep_backend or "asteroid",
            checkpoint=(args.sep_checkpoint or None),
            device=self.device,
            n_src=3,
        )

        # Speaker embedding extractor
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

        # 可选：OSD 用于监控和调试，不作为处理的控制信号
        self.osd = None
        if getattr(args, "enable_osd_monitoring", False):
            self.osd = OverlapAnalyzer(
                threshold=args.osd_thr,
                win_sec=args.osd_win,
                hop_sec=args.osd_hop,
                backend=args.osd_backend or "pyannote",
                device=self.device,
            )

        # 流式处理状态
        self.audio_buffer = []
        self.processing = False
        self.segment_queue = Queue()  # 输入任务队列
        self.results_queue = Queue()  # 输出结果队列
        self._segment_seq = 0  # 段序列号，保证顺序
        self._shutdown = False  # 关闭标志

        # 启动单线程工作器，保证处理顺序与输入顺序一致
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="StreamingWorker"
        )
        self._worker_thread.start()

        # 加载目标说话人
        self._load_target_speaker(target_wav_path)

    def _load_target_speaker(self, target_wav_path: str):
        """加载目标说话人语音和文本"""
        import torchaudio

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

        # 创建处理任务（带序列号）
        self._segment_seq += 1
        segment = StreamingSegment(
            audio_data=audio_data,
            start_time=current_time - chunk_duration,
            end_time=current_time,
            sample_rate=self.args.sample_rate,
            seq_id=self._segment_seq,
        )

        # 放入任务队列，由worker线程顺序处理
        self.segment_queue.put(segment)

    def _worker_loop(self):
        """工作线程循环 - 顺序处理任务队列中的segment

        使用单线程保证：
        1. 处理顺序与输入顺序一致
        2. 结果输出顺序与输入顺序一致
        """
        while not self._shutdown:
            try:
                # 阻塞等待任务，超时1秒检查shutdown标志
                segment = self.segment_queue.get(timeout=1.0)
                self._process_segment(segment)
                self.segment_queue.task_done()
            except Exception:
                # Queue.Empty 或其他异常，继续循环
                pass

    def _process_segment(self, segment: StreamingSegment):
        """处理单个音频段（优化版）

        核心流程：
        1. 对整个混合音频进行3源分离
        2. 对每个分离源进行说话人验证
        3. 选择匹配目标说话人的源进行ASR识别

        不依赖OSD检测结果控制处理流程。
        """
        try:
            audio_data = segment.audio_data
            sample_rate = segment.sample_rate

            # ===== 可选：OSD监控（不控制处理流程）=====
            osd_segments = None
            if self.osd is not None:
                try:
                    osd_segments = self.osd.analyze(audio_data, sample_rate)
                    # 可用于统计或可视化，但不影响处理
                except Exception as e:
                    pass

            # ===== 核心处理：直接分离3源 =====
            separation_start = time.time()
            separated_streams = self.sep.separate(audio_data, sample_rate)
            separation_time = time.time() - separation_start

            # ===== 说话人验证和识别 =====
            for stream_id, stream_audio in enumerate(separated_streams):
                # 说话人验证
                sv_score, matched = self._speaker_verification(
                    stream_audio, sample_rate
                )

                # 仅处理匹配目标说话人的流
                if matched:
                    asr_start = time.time()
                    text, _ = self._transcribe_audio(stream_audio, sample_rate)
                    asr_time = time.time() - asr_start

                    result = {
                        # 序列号（保证顺序）
                        "seq_id": segment.seq_id,
                        # 时间信息
                        "start": round(segment.start_time, 3),
                        "end": round(segment.end_time, 3),
                        # 处理标记（已统一为 'separated'）
                        "kind": "separated",
                        "stream": stream_id,
                        # 识别结果
                        "text": text,
                        "sv_score": (
                            round(sv_score, 4) if sv_score is not None else None
                        ),
                        "target_src_text": self.target_src_text,
                        # 性能指标
                        "asr_time": round(asr_time, 3),
                        "separation_time": round(separation_time, 3),
                        # OSD 监控信息（可选）
                        "osd_segments": osd_segments if self.osd is not None else None,
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
        if self.audio_buffer:
            self._process_audio_chunk()

    def wait_completion(self, timeout: float = None):
        """等待所有已提交的任务处理完成

        Args:
            timeout: 超时时间（秒），None表示无限等待
        """
        self.segment_queue.join()

    def shutdown(self):
        """关闭管道，停止工作线程"""
        self._shutdown = True
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
