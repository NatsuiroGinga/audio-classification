"""Overlap speech detection (OSD) utilities (pyannote.audio is REQUIRED).

Provides OverlapAnalyzer using pyannote.audio overlapped-speech-detection.
If pyannote.audio or its pretrained pipeline cannot be initialized, a RuntimeError
is raised. There is NO fallback path.

Usage:
        analyzer = OverlapAnalyzer(threshold=0.5)
        segments = analyzer.analyze(samples, sr)  # List[(start, end, is_overlap)]
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import numpy as np
import torch


@dataclass
class OverlapAnalyzer:
    """重叠语音检测器（强制使用 pyannote.audio 的 OSD pipeline）。

    参数：
    - threshold: 置信阈值（当前实现中由 pyannote pipeline 决定，保留接口）
    - win_sec / hop_sec: 将连续时间轴栅格化为帧（用于后处理 flags→segments）
    - device: 推理设备（"cpu"/"cuda"）
    - backend: 固定为 "pyannote"
    - auth_token: HF Token（如模型受限访问）
    """

    threshold: float = 0.5
    win_sec: float = 0.5
    hop_sec: float = 0.1
    device: str = "cpu"
    backend: str | None = None  # must be "pyannote" (required)
    auth_token: Optional[str] = None  # Optional HuggingFace token for private models

    def __post_init__(self):
        """初始化：
        - 强制设置后端为 pyannote
        - 从环境注入 HF Token（PYANNOTE_TOKEN / HF_TOKEN / HUGGINGFACE_TOKEN）
        - 构建 pyannote OverlappedSpeechDetection pipeline
        """
        self._pipeline = None
        # Force pyannote backend
        self.backend = "pyannote"
        # Auto-inject HF token from environment if not provided explicitly
        if not self.auth_token:
            self.auth_token = (
                os.environ.get("PYANNOTE_TOKEN")
                or os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
            )
        print(f"Using HF token: {'set' if self.auth_token else 'not set'}")
        self._init_pyannote()

    def _init_pyannote(self):
        """加载 pyannote 的 OverlappedSpeechDetection 预训练模型并移动到目标设备。"""
        try:
            from pyannote.audio.pipelines import OverlappedSpeechDetection

            # Try with/without token depending on availability/signature
            model_id = "pyannote/overlapped-speech-detection"
            if self.auth_token:
                self._pipeline = OverlappedSpeechDetection.from_pretrained(
                    model_id, use_auth_token=self.auth_token
                )
            else:
                self._pipeline = OverlappedSpeechDetection.from_pretrained(model_id)

            self._pipeline.to(torch.device(self.device))
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize pyannote OverlappedSpeechDetection. "
                "Please ensure pyannote.audio is installed and, if required, set a valid HuggingFace token."
            ) from e

    def analyze(self, samples: np.ndarray, sr: int) -> List[Tuple[float, float, bool]]:
        """Return segments within the utterance labeled as overlap or not.

        Returns a list of (start_sec, end_sec, is_overlap) covering the full utterance,
        with non-overlap and overlap intervals alternating without gaps.
        """
        dur = len(samples) / sr if sr else 0.0
        if dur <= 0:
            return []
        if self._pipeline is None:
            raise RuntimeError("OSD pipeline is not initialized.")
        import torch

        waveform = (
            torch.from_numpy(samples)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
        )  # (1, T)
        output = self._pipeline({"waveform": waveform, "sample_rate": sr})
        # output is a pyannote.core.Annotation with 'OVERLAP' label regions
        grid = np.arange(0, max(dur - self.win_sec, 0) + 1e-9, self.hop_sec)
        flags = np.zeros(len(grid), dtype=bool)
        # 遍历 pyannote 产出的时间段，打上窗口中心位置是否落在“重叠”区间内的标记
        for segment, _, label in output.itertracks(yield_label=True):
            if str(label).upper() != "OVERLAP":
                continue
            s, e = float(segment.start), float(segment.end)
            idx = np.where((grid >= s - self.win_sec / 2) & (grid <= e))[0]
            flags[idx] = True
        return self._flags_to_segments(flags, dur)

    def _flags_to_segments(
        self, flags: np.ndarray, dur: float
    ) -> List[Tuple[float, float, bool]]:
        """将布尔 flags（以 hop_sec 栅格）合并为连续时间段。

        - 合并相邻同标记的小段，消除 <50ms 的短间隙；
        - 截断到 [0, dur] 范围内并过滤空段。
        """
        segs: List[Tuple[float, float, bool]] = []
        if len(flags) == 0:
            # No overlap detected by pyannote -> all non-overlap
            return [(0.0, dur, False)]
        # Expand to sample timeline of windows
        cur_flag = flags[0]
        cur_start = 0.0
        for i in range(1, len(flags)):
            if flags[i] != cur_flag:
                segs.append(
                    (cur_start, i * self.hop_sec + self.win_sec, bool(cur_flag))
                )
                cur_flag = flags[i]
                cur_start = i * self.hop_sec
        # tail
        segs.append((cur_start, dur, bool(cur_flag)))
        # merge very small gaps
        merged: List[Tuple[float, float, bool]] = []
        for s, e, f in segs:
            if not merged:
                merged.append((s, e, f))
            else:
                ps, pe, pf = merged[-1]
                if f == pf and s - pe < 0.05:  # merge short gaps <50ms
                    merged[-1] = (ps, e, pf)
                else:
                    merged.append((s, e, f))
        # clip
        merged = [(max(0.0, s), min(dur, e), f) for s, e, f in merged if e > s]
        return merged
