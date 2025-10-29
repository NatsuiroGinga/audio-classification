"""Speech separation utilities for overlapped speech (REQUIRES asteroid).

Separator provides a simple API to perform 2-speaker separation using Asteroid
Conv-TasNet. If asteroid/torch is not available or model cannot be initialized,
it raises RuntimeError. There is NO fallback.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Separator:
    """说话人分离器（基于 Asteroid Conv-TasNet）。

    - backend 固定为 "asteroid"；
    - 若未提供 checkpoint，将从 Hugging Face 自动下载默认权重；
    - 输入/输出均为 numpy 波形；内部用 torch 推理。
    """

    backend: str | None = None  # must be "asteroid"
    device: str = "cpu"
    sample_rate: int = 16000
    checkpoint: Optional[str] = None  # optional path to pretrained weights
    n_src: int = 2  # 分离的路数（与 checkpoint 需一致）

    def __post_init__(self):
        """构造后初始化分离模型。"""
        self._model = None
        # Force asteroid backend
        self.backend = "asteroid"
        self._init_asteroid()

    def _init_asteroid(self):
        """加载 Conv-TasNet 模型与可选权重，并设置为 eval 模式。"""
        try:
            import torch  # type: ignore
            from asteroid.models import ConvTasNet  # type: ignore
            from asteroid.models import BaseModel

            if self.n_src == 3:
                self._model = BaseModel.from_pretrained(
                    "JorisCos/ConvTasNet_Libri3Mix_sepclean_16k"
                )
                print("Loaded pretrained ConvTasNet for 3 sources from Hugging Face.")
            else:
                # 根据 n_src 构建模型（注意需与权重训练时一致）
                self._model = ConvTasNet(n_src=self.n_src)
                ckpt_path = self._ensure_checkpoint()
                if ckpt_path:
                    state = torch.load(ckpt_path, map_location=self.device)
                    # Support both plain state_dict and wrapped dicts
                    sd = state.get("state_dict", state)
                    self._model.load_state_dict(sd, strict=False)
                    print(f"Loaded ConvTasNet weights from {ckpt_path}.")
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Asteroid ConvTasNet. Ensure 'asteroid' and 'torch' are installed, "
                "and provide a compatible checkpoint if required."
            ) from e

    def separate(self, samples: np.ndarray, sr: int) -> List[np.ndarray]:
        """执行 n_src 路语音分离。

        返回：长度为 n_src 的 List[np.ndarray]，每路与输入长度近似一致。
        若模型输出通道数小于 n_src，则抛出异常。
        """
        if self._model is None:
            raise RuntimeError("Separator model is not initialized.")
        import torch  # type: ignore

        wav = self._ensure_sr(samples, sr)
        x = (
            torch.from_numpy(wav)
            .to(device=self.device, dtype=torch.float32)
            .unsqueeze(0)
        )  # (1, T)
        with torch.no_grad():
            est = self._model(x)  # (1, n_src, T)
        out = est.squeeze(0).cpu().numpy()
        if out.shape[0] < self.n_src:
            raise RuntimeError(
                f"Separation output has < {self.n_src} sources; check model/config."
            )
        return [out[i] for i in range(self.n_src)]

    def _ensure_sr(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """若输入采样率与模型不一致，使用线性插值做简单重采样。"""
        if sr == self.sample_rate:
            return samples
        # simple linear resample
        if len(samples) <= 1:
            return samples
        tgt_n = int(round(len(samples) * self.sample_rate / sr))
        if tgt_n <= 1:
            return samples
        old_idx = np.arange(len(samples), dtype=np.float64)
        new_idx = np.linspace(0, len(samples) - 1, tgt_n, dtype=np.float64)
        return np.interp(new_idx, old_idx, samples).astype(np.float32)

    def _ensure_checkpoint(self) -> Optional[str]:
        """Return a checkpoint path (auto-download for n_src in {2, 3}).

        Priority:
          1) If self.checkpoint is provided and exists, use it.
          2) Else, auto-download a default public checkpoint from Hugging Face
             via huggingface_hub. Environment overrides are supported:
               - For 2 sources:
                   ASTEROID_SEP_REPO_ID_2 (fallback: ASTEROID_SEP_REPO_ID)
                   ASTEROID_SEP_FILENAME_2 (fallback: ASTEROID_SEP_FILENAME)
                   Defaults: repo='mpariente/ConvTasNet_WHAM_sepclean', file='pytorch_model.bin'
               - For 3 sources:
                   ASTEROID_SEP_REPO_ID_3
                   ASTEROID_SEP_FILENAME_3
                   Defaults: repo='JorisCos/ConvTasNet_Libri3Mix_sepclean_16k', file='pytorch_model.bin'
          3) For other n_src values, a compatible checkpoint must be provided.
        """
        import os

        if self.checkpoint:
            if os.path.isfile(self.checkpoint):
                return self.checkpoint
            raise FileNotFoundError(
                f"Separator checkpoint not found: {self.checkpoint}"
            )

        # Auto-download via huggingface_hub（支持常见的 2/3 路权重）
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to auto-download the default ConvTasNet checkpoint. "
                "Install it or provide --sep-checkpoint."
            ) from e

        nsrc = int(self.n_src or 2)
        if nsrc == 2:
            repo_id = os.environ.get(
                "ASTEROID_SEP_REPO_ID_2",
                os.environ.get(
                    "ASTEROID_SEP_REPO_ID", "mpariente/ConvTasNet_WHAM_sepclean"
                ),
            )
            filename = os.environ.get(
                "ASTEROID_SEP_FILENAME_2",
                os.environ.get("ASTEROID_SEP_FILENAME", "pytorch_model.bin"),
            )
            return hf_hub_download(repo_id=repo_id, filename=filename)
        elif nsrc == 3:
            repo_id = os.environ.get(
                "ASTEROID_SEP_REPO_ID_3", "JorisCos/ConvTasNet_Libri3Mix_sepclean_16k"
            )
            filename = os.environ.get("ASTEROID_SEP_FILENAME_3", "pytorch_model.bin")
            return hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise RuntimeError(
                "Auto-download supports only n_src in {2, 3}. "
                "Please provide a compatible checkpoint via --sep-checkpoint for other n_src values."
            )
