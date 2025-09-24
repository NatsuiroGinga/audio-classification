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
    backend: str | None = None  # must be "asteroid"
    device: str = "cpu"
    sample_rate: int = 16000
    checkpoint: Optional[str] = None  # optional path to pretrained weights

    def __post_init__(self):
        self._model = None
        # Force asteroid backend
        self.backend = "asteroid"
        self._init_asteroid()

    def _init_asteroid(self):
        try:
            import torch  # type: ignore
            from asteroid.models import ConvTasNet  # type: ignore

            self._model = ConvTasNet(n_src=2)
            ckpt_path = self._ensure_checkpoint()
            if ckpt_path:
                state = torch.load(ckpt_path, map_location=self.device)
                # Support both plain state_dict and wrapped dicts
                sd = state.get("state_dict", state)
                self._model.load_state_dict(sd, strict=False)
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize Asteroid ConvTasNet. Ensure 'asteroid' and 'torch' are installed, "
                "and provide a compatible checkpoint if required."
            ) from e

    def separate(self, samples: np.ndarray, sr: int) -> List[np.ndarray]:
        """Return two separated waveforms using Conv-TasNet (no fallback)."""
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
            est = self._model(x)  # (1, n_src=2, T)
        out = est.squeeze(0).cpu().numpy()
        if out.shape[0] < 2:
            raise RuntimeError("Separation output has < 2 sources; check model/config.")
        return [out[0], out[1]]

    def _ensure_sr(self, samples: np.ndarray, sr: int) -> np.ndarray:
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
        """Return a checkpoint path.

        Priority:
          1) If self.checkpoint is provided and exists, use it.
          2) Else, try to download a default public checkpoint from Hugging Face
             using huggingface_hub (provides its own integrity via etag caching).
             You can override defaults via env:
               ASTEROID_SEP_REPO_ID (default: 'mpariente/ConvTasNet_WHAM_sepclean')
               ASTEROID_SEP_FILENAME (default: 'pytorch_model.bin')
        """
        import os

        if self.checkpoint:
            if os.path.isfile(self.checkpoint):
                return self.checkpoint
            raise FileNotFoundError(
                f"Separator checkpoint not found: {self.checkpoint}"
            )

        # Auto-download via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to auto-download the default ConvTasNet checkpoint. "
                "Install it or provide --sep-checkpoint."
            ) from e

        repo_id = os.environ.get(
            "ASTEROID_SEP_REPO_ID", "mpariente/ConvTasNet_WHAM_sepclean"
        )
        filename = os.environ.get("ASTEROID_SEP_FILENAME", "pytorch_model.bin")
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return local_path
