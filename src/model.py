"""Model definitions for Speaker + ASR pipeline.

Contains SpeakerASRModels which encapsulates:
- ASR model loading (paraformer | sense-voice | transducer)
- Speaker embedding extractor & enrollment
- Identification (threshold search + top1 cosine score)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import json
import numpy as np
import sherpa_onnx

try:  # Optional torch usage for accepting GPU tensors
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    _TORCH_AVAILABLE = False

# Reuse global sample rate; fallback if not imported externally
G_SAMPLE_RATE = 16000


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def create_asr_model(
    *,
    paraformer: str,
    sense_voice: str,
    encoder: str,
    decoder: str,
    joiner: str,
    tokens: str,
    num_threads: int,
    feature_dim: int,
    decoding_method: str,
    debug: bool,
    language: str,
    provider: str,
) -> sherpa_onnx.OfflineRecognizer:
    if paraformer:
        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=paraformer,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=G_SAMPLE_RATE,
            feature_dim=feature_dim,
            decoding_method=decoding_method,
            debug=debug,
        )
    if sense_voice:
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=sense_voice,
            tokens=tokens,
            num_threads=num_threads,
            use_itn=True,
            debug=debug,
            language=language,
        )
    if encoder:
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=num_threads,
            sample_rate=G_SAMPLE_RATE,
            feature_dim=feature_dim,
            decoding_method=decoding_method,
            debug=debug,
        )
    raise ValueError("Provide one ASR model (paraformer | sense_voice | transducer)")


def create_extractor_model(
    *, model: str, num_threads: int, provider: str, debug: bool
) -> sherpa_onnx.SpeakerEmbeddingExtractor:
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model,
        num_threads=num_threads,
        debug=debug,
        provider=provider,
    )
    if not cfg.validate():
        raise ValueError("Invalid speaker embedding config")
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


class SpeakerASRModels:
    """Encapsulates ASR + Speaker embedding models and operations."""

    def __init__(self, args):
        self.args = args
        self.provider = getattr(args, 'provider', 'cpu')
        self.using_cuda = 'cuda' in self.provider.lower() or 'gpu' in self.provider.lower()
        self.asr = create_asr_model(
            paraformer=getattr(args, "paraformer", ""),
            sense_voice=getattr(args, "sense_voice", ""),
            encoder=getattr(args, "encoder", ""),
            decoder=getattr(args, "decoder", ""),
            joiner=getattr(args, "joiner", ""),
            tokens=getattr(args, "tokens", ""),
            num_threads=args.num_threads,
            feature_dim=getattr(args, "feature_dim", 80),
            decoding_method=getattr(args, "decoding_method", "greedy_search"),
            debug=getattr(args, "debug", False),
            language=getattr(args, "language", "auto"),
            provider=args.provider,
        )
        self.extractor = create_extractor_model(
            model=args.model,
            num_threads=args.num_threads,
            provider=args.provider,
            debug=getattr(args, "debug", False),
        )
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.enrolled: Dict[str, np.ndarray] = {}
        self.enrolled_norm: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers for tensor/array handling
    # ------------------------------------------------------------------
    def _to_numpy_waveform(self, samples) -> np.ndarray:
        """Accept np.ndarray or torch.Tensor (any device) and return float32 1-D numpy array.

        If a torch CUDA tensor is given, it is moved to CPU (non_blocking if possible).
        This is the narrowest integration point before handing audio to sherpa_onnx.
        NOTE: sherpa_onnx Streaming/Offline APIs expect CPU numpy arrays; ONNX Runtime
        will handle host->device transfer internally if provider is CUDA.
        """
        if isinstance(samples, np.ndarray):
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32, copy=False)
            return samples
        if _TORCH_AVAILABLE and isinstance(samples, torch.Tensor):  # type: ignore
            if samples.ndim > 1:
                samples = samples.view(-1)
            # dtype normalize
            if samples.dtype != getattr(torch, 'float32'):
                samples = samples.float()
            if samples.device.type != 'cpu':
                samples = samples.detach().to('cpu', non_blocking=True)
            return samples.numpy()
        # Fallback: try to coerce
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        return arr

    # ------------------------------------------------------------------
    # Introspection / debug representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - debug utility
        args = self.args
        # 判定 ASR 类型
        if getattr(args, 'paraformer', ''):
            asr_type = 'paraformer'
            asr_model_path = getattr(args, 'paraformer')
        elif getattr(args, 'sense_voice', ''):
            asr_type = 'sense_voice'
            asr_model_path = getattr(args, 'sense_voice')
        elif getattr(args, 'encoder', ''):
            asr_type = 'transducer'
            asr_model_path = f"encoder={getattr(args,'encoder')}|decoder={getattr(args,'decoder')}|joiner={getattr(args,'joiner')}"
        else:
            asr_type = 'unknown'
            asr_model_path = ''
        provider = getattr(args, 'provider', 'cpu')
        using_gpu = 'cuda' in provider.lower()
        emb_model = getattr(args, 'model', '')
        num_spk = len(self.enrolled)
        dim = getattr(self.extractor, 'dim', None)
        cache_dir = getattr(args, 'emb_cache_dir', '') or 'N/A'
        loaded_precomputed = bool(getattr(args, 'load_speaker_embeds', ''))
        lines = [
            'SpeakerASRModels(',
            f"  asr_type='{asr_type}',",
            f"  asr_model='{asr_model_path}',",
            f"  speaker_embedding_model='{emb_model}',",
            f"  embedding_dim={dim},",
            f"  provider='{provider}', using_gpu={using_gpu},",
            f"  enrolled_speakers={num_spk},",
            f"  cache_dir='{cache_dir}',",
            f"  loaded_precomputed_embeds={loaded_precomputed},",
            f"  threshold={getattr(args,'threshold', None)},",
            f"  num_threads={getattr(args,'num_threads', None)}",
            ')'
        ]
        return '\n'.join(lines)

    def enroll_from_map(self, spk_map: Dict[str, List[str]], load_audio_func):
        """Enroll speakers, with optional per-wav embedding cache & precomputed speaker loads.

        Args expects these optional attributes on self.args:
            emb_cache_dir: str, directory to cache per-wav embeddings (*.npy)
            load_speaker_embeds: path to npz of {spk: vector}
            save_speaker_embeds: path to save aggregated embeddings
        """
        # If loading precomputed speaker embeddings
        load_npz = getattr(self.args, 'load_speaker_embeds', '')
        if load_npz:
            data = np.load(load_npz, allow_pickle=True)
            for spk in data.files:
                vec = data[spk].astype(np.float32)
                self.enrolled[spk] = vec
                self.enrolled_norm[spk] = l2norm(vec)
                if not self.manager.add(spk, vec):
                    raise RuntimeError(f"Failed to add speaker {spk} from preloaded embeds")
            return

        cache_dir = getattr(self.args, 'emb_cache_dir', '')
        use_cache = bool(cache_dir)
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)

        speaker_means: Dict[str, np.ndarray] = {}

        for spk, wavs in spk_map.items():
            if not wavs:
                continue
            acc: Optional[np.ndarray] = None
            for w in wavs:
                emb: Optional[np.ndarray] = None
                cache_path = None
                if use_cache:
                    base = os.path.splitext(os.path.basename(w))[0]
                    cache_path = os.path.join(cache_dir, base + '.npy')
                    if os.path.isfile(cache_path):
                        try:
                            emb = np.load(cache_path)
                        except Exception:
                            emb = None
                if emb is None:
                    # Support load_audio returning either (samples, sr) or (samples, sr, ...)
                    loaded = load_audio_func(w)
                    if isinstance(loaded, tuple):
                        if len(loaded) >= 2:
                            samples, sr = loaded[0], loaded[1]
                        elif len(loaded) == 1:
                            # Only samples provided
                            samples, sr = loaded[0], G_SAMPLE_RATE
                        else:
                            raise ValueError(f"load_audio returned empty tuple for {w}")
                    else:
                        # Non-tuple: assume it's the samples and use default SR
                        samples, sr = loaded, G_SAMPLE_RATE
                    s = self.extractor.create_stream()
                    s.accept_waveform(sr, samples)
                    s.input_finished()
                    assert self.extractor.is_ready(s)
                    emb = np.array(self.extractor.compute(s), dtype=np.float32)
                    emb = l2norm(emb)
                    if cache_path:
                        try:
                            np.save(cache_path, emb)
                        except Exception:
                            pass
                else:
                    emb = l2norm(emb.astype(np.float32))
                acc = emb if acc is None else acc + emb
            if acc is None:
                raise RuntimeError(f"No embeddings for speaker {spk}")
            mean_emb = (acc / float(len(wavs))).astype(np.float32)
            speaker_means[spk] = mean_emb
            self.enrolled[spk] = mean_emb
            self.enrolled_norm[spk] = l2norm(mean_emb)
            if not self.manager.add(spk, mean_emb):
                raise RuntimeError(f"Failed to add speaker {spk}")

        save_npz = getattr(self.args, 'save_speaker_embeds', '')
        if save_npz:
            try:
                np.savez_compressed(save_npz, **speaker_means)
            except Exception:
                pass

    def identify(self, samples, sr: int, threshold: float) -> Tuple[str, float]:
        samples_np = self._to_numpy_waveform(samples)
        s = self.extractor.create_stream()
        s.accept_waveform(sr, samples_np)
        s.input_finished()
        assert self.extractor.is_ready(s)
        emb = np.array(self.extractor.compute(s), dtype=np.float32)
        emb_n = l2norm(emb)
        pred = self.manager.search(emb, threshold=threshold)
        if not pred:
            pred = "unknown"
        if self.enrolled_norm:
            names = list(self.enrolled_norm.keys())
            mat = np.stack([self.enrolled_norm[n] for n in names], axis=0)
            scores = mat @ emb_n
            top1_score = float(scores[np.argmax(scores)])
        else:
            top1_score = float("nan")
        return pred, top1_score

    def asr_infer(self, samples, sr: int) -> str:
        samples_np = self._to_numpy_waveform(samples)
        st = self.asr.create_stream()
        st.accept_waveform(sr, samples_np)
        self.asr.decode_stream(st)
        return st.result.text
