#!/usr/bin/env python3
"""
Core logic for Offline OSD + 3-source Separation + ASR.

This module encapsulates the compute-only pipeline so that the caller
can handle all file I/O (writing JSON/CSV/metrics) separately. Timing
and resource metrics reported here exclude any caller-side file writes.
"""
from __future__ import annotations

import csv
import os
import random
import threading
import time
from dataclasses import dataclass
from statistics import mean
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import torchaudio
    import torchaudio.functional as AF
    from torchaudio.datasets import LibriMix
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore
    AF = None  # type: ignore
    LibriMix = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

try:
    import sherpa_onnx  # type: ignore
except Exception:  # pragma: no cover
    sherpa_onnx = None  # type: ignore

from src.model import create_asr_model, l2norm, G_SAMPLE_RATE
from src.osd import OverlapAnalyzer
from src.osd.separation import Separator


def _si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    if reference.shape != estimation.shape:
        n = min(reference.shape[-1], estimation.shape[-1])
        reference = reference[..., :n]
        estimation = estimation[..., :n]
    ref = reference.astype(np.float32) - float(np.mean(reference))
    est = estimation.astype(np.float32) - float(np.mean(estimation))
    ref_energy = float(np.sum(ref**2)) + 1e-12
    if ref_energy <= 0:
        return float("nan")
    scale = float(np.dot(est, ref)) / ref_energy
    proj = scale * ref
    e_noise = est - proj
    num = float(np.sum(proj**2)) + 1e-12
    den = float(np.sum(e_noise**2)) + 1e-12
    return 10.0 * float(np.log10(num / den))


def _pit_best_si_sdr_k(
    refs: List[np.ndarray], preds: List[np.ndarray]
) -> Tuple[float, List[int]]:
    import itertools

    K = len(refs)
    if K not in (2, 3):
        raise ValueError("_pit_best_si_sdr_k supports K=2 or 3")
    if len(preds) < K:
        return float("nan"), []

    sdr_mat = [[_si_sdr(refs[i], p) for p in preds] for i in range(K)]
    N = len(preds)
    best = -1e9
    best_idx: List[int] = []
    for cols in itertools.combinations(range(N), K):
        for perm in itertools.permutations(range(K), K):
            s = 0.0
            valid = True
            for r_i, c_i in enumerate(cols):
                sdr_val = sdr_mat[perm[r_i]][c_i]
                if np.isnan(sdr_val):
                    valid = False
                    break
                s += sdr_val
            if not valid:
                continue
            mean_sdr = s / float(K)
            if mean_sdr > best:
                best = mean_sdr
                assigned = [cols[perm.index(i)] for i in range(K)]
                best_idx = list(assigned)
    if best_idx == []:
        return float("nan"), []
    return float(best), best_idx


def _compute_sdr_improvement_pit_k(
    mix_chunk: np.ndarray, refs: List[np.ndarray], preds: List[np.ndarray]
) -> Tuple[float, float, List[int]]:
    if len(refs) not in (2, 3):
        return float("nan"), float("nan"), []
    base_vals = []
    for r in refs:
        base_vals.append(_si_sdr(r, mix_chunk))
    if any(np.isnan(x) for x in base_vals):
        return float("nan"), float("nan"), []
    base = float(np.mean(base_vals))
    best, indices = _pit_best_si_sdr_k(refs, preds)
    if np.isnan(best):
        return float("nan"), float("nan"), []
    return float(best), float(best - base), indices


def _provider_to_torch_device(provider: str) -> str:
    p = (provider or "").lower()
    if "cuda" in p or "gpu" in p:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _to_mono_float(wav: torch.Tensor) -> torch.Tensor:
    if wav.dim() == 2:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav[0]
    return wav.float().contiguous()


def _ensure_sr_np(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if AF is None:
        raise RuntimeError("torchaudio.functional is required")
    wav = _to_mono_float(wav)
    if sr != target_sr and wav.numel() > 1:
        wav = AF.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    return wav.cpu().numpy(), sr


def _build_asr(args):
    return create_asr_model(
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


@dataclass
class PipelineResult:
    segments: List[Dict[str, Any]]
    sep_details_rows: List[List[Any]]
    metrics: Dict[str, Any]
    dataset_name: str
    subset: str
    processed_mixtures: int
    sample_rate: int


class Overlap3Pipeline:
    """Compute-only pipeline: OSD → separation → target gating → ASR.

    Notes:
      - Handles both LibriMix and file-mode.
      - Returns all per-segment records and metrics. Caller is responsible for writing files.
      - The reported wall-clock excludes caller-side file writes.
    """

    def __init__(self, args):
        if torchaudio is None:
            raise RuntimeError(
                "torchaudio is required. Please install torchaudio matching your torch version."
            )
        # Optional seed
        try:
            if getattr(args, "seed", -1) is not None and int(args.seed) >= 0:
                random.seed(int(args.seed))
                np.random.seed(int(args.seed))
                try:
                    torch.manual_seed(int(args.seed))
                except Exception:
                    pass
        except Exception:
            pass
        self.args = args
        self.device = _provider_to_torch_device(args.provider)

        # Build components
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

        if sherpa_onnx is None:
            raise RuntimeError(
                "sherpa_onnx is required for speaker embedding. Please install sherpa_onnx."
            )
        se_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=args.spk_embed_model,
            num_threads=args.num_threads,
            debug=getattr(args, "debug", False),
            provider=args.provider,
        )
        if not se_config.validate():
            raise ValueError(f"Invalid speaker embedding config: {se_config}")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(se_config)

    def _resource_monitor(self, interval: float):
        class _Mon:
            def __init__(self, itv: float):
                self.interval = max(0.1, itv)
                self.samples: List[dict] = []
                self._stop = threading.Event()
                self._thread: Optional[threading.Thread] = None
                self._proc = psutil.Process(os.getpid()) if psutil else None

            def _gpu_info(self):
                if torch.cuda.is_available():
                    try:
                        return {
                            "gpu_mem_allocated": torch.cuda.memory_allocated()
                            / (1024**2),
                            "gpu_mem_reserved": torch.cuda.memory_reserved()
                            / (1024**2),
                            "gpu_max_mem_allocated": torch.cuda.max_memory_allocated()
                            / (1024**2),
                        }
                    except Exception:
                        return {}
                return {}

            def _loop(self):
                if self._proc:
                    self._proc.cpu_percent(interval=None)
                while not self._stop.wait(self.interval):
                    if not self._proc:
                        break
                    try:
                        cpu = self._proc.cpu_percent(interval=None)
                        mem_info = self._proc.memory_info()
                        rss_mb = mem_info.rss / (1024**2)
                        rec = {"cpu": cpu, "rss_mb": rss_mb}
                        rec.update(self._gpu_info())
                        self.samples.append(rec)
                    except Exception:
                        break

            def start(self):
                if self._proc is None:
                    return
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

            def stop(self):
                if self._proc is None:
                    return
                self._stop.set()
                if self._thread:
                    self._thread.join(timeout=2)

            def aggregate(self):
                if not self.samples:
                    return {}
                cpu_list = [s["cpu"] for s in self.samples if "cpu" in s]
                rss_list = [s["rss_mb"] for s in self.samples if "rss_mb" in s]
                gpu_alloc = [s.get("gpu_mem_allocated", 0.0) for s in self.samples]
                gpu_res = [s.get("gpu_mem_reserved", 0.0) for s in self.samples]
                return {
                    "cpu_avg": round(mean(cpu_list), 2) if cpu_list else None,
                    "cpu_peak": round(max(cpu_list), 2) if cpu_list else None,
                    "rss_avg_mb": round(mean(rss_list), 2) if rss_list else None,
                    "rss_peak_mb": round(max(rss_list), 2) if rss_list else None,
                    "gpu_mem_allocated_avg_mb": (
                        round(mean(gpu_alloc), 2) if gpu_alloc else None
                    ),
                    "gpu_mem_allocated_peak_mb": (
                        round(max(gpu_alloc), 2) if gpu_alloc else None
                    ),
                    "gpu_mem_reserved_peak_mb": (
                        round(max(gpu_res), 2) if gpu_res else None
                    ),
                }

        return _Mon(interval)

    def run(self) -> PipelineResult:
        args = self.args
        # Assure optional deps are available at runtime
        assert torchaudio is not None, "torchaudio must be available at run time"
        assert sherpa_onnx is not None, "sherpa_onnx must be available at run time"

        # Determine mode and load dataset or files
        file_mode = bool(getattr(args, "input_wavs", None))
        ds = None
        items: List[
            Tuple[int, torch.Tensor, Optional[List[torch.Tensor]], str, List[str]]
        ] = []
        dataset_name = "LibriMix"
        if not file_mode:
            if LibriMix is None:
                raise RuntimeError(
                    "torchaudio.datasets.LibriMix is required in dataset mode. Install or use --input-wavs."
                )
            ds = LibriMix(
                root=args.librimix_root,
                subset=args.subset,
                num_speakers=3,
                sample_rate=args.sample_rate,
                task=args.task,
                mode=args.mode,
            )
            total = len(ds)
            limit = args.max_files if args.max_files and args.max_files > 0 else total
        else:
            dataset_name = "manual-files"
            if not getattr(args, "target_wav", ""):
                raise ValueError(
                    "In file mode (--input-wavs), --target-wav is required."
                )
            in_paths = [str(Path(p)) for p in (args.input_wavs or [])]
            for p in in_paths:
                if not Path(p).is_file():
                    continue
                w, sr_item = torchaudio.load(p)
                items.append((sr_item, w, None, p, []))
            total = len(items)
            limit = total

        # Metrics accumulators
        n_segments = 0
        n_clean_segments = 0
        n_overlap_segments = 0
        n_separated_streams = 0
        n_matched_segments = 0
        n_seen_clean_segments = 0
        n_seen_overlap_segments = 0
        n_missed_segments = 0
        n_missed_clean_segments = 0
        n_missed_overlap_segments = 0
        total_audio_sec = 0.0
        total_overlap_audio_sec = 0.0
        total_clean_audio_sec = 0.0
        total_matched_audio_sec = 0.0
        total_seen_clean_audio_sec = 0.0
        total_seen_overlap_audio_sec = 0.0
        total_missed_audio_sec = 0.0
        time_osd = 0.0
        time_sep = 0.0
        time_asr = 0.0

        sep_eval_enabled = getattr(args, "eval_separation", False)
        sep_sisdr_list: List[float] = []
        sep_sisdri_list: List[float] = []
        sep_details_rows: List[List[Any]] = []

        monitor = None
        if args.enable_metrics and psutil is not None:
            monitor = self._resource_monitor(args.monitor_interval)
            monitor.start()

        t0_all = time.time()

        # Target enrollment (file mode)
        extractor = self.extractor
        asr = self.asr

        manager: Any = None
        enrolled_vec_norm = None
        global_target_src_abs = None
        global_target_src_text = ""
        global_target_src_np: Optional[np.ndarray] = None
        if file_mode:
            manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

            def _compute_emb_from_np(wav_np: np.ndarray, sr: int) -> np.ndarray:
                s = extractor.create_stream()
                s.accept_waveform(sr, wav_np)
                s.input_finished()
                assert extractor.is_ready(s)
                emb = np.array(extractor.compute(s), dtype=np.float32)
                return l2norm(emb)

            tgt_p = str(Path(args.target_wav))
            t_wav, t_sr = torchaudio.load(tgt_p)
            t_np, _ = _ensure_sr_np(t_wav, int(t_sr), G_SAMPLE_RATE)
            enrolled_vec = _compute_emb_from_np(t_np, G_SAMPLE_RATE)
            enrolled_vec_norm = l2norm(enrolled_vec)
            _ = manager.add("target", enrolled_vec)
            global_target_src_abs = tgt_p
            global_target_src_np = t_np
            try:
                st_tgt = asr.create_stream()
                st_tgt.accept_waveform(G_SAMPLE_RATE, t_np)
                asr.decode_stream(st_tgt)
                global_target_src_text = st_tgt.result.text or ""
            except Exception:
                global_target_src_text = ""

        # Build refs map (file mode)
        refs_map: Dict[str, List[str]] = {}
        if file_mode and getattr(args, "refs_csv", ""):

            def _norm(p: str) -> str:
                try:
                    return str(Path(p))
                except Exception:
                    return p

            with open(args.refs_csv, "r", encoding="utf-8") as fcsv:
                rdr = csv.reader(fcsv)
                header = next(rdr, None)
                if header and any("mix" in (c or "").lower() for c in header):
                    pass
                else:
                    if header and len(header) >= 3:
                        mix = _norm(header[0])
                        refs_map[mix] = [
                            _norm(x) for x in header[1:] if (x or "").strip()
                        ]
                for row in rdr:
                    if not row or len(row) < 3:
                        continue
                    mix = _norm(row[0])
                    refs_map[mix] = [_norm(x) for x in row[1:] if (x or "").strip()]

        segments_out: List[Dict[str, Any]] = []

        for idx in range(limit):
            if not file_mode:
                assert ds is not None
                sr_item, mix_wav, _sources = ds[idx]
                try:
                    _sr_meta, mix_path, _src_paths = ds.get_metadata(idx)
                    src_paths = list(_src_paths) if _src_paths is not None else []
                except Exception:
                    mix_path = f"index:{idx}"
                    src_paths = []
            else:
                sr_item, mix_wav, _sources, mix_path, src_paths = items[idx]
                try:
                    mix_norm = str(Path(mix_path))
                    if mix_norm in refs_map:
                        src_paths = refs_map[mix_norm]
                    elif getattr(args, "ref_wavs", None) and limit == 1:
                        src_paths = [str(Path(p)) for p in (args.ref_wavs or [])]
                except Exception:
                    pass

            def _resolve_mix_path(root: str, p: str) -> str:
                try:
                    if isinstance(p, str) and (
                        p.startswith("index:") or Path(p).is_absolute()
                    ):
                        return p
                    return str(Path(root) / p)
                except Exception:
                    return p

            if not file_mode:
                abs_mix_path = _resolve_mix_path(str(ds.root), mix_path)  # type: ignore[arg-type]
            else:
                abs_mix_path = mix_path

            mix_np, sr = _ensure_sr_np(mix_wav, sr_item, G_SAMPLE_RATE)
            dur = len(mix_np) / sr
            total_audio_sec += dur

            t_osd0 = time.time()
            osd_segs = self.osd.analyze(mix_np, sr)
            time_osd += time.time() - t_osd0
            if not osd_segs:
                osd_segs = [(0.0, dur, False)]

            # Exclusivity processing
            if getattr(args, "exclusive_segments", True):
                olaps: List[Tuple[float, float]] = []
                for s0, e0, is_ol in osd_segs:
                    if is_ol and (e0 - s0) >= args.min_overlap_dur:
                        s_cl = max(0.0, float(s0))
                        e_cl = min(float(dur), float(e0))
                        if e_cl > s_cl:
                            olaps.append((s_cl, e_cl))

                def _merge_intervals(iv: List[Tuple[float, float]]):
                    if not iv:
                        return []
                    iv = [(max(0.0, s), min(dur, e)) for s, e in iv if e > s]
                    iv.sort(key=lambda x: (x[0], x[1]))
                    merged: List[List[float]] = []
                    for s, e in iv:
                        if not merged or s > merged[-1][1]:
                            merged.append([s, e])
                        else:
                            if e > merged[-1][1]:
                                merged[-1][1] = e
                    return [(float(s), float(e)) for s, e in merged]

                merged_olaps = _merge_intervals(olaps)

                def _complement(
                    iv: List[Tuple[float, float]], start: float, end: float
                ):
                    res: List[Tuple[float, float]] = []
                    cur = start
                    for s, e in iv:
                        if s > cur:
                            res.append((cur, s))
                        cur = max(cur, e)
                    if cur < end:
                        res.append((cur, end))
                    return res

                clean_iv = _complement(merged_olaps, 0.0, float(dur))
                segments = [(s, e, True) for s, e in merged_olaps] + [
                    (s, e, False) for s, e in clean_iv
                ]
                segments.sort(key=lambda x: (x[0], x[1], not x[2]))
            else:
                segments = [
                    (float(s), float(e), bool(is_ol)) for s, e, is_ol in osd_segs
                ]

            # Target per mixture
            target_src_abs = None
            target_src_text = ""
            target_src_np: Optional[np.ndarray] = None
            target_src_text_fallback = ""
            manager_local: Any = None
            enrolled_vec_norm_local = None

            if not file_mode:
                try:
                    target_src_idx = 0
                    if _sources and len(_sources) > 0:
                        target_src_idx = random.randrange(len(_sources))
                    if src_paths and len(src_paths) > target_src_idx:
                        base_root = (
                            str(ds.root) if ds is not None else args.librimix_root
                        )
                        target_src_abs = str(
                            Path(base_root) / src_paths[target_src_idx]
                        )

                    manager_local = sherpa_onnx.SpeakerEmbeddingManager(
                        self.extractor.dim
                    )

                    def _compute_emb(wav_np: np.ndarray, sr: int) -> np.ndarray:
                        s = self.extractor.create_stream()
                        s.accept_waveform(sr, wav_np)
                        s.input_finished()
                        assert self.extractor.is_ready(s)
                        emb = np.array(self.extractor.compute(s), dtype=np.float32)
                        return l2norm(emb)

                    if _sources is not None and len(_sources) > 0:
                        src_wav_t = _sources[target_src_idx]
                        src_np_t, _ = _ensure_sr_np(src_wav_t, sr_item, G_SAMPLE_RATE)
                        target_src_np = src_np_t
                        enrolled_vec = _compute_emb(src_np_t, G_SAMPLE_RATE)
                        enrolled_vec_norm_local = l2norm(enrolled_vec)
                        _ = manager_local.add("target", enrolled_vec)
                        try:
                            st_tgt = asr.create_stream()
                            st_tgt.accept_waveform(G_SAMPLE_RATE, src_np_t)
                            asr.decode_stream(st_tgt)
                            target_src_text_fallback = st_tgt.result.text or ""
                        except Exception:
                            target_src_text_fallback = ""
                except Exception:
                    manager_local = None
                    enrolled_vec_norm_local = None
            else:
                target_src_abs = global_target_src_abs
                target_src_text_fallback = global_target_src_text
                target_src_np = global_target_src_np
                manager_local = manager
                enrolled_vec_norm_local = enrolled_vec_norm

            for s, e, is_olap in segments:
                if e - s <= 0:
                    continue
                s_i = int(s * sr)
                e_i = int(e * sr)
                chunk = mix_np[s_i:e_i]

                if (not is_olap) or (e - s) < args.min_overlap_dur:
                    sv_score = None
                    matched = True
                    seg_dur = e - s
                    n_seen_clean_segments += 1
                    total_seen_clean_audio_sec += seg_dur
                    if (
                        self.extractor is not None
                        and enrolled_vec_norm_local is not None
                    ):
                        sstream = self.extractor.create_stream()
                        sstream.accept_waveform(sr, chunk)
                        sstream.input_finished()
                        if self.extractor.is_ready(sstream):
                            emb = np.array(
                                self.extractor.compute(sstream), dtype=np.float32
                            )
                            emb = l2norm(emb)
                            sv_score = float(np.dot(emb, enrolled_vec_norm_local))
                            if manager_local is not None:
                                pred = manager_local.search(
                                    emb, threshold=args.sv_threshold
                                )
                                matched = pred == "target"
                            else:
                                matched = sv_score >= args.sv_threshold
                        else:
                            matched = False
                    if not matched:
                        n_missed_segments += 1
                        n_missed_clean_segments += 1
                        total_missed_audio_sec += seg_dur
                        continue
                    asr_t0 = time.time()
                    st = asr.create_stream()
                    st.accept_waveform(sr, chunk)
                    asr.decode_stream(st)
                    text = st.result.text
                    asr_t1 = time.time()

                    def _asr_text(np_chunk: Optional[np.ndarray]) -> str:
                        if np_chunk is None or np_chunk.size == 0:
                            return ""
                        try:
                            stx = asr.create_stream()
                            stx.accept_waveform(sr, np_chunk)
                            asr.decode_stream(stx)
                            return stx.result.text or ""
                        except Exception:
                            return ""

                    tgt_text_seg = _asr_text(
                        target_src_np[s_i:e_i] if target_src_np is not None else None
                    )
                    if not tgt_text_seg:
                        tgt_text_seg = target_src_text_fallback
                    rec = {
                        "wav": abs_mix_path,
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "kind": "clean",
                        "stream": None,
                        "text": text,
                        "asr_time": round(asr_t1 - asr_t0, 3),
                        "sv_score": (
                            round(sv_score, 4) if sv_score is not None else None
                        ),
                        "target_src": target_src_abs,
                        "target_src_text": tgt_text_seg,
                    }
                    segments_out.append(rec)
                    n_segments += 1
                    n_clean_segments += 1
                    n_matched_segments += 1
                    total_clean_audio_sec += e - s
                    total_matched_audio_sec += e - s
                    time_asr += asr_t1 - asr_t0
                else:
                    t_sep0 = time.time()
                    pred_wavs = self.sep.separate(chunk, sr)
                    time_sep += time.time() - t_sep0
                    branches = list(pred_wavs)
                    seg_dur = e - s
                    n_seen_overlap_segments += 1
                    total_seen_overlap_audio_sec += seg_dur
                    total_overlap_audio_sec += seg_dur

                    if sep_eval_enabled and src_paths:
                        try:
                            if not file_mode:
                                base_root = (
                                    str(ds.root)
                                    if ds is not None
                                    else args.librimix_root
                                )
                                ref_paths = [
                                    str(Path(base_root) / sp) for sp in src_paths
                                ]
                            else:
                                ref_paths = [str(Path(sp)) for sp in src_paths]
                            K = 3 if len(ref_paths) >= 3 else len(ref_paths)
                            if K not in (2, 3) or len(branches) < K:
                                raise ValueError(
                                    "Need at least 2 references and K<=pred branches"
                                )
                            refs: List[np.ndarray] = []
                            for sp in ref_paths[:K]:
                                sw, ssr = torchaudio.load(sp)
                                snp, _ = _ensure_sr_np(sw, ssr, sr)
                                s_i_ref = int(s * sr)
                                e_i_ref = int(e * sr)
                                refs.append(snp[s_i_ref:e_i_ref])
                            preds_np = [
                                np.asarray(b, dtype=np.float32) for b in branches
                            ]
                            best, sdri, idx_sel = _compute_sdr_improvement_pit_k(
                                chunk, refs, preds_np
                            )
                            if not (np.isnan(best) or np.isnan(sdri)):
                                sep_sisdr_list.append(float(best))
                                sep_sisdri_list.append(float(sdri))
                                sep_details_rows.append(
                                    [
                                        mix_path,
                                        f"{s:.3f}",
                                        f"{e:.3f}",
                                        K,
                                        f"{best:.4f}",
                                        f"{sdri:.4f}",
                                        ";".join(str(i) for i in idx_sel),
                                    ]
                                )
                        except Exception:
                            pass

                    if (
                        self.extractor is not None
                        and enrolled_vec_norm_local is not None
                    ):
                        scores: List[float] = []
                        preds: List[str] = []
                        for w in branches:
                            sstream = self.extractor.create_stream()
                            sstream.accept_waveform(sr, w)
                            sstream.input_finished()
                            if self.extractor.is_ready(sstream):
                                emb = np.array(
                                    self.extractor.compute(sstream), dtype=np.float32
                                )
                                emb = l2norm(emb)
                                score = float(np.dot(emb, enrolled_vec_norm_local))
                                scores.append(score)
                                if manager_local is not None:
                                    pred = manager_local.search(
                                        emb, threshold=args.sv_threshold
                                    )
                                else:
                                    pred = (
                                        "target"
                                        if score >= args.sv_threshold
                                        else "unknown"
                                    )
                                preds.append(pred)
                            else:
                                scores.append(-1.0)
                                preds.append("unknown")
                        best_idx = int(np.argmax(scores)) if scores else 0
                        best_score = scores[best_idx] if scores else -1.0
                        if best_score < args.sv_threshold or (
                            manager_local is not None and preds[best_idx] != "target"
                        ):
                            n_missed_segments += 1
                            n_missed_overlap_segments += 1
                            total_missed_audio_sec += seg_dur
                            continue
                        selected = [(best_idx, branches[best_idx], best_score)]
                    else:
                        n_missed_segments += 1
                        n_missed_overlap_segments += 1
                        total_missed_audio_sec += seg_dur
                        continue

                    for k, w, score in selected:
                        sv_score = float(score) if score is not None else None
                        asr_t0 = time.time()
                        st = asr.create_stream()
                        st.accept_waveform(sr, w)
                        asr.decode_stream(st)
                        text = st.result.text
                        asr_t1 = time.time()

                        def _asr_text2(np_chunk: Optional[np.ndarray]) -> str:
                            if np_chunk is None or np_chunk.size == 0:
                                return ""
                            try:
                                stx = asr.create_stream()
                                stx.accept_waveform(sr, np_chunk)
                                asr.decode_stream(stx)
                                return stx.result.text or ""
                            except Exception:
                                return ""

                        tgt_text_seg = _asr_text2(
                            target_src_np[s_i:e_i]
                            if target_src_np is not None
                            else None
                        )
                        if not tgt_text_seg:
                            tgt_text_seg = target_src_text_fallback
                        rec = {
                            "wav": abs_mix_path,
                            "start": round(s, 3),
                            "end": round(e, 3),
                            "kind": "overlap",
                            "stream": int(k),
                            "text": text,
                            "asr_time": round(asr_t1 - asr_t0, 3),
                            "sv_score": (
                                round(sv_score, 4) if sv_score is not None else None
                            ),
                            "target_src": target_src_abs,
                            "target_src_text": tgt_text_seg,
                        }
                        segments_out.append(rec)
                        n_segments += 1
                        n_overlap_segments += 1
                        n_separated_streams += 1
                        n_matched_segments += 1
                        total_matched_audio_sec += e - s
                        time_asr += asr_t1 - asr_t0

        elapsed_compute = time.time() - t0_all

        resource_stats: Dict[str, Any] = {}
        if args.enable_metrics and monitor is not None:
            try:
                monitor.stop()
                resource_stats = monitor.aggregate()
            except Exception:
                resource_stats = {}

        def _maybe_round(x, nd=4):
            if x is None:
                return None
            try:
                return round(x, nd)
            except Exception:
                return None

        def _agg(vals: List[float]) -> Dict[str, Optional[float]]:
            if not vals:
                return {"mean": None, "median": None, "std": None, "count": 0}
            arr = np.asarray(vals, dtype=np.float32)
            return {
                "mean": round(float(np.mean(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "count": int(arr.size),
            }

        rtf_total = elapsed_compute / total_audio_sec if total_audio_sec > 0 else None
        rtf_asr = time_asr / total_audio_sec if total_audio_sec > 0 else None

        metrics: Dict[str, Any] = {
            "total_audio_sec": round(total_audio_sec, 3),
            "audio_overlap_sec": round(total_overlap_audio_sec, 3),
            "audio_clean_sec": round(total_clean_audio_sec, 3),
            "audio_matched_sec": round(total_matched_audio_sec, 3),
            "audio_seen_clean_sec": round(total_seen_clean_audio_sec, 3),
            "audio_seen_overlap_sec": round(total_seen_overlap_audio_sec, 3),
            "audio_missed_sec": round(total_missed_audio_sec, 3),
            "segments_total": n_segments,
            "segments_clean": n_clean_segments,
            "segments_overlap_streams": n_overlap_segments,
            "separated_streams": n_separated_streams,
            "segments_matched": n_matched_segments,
            "segments_seen_clean": n_seen_clean_segments,
            "segments_seen_overlap": n_seen_overlap_segments,
            "segments_missed": n_missed_segments,
            "segments_missed_clean": n_missed_clean_segments,
            "segments_missed_overlap": n_missed_overlap_segments,
            "target_hit_rate_segments": (
                round(
                    (
                        n_matched_segments
                        / (n_seen_clean_segments + n_seen_overlap_segments)
                    ),
                    4,
                )
                if (n_seen_clean_segments + n_seen_overlap_segments) > 0
                else None
            ),
            "time_osd_sec": round(time_osd, 3),
            "time_sep_sec": round(time_sep, 3),
            "time_asr_sec": round(time_asr, 3),
            "time_compute_total_sec": round(elapsed_compute, 3),
            "rtf_total": _maybe_round(rtf_total, 4),
            "rtf_asr": _maybe_round(rtf_asr, 4),
        }

        if sep_eval_enabled:
            sisdr_stats = _agg(sep_sisdr_list)
            sisdri_stats = _agg(sep_sisdri_list)
            metrics.update(
                {
                    "sep_eval_k_refs": None,
                    "sep_eval_segments": sisdr_stats["count"],
                    "sep_sisdr_mean": sisdr_stats["mean"],
                    "sep_sisdr_median": sisdr_stats["median"],
                    "sep_sisdr_std": sisdr_stats["std"],
                    "sep_sisdri_mean": sisdri_stats["mean"],
                    "sep_sisdri_median": sisdri_stats["median"],
                    "sep_sisdri_std": sisdri_stats["std"],
                }
            )

        metrics.update(resource_stats)

        return PipelineResult(
            segments=segments_out,
            sep_details_rows=sep_details_rows,
            metrics=metrics,
            dataset_name=dataset_name,
            subset=args.subset,
            processed_mixtures=limit,
            sample_rate=args.sample_rate,
        )
