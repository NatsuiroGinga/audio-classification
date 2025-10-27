#!/usr/bin/env python3
"""
Offline OSD + 3-source Separation + ASR (Libri3Mix/LibriMix via torchaudio)

- Uses torchaudio.datasets.LibriMix with num_speakers=3.
- Overlap segments are separated into 3 branches using Conv-TasNet (asteroid backend).
- Target-speaker filtering (REQUIRED): only ASR the branches matching enrolled target speaker.
- Produces per-segment JSONL and CSV, plus optional metrics JSON.

Notes:
- Requires Libri3Mix/LibriMix to be generated beforehand (see https://github.com/JorisCos/LibriMix).
- By default expects 16k (recommended for 3-source HF model), but you can set --sample-rate to 8k if needed.
- Separator will auto-download 3-source checkpoint when not provided if your project integrates it.
"""
import argparse
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import torchaudio
    import torchaudio.functional as AF
    from torchaudio.datasets import LibriMix
except Exception as _e:  # pragma: no cover
    torchaudio = None  # type: ignore
    AF = None  # type: ignore
    LibriMix = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# Alias for resample
try:
    _RESAMPLE = AF.resample if AF is not None else None  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _RESAMPLE = None

from src.model import create_asr_model, l2norm, G_SAMPLE_RATE
from src.osd import OverlapAnalyzer
from src.osd.separation import Separator

try:
    import sherpa_onnx  # type: ignore
except Exception:  # pragma: no cover
    sherpa_onnx = None  # type: ignore


def _log(msg: str):
    print(f"[3src] {msg}")


def _provider_to_torch_device(provider: str) -> str:
    p = (provider or "").lower()
    if "cuda" in p or "gpu" in p:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def _to_mono_float(wav: torch.Tensor) -> torch.Tensor:
    # Accept (T,) or (C, T); convert to mono float32 (T,)
    if wav.dim() == 2:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav[0]
    return wav.float().contiguous()


def _ensure_sr_np(wav: torch.Tensor, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    wav = _to_mono_float(wav)
    if sr != target_sr and wav.numel() > 1:
        if _RESAMPLE is None:
            raise RuntimeError("torchaudio.functional.resample is unavailable")
        wav = _RESAMPLE(wav.unsqueeze(0), sr, target_sr).squeeze(0)
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


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset (LibriMix)
    p.add_argument(
        "--librimix-root", required=True, help="Parent dir of Libri2Mix/Libri3Mix"
    )
    p.add_argument(
        "--subset", default="test", choices=["train-360", "train-100", "dev", "test"]
    )
    p.add_argument("--sample-rate", type=int, default=16000, choices=[8000, 16000])
    p.add_argument(
        "--task",
        default="sep_clean",
        choices=["enh_single", "enh_both", "sep_clean", "sep_noisy"],
    )
    p.add_argument("--mode", default="min", choices=["min", "max"])
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of mixtures processed (0=all)",
    )

    # OSD
    p.add_argument("--osd-backend", default="pyannote")
    p.add_argument("--osd-thr", type=float, default=0.5)
    p.add_argument("--osd-win", type=float, default=0.5)
    p.add_argument("--osd-hop", type=float, default=0.1)

    # Separation (fixed 3 sources here)
    p.add_argument("--sep-backend", default="asteroid")
    p.add_argument(
        "--sep-checkpoint", default="", help="Optional Conv-TasNet checkpoint path"
    )

    # ASR
    p.add_argument("--paraformer", default="")
    p.add_argument("--sense-voice", default="")
    p.add_argument("--encoder", default="")
    p.add_argument("--decoder", default="")
    p.add_argument("--joiner", default="")
    p.add_argument("--tokens", default="")
    p.add_argument("--decoding-method", default="greedy_search")
    p.add_argument("--feature-dim", type=int, default=80)
    p.add_argument("--language", default="auto")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--provider", default="cpu")

    # Target-speaker (optional)
    p.add_argument(
        "--spk-embed-model",
        required=True,
        help="Speaker embedding ONNX model path (REQUIRED)",
    )
    p.add_argument(
        "--enroll-wavs",
        action="append",
        required=True,
        help="Target enrollment wav(s). REQUIRED. Can repeat or comma-separated",
    )
    p.add_argument(
        "--sv-threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold (0~1)",
    )

    # Overlap handling
    p.add_argument("--min-overlap-dur", type=float, default=0.4)

    # Output / metrics
    p.add_argument("--out-dir", default="test/overlap3")
    p.add_argument("--enable-metrics", action="store_true")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--metrics-out", default="metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    if torchaudio is None or LibriMix is None:
        raise RuntimeError(
            "torchaudio and torchaudio.datasets.LibriMix are required. Please install torchaudio matching your torch version."
        )

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_device = _provider_to_torch_device(args.provider)

    # Build components
    asr = _build_asr(args)
    osd = OverlapAnalyzer(
        threshold=args.osd_thr,
        win_sec=args.osd_win,
        hop_sec=args.osd_hop,
        backend=args.osd_backend or "pyannote",
        device=torch_device,
    )
    sep = Separator(
        backend=args.sep_backend or "asteroid",
        checkpoint=(args.sep_checkpoint or None),
        device=torch_device,
        n_src=3,
    )

    # Enrollment (optional)
    extractor: Any = None
    manager: Any = None
    enrolled_vec = None
    enrolled_vec_norm = None
    enroll_list: List[str] = []
    if args.spk_embed_model:
        if sherpa_onnx is None:
            raise RuntimeError(
                "sherpa_onnx is required for speaker enrollment. Please install sherpa_onnx."
            )
        for item in args.enroll_wavs or []:
            for pth in str(item).split(","):
                pth = pth.strip()
                if pth:
                    enroll_list.append(pth)
        if not enroll_list:
            raise RuntimeError(
                "--enroll-wavs is required and must contain at least one wav path"
            )
        if enroll_list:
            # Build sherpa_onnx extractor and manager (follow SpeakerEmbeddingManager usage)
            se_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=args.spk_embed_model,
                num_threads=args.num_threads,
                debug=getattr(args, "debug", False),
                provider=args.provider,
            )
            if not se_config.validate():
                raise ValueError(f"Invalid speaker embedding config: {se_config}")
            extractor = sherpa_onnx.SpeakerEmbeddingExtractor(se_config)
            manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)

            def _compute_emb(wav_np: np.ndarray, sr: int) -> np.ndarray:
                s = extractor.create_stream()
                s.accept_waveform(sr, wav_np)
                s.input_finished()
                assert extractor.is_ready(s)
                emb = np.array(extractor.compute(s), dtype=np.float32)
                return l2norm(emb)

            acc = None
            n_ok = 0
            for ep in enroll_list:
                try:
                    wav, sr = torchaudio.load(ep)
                    wav_np, sr = _ensure_sr_np(wav, sr, G_SAMPLE_RATE)
                    emb = _compute_emb(wav_np, sr)
                    acc = emb if acc is None else (acc + emb)
                    n_ok += 1
                except Exception as _e:
                    _log(f"Enroll failed for {ep}: {_e}")
            if acc is not None and n_ok > 0:
                enrolled_vec = (acc / float(n_ok)).astype(np.float32)
                enrolled_vec_norm = l2norm(enrolled_vec)
                # Register into manager with label 'target'
                _ok = manager.add("target", enrolled_vec)
                if not _ok:
                    _log(
                        "Failed to register target speaker into Manager; filtering disabled."
                    )
                    raise RuntimeError("Failed to register target speaker into Manager")
                else:
                    _log(
                        f"Enrolled target speaker from {n_ok} file(s). Threshold={args.sv_threshold}"
                    )
            else:
                raise RuntimeError(
                    "No valid enrollment wav(s). Please check --enroll-wavs files"
                )

    # Dataset
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
    _log(
        f"Loaded LibriMix subset={args.subset} num_speakers=3 sr={args.sample_rate} size={total}, processing={limit}"
    )

    # Outputs
    seg_jsonl = (out_dir / "segments.jsonl").open("w", encoding="utf-8")
    pred_csv = (out_dir / "segments.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(pred_csv)
    csv_writer.writerow(
        [
            "wav",
            "start",
            "end",
            "kind",
            "stream",
            "text",
            "asr_time",
            "sv_score",
            "matched",
            "match",
        ]
    )

    n_segments = 0
    n_clean_segments = 0
    n_overlap_segments = 0
    n_separated_streams = 0
    n_matched_segments = 0
    # Target hit/miss accounting
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

    class _ResourceMonitor:
        def __init__(self, interval: float):
            self.interval = max(0.1, interval)
            self.samples: List[dict] = []
            self._stop = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self._proc = psutil.Process(os.getpid()) if psutil else None

        def _gpu_info(self):
            if torch.cuda.is_available():
                try:
                    return {
                        "gpu_mem_allocated": torch.cuda.memory_allocated() / (1024**2),
                        "gpu_mem_reserved": torch.cuda.memory_reserved() / (1024**2),
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
                "gpu_mem_reserved_peak_mb": round(max(gpu_res), 2) if gpu_res else None,
            }

    monitor = None
    if args.enable_metrics:
        if psutil is None:
            _log("psutil not installed; resource monitoring disabled.")
        else:
            monitor = _ResourceMonitor(args.monitor_interval)
            monitor.start()

    t0_all = time.time()

    for idx in range(limit):
        sr_item, mix_wav, _sources = ds[idx]
        # Optional: file path via metadata
        try:
            _sr_meta, mix_path, _src_paths = ds.get_metadata(idx)
        except Exception:
            mix_path = f"index:{idx}"

        # OSD on mixture at dataset SR
        mix_np, sr = _ensure_sr_np(mix_wav, sr_item, G_SAMPLE_RATE)
        dur = len(mix_np) / sr
        total_audio_sec += dur

        t_osd0 = time.time()
        segs = osd.analyze(mix_np, sr)
        time_osd += time.time() - t_osd0
        if not segs:
            segs = [(0.0, dur, False)]

        for s, e, is_olap in segs:
            if e - s <= 0:
                continue
            s_i = int(s * sr)
            e_i = int(e * sr)
            chunk = mix_np[s_i:e_i]

            if (not is_olap) or (e - s) < args.min_overlap_dur:
                sv_score = None
                matched = True
                seg_dur = e - s
                # accounting for seen clean segments
                n_seen_clean_segments += 1
                total_seen_clean_audio_sec += seg_dur
                if extractor is not None and enrolled_vec_norm is not None:
                    sstream = extractor.create_stream()
                    sstream.accept_waveform(sr, chunk)
                    sstream.input_finished()
                    if extractor.is_ready(sstream):
                        emb = np.array(extractor.compute(sstream), dtype=np.float32)
                        emb = l2norm(emb)
                        sv_score = float(np.dot(emb, enrolled_vec_norm))
                        # Gate by Manager label match if available
                        if manager is not None:
                            pred = manager.search(emb, threshold=args.sv_threshold)
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
                rec = {
                    "wav": mix_path,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "kind": "clean",
                    "stream": None,
                    "text": text,
                    "asr_time": round(asr_t1 - asr_t0, 3),
                    "sv_score": (round(sv_score, 4) if sv_score is not None else None),
                    "matched": True,
                }
                seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                csv_writer.writerow(
                    [
                        mix_path,
                        f"{s:.3f}",
                        f"{e:.3f}",
                        "clean",
                        "",
                        text,
                        f"{(asr_t1 - asr_t0):.3f}",
                        f"{rec['sv_score'] if rec['sv_score'] is not None else ''}",
                        1,
                        "target",
                    ]
                )
                n_segments += 1
                n_clean_segments += 1
                total_clean_audio_sec += e - s
                total_matched_audio_sec += e - s
                time_asr += asr_t1 - asr_t0
            else:
                # Overlap → separate 3 branches then ASR
                t_sep0 = time.time()
                pred_wavs = sep.separate(chunk, sr)
                time_sep += time.time() - t_sep0
                branches = list(pred_wavs)
                # accounting for seen overlap segments
                seg_dur = e - s
                n_seen_overlap_segments += 1
                total_seen_overlap_audio_sec += seg_dur

                if extractor is not None and enrolled_vec_norm is not None:
                    scores: List[float] = []
                    preds: List[str] = []
                    for w in branches:
                        sstream = extractor.create_stream()
                        sstream.accept_waveform(sr, w)
                        sstream.input_finished()
                        if extractor.is_ready(sstream):
                            emb = np.array(extractor.compute(sstream), dtype=np.float32)
                            emb = l2norm(emb)
                            score = float(np.dot(emb, enrolled_vec_norm))
                            scores.append(score)
                            if manager is not None:
                                pred = manager.search(emb, threshold=args.sv_threshold)
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
                    # Pick best by cosine score, then require manager match 'target'
                    best_idx = int(np.argmax(scores)) if scores else 0
                    best_score = scores[best_idx] if scores else -1.0
                    if best_score < args.sv_threshold or (
                        manager is not None and preds[best_idx] != "target"
                    ):
                        # no branch matches target sufficiently
                        n_missed_segments += 1
                        n_missed_overlap_segments += 1
                        total_missed_audio_sec += seg_dur
                        continue
                    selected = [(best_idx, branches[best_idx], best_score)]
                else:
                    # Filtering is required; without embeddings we cannot match target
                    # Mark as miss for this segment and skip
                    n_missed_segments += 1
                    n_missed_overlap_segments += 1
                    total_missed_audio_sec += seg_dur
                    continue

                for k, w, score in selected:
                    sv_score = float(score) if score is not None else None
                    matched = True
                    asr_t0 = time.time()
                    st = asr.create_stream()
                    st.accept_waveform(sr, w)
                    asr.decode_stream(st)
                    text = st.result.text
                    asr_t1 = time.time()
                    rec = {
                        "wav": mix_path,
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "kind": "overlap",
                        "stream": int(k),
                        "text": text,
                        "asr_time": round(asr_t1 - asr_t0, 3),
                        "sv_score": (
                            round(sv_score, 4) if sv_score is not None else None
                        ),
                        "matched": matched,
                    }
                    seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    csv_writer.writerow(
                        [
                            mix_path,
                            f"{s:.3f}",
                            f"{e:.3f}",
                            "overlap",
                            int(k),
                            text,
                            f"{(asr_t1 - asr_t0):.3f}",
                            f"{rec['sv_score'] if rec['sv_score'] is not None else ''}",
                            1,
                            "target",
                        ]
                    )
                    n_segments += 1
                    n_overlap_segments += 1
                    n_separated_streams += 1
                    n_matched_segments += 1
                    total_matched_audio_sec += e - s
                    time_asr += asr_t1 - asr_t0

    # Close outputs
    seg_jsonl.close()
    pred_csv.close()

    elapsed = time.time()
    # monitor stop and aggregate
    if args.enable_metrics and psutil is not None:
        try:
            monitor.stop()  # type: ignore[union-attr]
            resource_stats = monitor.aggregate()  # type: ignore[union-attr]
        except Exception:
            resource_stats = {}
    else:
        resource_stats = {}

    # Derive metrics
    rtf_total = (elapsed - 0.0) / total_audio_sec if total_audio_sec > 0 else None
    rtf_asr = time_asr / total_audio_sec if total_audio_sec > 0 else None

    def _maybe_round(x, nd=4):
        if x is None:
            return None
        try:
            return round(x, nd)
        except Exception:
            return None

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
        "rtf_total": _maybe_round(rtf_total, 4),
        "rtf_asr": _maybe_round(rtf_asr, 4),
    }
    metrics.update(resource_stats)

    summary = {
        "segments": n_segments,
        "dataset": "LibriMix",
        "subset": args.subset,
        "num_speakers": 3,
        "sample_rate": args.sample_rate,
        "processed_mixtures": limit,
        "notes": "ASR only; overlap segments separated into 3 branches; no CER.",
    }
    # always include target hit/miss summary
    summary.update(
        {
            "target_hits_segments": n_matched_segments,
            "target_misses_segments": n_missed_segments,
            "target_hits_clean_segments": n_clean_segments,
            "target_misses_clean_segments": n_missed_clean_segments,
            "target_hits_overlap_segments": n_overlap_segments,
            "target_misses_overlap_segments": n_missed_overlap_segments,
        }
    )
    if args.enable_metrics:
        with (out_dir / args.metrics_out).open("w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        summary["metrics"] = metrics  # type: ignore[index]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. segments={n_segments}, mixtures={limit}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
