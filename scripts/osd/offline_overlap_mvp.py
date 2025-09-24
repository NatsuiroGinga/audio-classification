#!/usr/bin/env python3
"""Offline MVP: OSD + Separation + ASR (Libri2Mix 8k, no Speaker ID, no CER).

Pipeline per mixture:
    1. Load mixture wav from Libri2Mix 8k test split (field: mix_wav:FILE).
    2. Overlapped Speech Detection (pyannote) -> segment list (start, end, overlap_flag).
    3. Non-overlap segments: direct ASR.
    4. Overlap segments: Conv-TasNet separation -> ASR on each branch.
    5. Output per-segment JSONL/CSV (no speaker columns). Summary has timing & counts only.

Notes:
    - Libri2Mix test split here assumed without paired transcripts (CER disabled).
    - Longest textual hypothesis among separated branches conceptually retained (not reassembled here).
    - Use --max-files to limit processed mixtures for quick smoke tests.
"""
import argparse
import csv
import json
import os
import time
import threading
from statistics import mean
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional psutil for resource monitoring
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

# torchaudio is required for PyTorch-based audio I/O
try:
    import torchaudio
    import torchaudio.functional as AF
except Exception as _e:
    torchaudio = None  # will raise with guidance when used

from src.model import create_asr_model, G_SAMPLE_RATE
from src.osd import OverlapAnalyzer
from src.osd.separation import Separator
from src.mossformer.dataset import Libri2Mix8kDataset


def _log(msg: str):
    print(f"[overlap_mvp] {msg}")


# Cache to store tensors on device while still returning numpy for existing code paths
_AUDIO_TENSOR_CACHE: Dict[str, Tuple[torch.Tensor, int]] = {}


def _provider_to_torch_device(provider: str) -> str:
    p = (provider or "").lower()
    if "cuda" in p or "gpu" in p:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_audio(fname: str, target_sr: int = 16000, device: Optional[str] = None):
    """Load audio via torchaudio, return (numpy_waveform, sr) and cache tensor on device.

    - Loads with torchaudio.load (C, T), converts to mono float32 (T,)
    - Resamples via torchaudio.functional.resample if needed
    - Moves final tensor to the requested device and caches it in _AUDIO_TENSOR_CACHE
    - Returns numpy waveform (on CPU) and sample rate for existing consumers
    """
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is required for PyTorch-based audio loading. Please install torchaudio matching your torch version."
        )
    wav, sr = torchaudio.load(fname)  # (C, T), dtype per backend
    # convert to mono
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0)
    elif wav.dim() == 2 and wav.size(0) == 1:
        wav = wav[0]
    wav = wav.float()
    if sr != target_sr and wav.numel() > 1:
        wav = AF.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    # ensure contiguous and range in [-1, 1]
    wav = wav.contiguous()
    dev = device or "cpu"
    wav_dev = wav.to(dev, non_blocking=True)
    _AUDIO_TENSOR_CACHE[fname] = (wav_dev, sr)
    # Return numpy copy for downstream numpy-based components
    wav_np = wav.cpu().numpy()
    return wav_np, sr


############################################################
# ASR ONLY (No speaker identification, CER disabled)
############################################################


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
    # Speaker ID removed
    p.add_argument(
        "--model", default="", help="(Ignored) speaker embedding path placeholder"
    )
    # ASR options
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
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="(Ignored) kept for backward CLI compatibility",
    )
    # References for CER
    # References disabled for Libri2Mix test
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit number of mixtures processed (0 = all)",
    )
    # OSD
    p.add_argument(
        "--osd-backend", default="pyannote", help="OSD backend (required: pyannote)"
    )
    p.add_argument("--osd-thr", type=float, default=0.5)
    p.add_argument("--osd-win", type=float, default=0.5)
    p.add_argument("--osd-hop", type=float, default=0.1)
    # Separation
    p.add_argument(
        "--sep-backend",
        default="asteroid",
        help="Separation backend (required: asteroid)",
    )
    p.add_argument(
        "--sep-checkpoint", default="", help="Optional: path to Conv-TasNet checkpoint"
    )
    p.add_argument("--min-overlap-dur", type=float, default=0.4)
    # Output
    p.add_argument("--out-dir", default="test_overlap")
    # Metrics / monitoring
    p.add_argument(
        "--enable-metrics",
        action="store_true",
        help="Enable extended metrics collection (RTF, per-stage timing, resource usage)",
    )
    p.add_argument(
        "--monitor-interval",
        type=float,
        default=0.5,
        help="Resource monitor sampling interval seconds (effective when --enable-metrics)",
    )
    p.add_argument(
        "--metrics-out",
        default="metrics.json",
        help="Filename for detailed metrics JSON (inside run output dir)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _provider_to_torch_device(args.provider)

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
    )

    ds = Libri2Mix8kDataset.load_test()
    total = len(ds)
    limit = args.max_files if args.max_files and args.max_files > 0 else total
    _log(
        f"Loaded {Libri2Mix8kDataset.dataset_name} test split size={total}, processing={limit}"
    )

    seg_jsonl = (out_dir / "segments.jsonl").open("w", encoding="utf-8")
    pred_csv = (out_dir / "segments.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(pred_csv)
    csv_writer.writerow(["wav", "start", "end", "kind", "stream", "text", "asr_time"])
    n_segments = 0
    n_clean_segments = 0
    n_overlap_segments = 0
    n_separated_streams = 0
    total_audio_sec = 0.0
    total_overlap_audio_sec = 0.0
    total_clean_audio_sec = 0.0
    time_osd = 0.0
    time_sep = 0.0
    time_asr = 0.0

    # ----------------------------------------------------------------------------
    # Resource Monitor (optional)
    # ----------------------------------------------------------------------------
    class _ResourceMonitor:
        def __init__(self, interval: float):
            self.interval = max(0.1, interval)
            self.samples: List[dict] = []
            self._stop = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self._proc = psutil.Process(os.getpid()) if psutil else None

        def _gpu_info(self):  # lightweight GPU mem (torch only)
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
            # Prime cpu_percent to avoid first 0.0 reading
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
    processed = 0
    for idx in range(limit):
        item = ds[idx]
        wav_path_obj = item.get("mix_wav:FILE")  # type: ignore
        # Ensure we extract a string path (Dataset may wrap value)
        wav_path = str(wav_path_obj) if wav_path_obj is not None else ""
        if not wav_path or not os.path.isfile(wav_path):
            continue
        samples, sr = load_audio(wav_path, device=torch_device, target_sr=G_SAMPLE_RATE)
        dur = len(samples) / sr
        # OSD timing
        t_osd0 = time.time()
        segs = osd.analyze(samples, sr)
        time_osd += time.time() - t_osd0
        if not segs:
            segs = [(0.0, dur, False)]
        total_audio_sec += dur
        for s, e, is_olap in segs:
            if e - s <= 0:
                continue
            s_i = int(s * sr)
            e_i = int(e * sr)
            chunk = samples[s_i:e_i]
            if (not is_olap) or (e - s) < args.min_overlap_dur:
                asr_t0 = time.time()
                st = asr.create_stream()
                st.accept_waveform(sr, chunk)
                asr.decode_stream(st)
                text = st.result.text
                asr_t1 = time.time()
                seg_dur = e - s
                total_clean_audio_sec += seg_dur
                rec = {
                    "wav": wav_path,
                    "start": round(s, 3),
                    "end": round(e, 3),
                    "kind": "clean",
                    "stream": None,
                    "text": text,
                    "asr_time": round(asr_t1 - asr_t0, 3),
                }
                seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                csv_writer.writerow(
                    [
                        wav_path,
                        f"{s:.3f}",
                        f"{e:.3f}",
                        "clean",
                        "",
                        text,
                        f"{(asr_t1 - asr_t0):.3f}",
                    ]
                )
                n_segments += 1
                n_clean_segments += 1
                time_asr += asr_t1 - asr_t0
            else:
                seg_dur = e - s
                total_overlap_audio_sec += seg_dur
                t_sep0 = time.time()
                w1, w2 = sep.separate(chunk, sr)
                time_sep += time.time() - t_sep0
                branches = [w1, w2]
                for k, w in enumerate(branches):
                    asr_t0 = time.time()
                    st = asr.create_stream()
                    st.accept_waveform(sr, w)
                    asr.decode_stream(st)
                    text = st.result.text
                    asr_t1 = time.time()
                    rec = {
                        "wav": wav_path,
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "kind": "overlap",
                        "stream": k,
                        "text": text,
                        "asr_time": round(asr_t1 - asr_t0, 3),
                    }
                    seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    csv_writer.writerow(
                        [
                            wav_path,
                            f"{s:.3f}",
                            f"{e:.3f}",
                            "overlap",
                            k,
                            text,
                            f"{(asr_t1 - asr_t0):.3f}",
                        ]
                    )
                    n_segments += 1
                    n_overlap_segments += 1
                    n_separated_streams += 1
                    time_asr += asr_t1 - asr_t0
        processed += 1
        if processed % 50 == 0:
            _log(f"Processed {processed}/{limit} mixtures")

    seg_jsonl.close()
    pred_csv.close()
    elapsed = time.time() - t0_all

    if monitor:
        monitor.stop()
        resource_stats = monitor.aggregate()
    else:
        resource_stats = {}

    # Derived metrics
    rtf_total = elapsed / total_audio_sec if total_audio_sec > 0 else None
    rtf_asr = time_asr / total_audio_sec if total_audio_sec > 0 else None
    stage_share = lambda t: (t / elapsed) if elapsed > 0 else None  # noqa: E731

    def _maybe_round(val, nd=4):
        if val is None:
            return None
        try:
            return round(val, nd)
        except Exception:
            return None

    metrics: Dict[str, object] = {
        "total_audio_sec": round(total_audio_sec, 3),
        "audio_overlap_sec": round(total_overlap_audio_sec, 3),
        "audio_clean_sec": round(total_clean_audio_sec, 3),
        "segments_total": n_segments,
        "segments_clean": n_clean_segments,
        "segments_overlap_streams": n_overlap_segments,
        "separated_streams": n_separated_streams,
        "time_wall_sec": round(elapsed, 3),
        "time_osd_sec": round(time_osd, 3),
        "time_sep_sec": round(time_sep, 3),
        "time_asr_sec": round(time_asr, 3),
        "share_osd": _maybe_round(stage_share(time_osd), 4),
        "share_sep": _maybe_round(stage_share(time_sep), 4),
        "share_asr": _maybe_round(stage_share(time_asr), 4),
        "rtf_total": _maybe_round(rtf_total, 4),
        "rtf_asr": _maybe_round(rtf_asr, 4),
    }
    metrics.update(resource_stats)

    summary = {
        "segments": n_segments,
        "elapsed_wall_sec": round(elapsed, 3),
        "dataset": Libri2Mix8kDataset.dataset_name,
        "processed_mixtures": processed,
        "sample_rate_target": G_SAMPLE_RATE,
        "notes": "ASR only; overlap segments separated; no CER (no refs).",
    }
    if args.enable_metrics:
        summary["metrics"] = metrics  # type: ignore[index]
        with (out_dir / args.metrics_out).open("w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Done. segments={n_segments}, mixtures={processed}, elapsed={elapsed:.3f}s, RTF={metrics.get('rtf_total') if args.enable_metrics else 'n/a'}, out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
