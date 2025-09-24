#!/usr/bin/env python3
"""Evaluate overlap detection + separation quality using source references.

Requirements:
  - Dataset items provide fields:
      mix_wav:FILE, s1_wav:FILE, s2_wav:FILE
  - We compute:
      * OSD precision/recall/F1 (frame / hop grid level) using energy-based GT mask
      * Separation SI-SDR & SI-SDRi on predicted overlap segments (>= min-overlap-dur)
  - ASR is NOT invoked here (focus is structural + signal quality). Can be extended later.

Pipeline per mixture:
  1. Load mixture and source waveforms (mono, target_sr=16k) with torchaudio.
  2. Run OverlapAnalyzer to get predicted overlap segments.
  3. Build GT overlap mask via energy activity threshold on sources; derive GT segments.
  4. Convert predicted segments to frame mask; compute OSD metrics.
  5. For each predicted overlap segment (long enough), run separation, compute SI-SDR / SI-SDRi (permutation invariant) against references within that span; aggregate.
  6. Output evaluation JSON and CSV (optional) with per-overlap segment SI-SDR details.

Usage example:
  python3 evaluate_with_sources.py \
      --max-files 50 \
      --osd-backend pyannote --sep-backend asteroid \
      --sep-checkpoint /path/to/conv_tasnet.pt (optional) \
      --activity-thr 0.03 --min-overlap-dur 0.4

Outputs (default directory test_overlap_eval/<timestamp>):
   evaluation.json         # aggregated metrics
   overlap_details.csv     # per predicted overlap segment (optional)

Notes:
  - Energy activity threshold heuristic; adjust --activity-thr for stricter/looser speech detection.
  - If no overlap segments predicted or GT empty, metrics degrade gracefully.
  - SI-SDR computed only on predicted overlap segments (not on GT-only overlaps not predicted).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import threading

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None

# torchaudio for audio I/O
try:
    import torchaudio
    import torchaudio.functional as AF
except Exception as _e:  # pragma: no cover
    torchaudio = None  # type: ignore

# ---------------------------------------------------------------------------
# Add project root to sys.path when executed from scripts/osd
# ---------------------------------------------------------------------------
_FILE_PATH = Path(__file__).resolve()
_ROOT = _FILE_PATH.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.osd import OverlapAnalyzer  # noqa: E402
from src.osd.separation import Separator  # noqa: E402
from src.model import G_SAMPLE_RATE, create_asr_model  # noqa: E402
from src.mossformer.dataset import Libri2Mix8kDataset  # noqa: E402


def _log(msg: str):
    print(f"[eval] {msg}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--max-files", type=int, default=0, help="Limit number of mixtures (0=all)"
    )
    # OSD
    p.add_argument("--osd-backend", default="pyannote")
    p.add_argument("--osd-thr", type=float, default=0.5)
    p.add_argument("--osd-win", type=float, default=0.5)
    p.add_argument("--osd-hop", type=float, default=0.1)
    # Separation
    p.add_argument("--sep-backend", default="asteroid")
    p.add_argument("--sep-checkpoint", default="")
    p.add_argument("--min-overlap-dur", type=float, default=0.4)
    # Activity threshold (ratio to peak frame RMS)
    p.add_argument(
        "--activity-thr",
        type=float,
        default=0.03,
        help="Frame considered active if RMS > peak_rms * activity_thr",
    )
    # Output
    p.add_argument("--out-dir", default="test_overlap_eval")
    p.add_argument(
        "--save-details",
        action="store_true",
        help="Save per overlap segment details CSV",
    )
    p.add_argument("--provider", default="cpu")
    # ASR evaluation (optional)
    p.add_argument(
        "--enable-asr",
        action="store_true",
        help="Enable pseudo-reference ASR evaluation (overlap vs clean)",
    )
    p.add_argument("--paraformer", default="")
    p.add_argument("--sense-voice", default="")
    p.add_argument("--encoder", default="")
    p.add_argument("--decoder", default="")
    p.add_argument("--joiner", default="")
    p.add_argument("--tokens", default="")
    p.add_argument("--decoding-method", default="greedy_search")
    p.add_argument("--feature-dim", type=int, default=80)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--language", default="auto")
    return p.parse_args()


def _provider_to_torch_device(provider: str) -> str:
    p = (provider or "").lower()
    if "cuda" in p or "gpu" in p:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def load_audio(
    path: str, target_sr: int, device: Optional[str] = None
) -> Tuple[np.ndarray, int]:
    if torchaudio is None:
        raise RuntimeError("torchaudio not available; please install it.")
    wav, sr = torchaudio.load(path)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0)
    elif wav.dim() == 2 and wav.size(0) == 1:
        wav = wav[0]
    wav = wav.float()
    if sr != target_sr:
        wav = AF.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
        sr = target_sr
    if device:
        wav = wav.to(device)
    return wav.cpu().numpy(), sr


def frame_rms(wav: np.ndarray, sr: int, win: float, hop: float) -> np.ndarray:
    win_s = int(win * sr)
    hop_s = int(hop * sr)
    if win_s <= 0:
        raise ValueError("win too small")
    if hop_s <= 0:
        raise ValueError("hop too small")
    out = []
    for start in range(0, max(len(wav) - win_s + 1, 1), hop_s):
        seg = wav[start : start + win_s]
        if len(seg) == 0:
            out.append(0.0)
        else:
            out.append(float(np.sqrt(np.mean(seg**2) + 1e-12)))
    return np.asarray(out, dtype=np.float32)


def masks_to_segments(
    mask: np.ndarray, hop: float, win: float, total_dur: float
) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    if len(mask) == 0:
        return []
    cur = mask[0]
    start_t = 0.0
    for i in range(1, len(mask)):
        if mask[i] != cur:
            if cur:
                end_t = i * hop + win
                segs.append((start_t, min(end_t, total_dur)))
            start_t = i * hop
            cur = mask[i]
    if cur:
        segs.append((start_t, total_dur))
    # sanitize
    return [(max(0.0, s), min(total_dur, e)) for s, e in segs if e > s]


def build_gt_overlap_mask(
    s1: np.ndarray, s2: np.ndarray, sr: int, win: float, hop: float, thr_ratio: float
) -> np.ndarray:
    # Compute frame RMS
    rms1 = frame_rms(s1, sr, win, hop)
    rms2 = frame_rms(s2, sr, win, hop)
    peak = max(rms1.max(initial=0.0), rms2.max(initial=0.0), 1e-9)
    active1 = rms1 > peak * thr_ratio
    active2 = rms2 > peak * thr_ratio
    return active1 & active2


def segments_to_mask(
    segments: List[Tuple[float, float, bool]], dur: float, hop: float, win: float
) -> np.ndarray:
    # Create frame centers identical to GT frame indexing (start from 0, step hop)
    grid = np.arange(0, max(dur - win, 0) + 1e-9, hop)
    mask = np.zeros(len(grid), dtype=bool)
    for s, e, is_olap in segments:
        if not is_olap:
            continue
        # Mark frames whose window midpoints overlap [s, e)
        # Approx: consider frame start at t, coverage until t+win
        frame_starts = grid
        frame_ends = grid + win
        idx = np.where((frame_starts < e) & (frame_ends > s))[0]
        mask[idx] = True
    return mask


def compute_osd_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    if len(gt_mask) == 0 or len(pred_mask) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    n = min(len(gt_mask), len(pred_mask))
    gt = gt_mask[:n]
    pr = pred_mask[:n]
    tp = float(np.sum(gt & pr))
    fp = float(np.sum(~gt & pr))
    fn = float(np.sum(gt & ~pr))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tp_frames": int(tp),
        "fp_frames": int(fp),
        "fn_frames": int(fn),
    }


def si_sdr(reference: np.ndarray, estimation: np.ndarray) -> float:
    """Scale-Invariant SDR (in dB)."""
    if reference.shape != estimation.shape:
        m = min(len(reference), len(estimation))
        reference = reference[:m]
        estimation = estimation[:m]
    ref = reference - np.mean(reference)
    est = estimation - np.mean(estimation)
    ref_energy = np.sum(ref**2) + 1e-12
    # projection
    scale = np.dot(est, ref) / ref_energy
    proj = scale * ref
    e_noise = est - proj
    ratio = (np.sum(proj**2) + 1e-12) / (np.sum(e_noise**2) + 1e-12)
    return 10 * math.log10(ratio)


def perm_si_sdr(
    s1_ref: np.ndarray, s2_ref: np.ndarray, w1: np.ndarray, w2: np.ndarray
) -> Tuple[float, float]:
    # Two permutations
    sdr_12 = (si_sdr(s1_ref, w1) + si_sdr(s2_ref, w2)) / 2.0
    sdr_21 = (si_sdr(s1_ref, w2) + si_sdr(s2_ref, w1)) / 2.0
    if sdr_21 > sdr_12:
        return sdr_21, 1.0  # best is swapped (flag=1)
    return sdr_12, 0.0


def compute_sdr_improvement(
    mix_chunk: np.ndarray,
    s1_ref: np.ndarray,
    s2_ref: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
) -> Tuple[float, float]:
    # Baseline: mixture vs each ref (avg)
    base = (si_sdr(s1_ref, mix_chunk) + si_sdr(s2_ref, mix_chunk)) / 2.0
    best_sdr, perm_flag = perm_si_sdr(s1_ref, s2_ref, w1, w2)
    return best_sdr, best_sdr - base


#########################
# CPU monitor utilities #
#########################
class CPUMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = max(0.1, interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[float] = []
        self.proc = psutil.Process(os.getpid()) if psutil else None
        self.started = self.proc is not None
        if self.proc:
            # prime
            try:
                self.proc.cpu_percent(interval=None)
            except Exception:
                self.started = False

    def start(self):
        if not self.started:
            return

        def _loop():
            while not self._stop.wait(self.interval):
                try:
                    self.samples.append(self.proc.cpu_percent(interval=None))  # type: ignore
                except Exception:
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        if not self.started:
            return {"enabled": False, "reason": "psutil_unavailable"}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"enabled": True, "count": 0}
        raw_avg = float(np.mean(self.samples))
        raw_peak = float(np.max(self.samples))
        try:
            cores = psutil.cpu_count(logical=True) if psutil else 1
        except Exception:
            cores = 1
        cores = cores or 1
        norm_avg = min(100.0, raw_avg / cores)
        norm_peak = min(100.0, raw_peak / cores)
        return {
            "enabled": True,
            "count": len(self.samples),
            "interval_sec": self.interval,
            "cpu_logical_cores": cores,
            "cpu_avg_percent": round(norm_avg, 2),
            "cpu_peak_percent": round(norm_peak, 2),
            "cpu_avg_percent_raw": round(raw_avg, 2),
            "cpu_peak_percent_raw": round(raw_peak, 2),
            "normalized": True,
        }


#########################
# ASR utility functions #
#########################


def _normalize_text(t: str) -> str:
    return t.strip()


def _split_words(t: str) -> List[str]:
    t = _normalize_text(t)
    return t.split() if t else []


def _cer(ref: str, hyp: str) -> float:
    ref_chars = list(_normalize_text(ref))
    hyp_chars = list(_normalize_text(hyp))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n] / m


def _wer(ref: str, hyp: str) -> float:
    r = _split_words(ref)
    h = _split_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n] / m


def _build_asr(args):
    return create_asr_model(
        paraformer=args.paraformer,
        sense_voice=args.sense_voice,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        tokens=args.tokens,
        num_threads=args.num_threads,
        feature_dim=args.feature_dim,
        decoding_method=args.decoding_method,
        debug=False,
        language=args.language,
        provider=args.provider,
    )


def _asr_infer(model, sr: int, wav: np.ndarray) -> str:
    st = model.create_stream()
    st.accept_waveform(sr, wav)
    model.decode_stream(st)
    return _normalize_text(st.result.text)


def main():
    args = parse_args()
    # Prepare dirs
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_base / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _provider_to_torch_device(args.provider)
    _log(f"Device={device}")

    # Initialize models
    osd = OverlapAnalyzer(
        threshold=args.osd_thr,
        win_sec=args.osd_win,
        hop_sec=args.osd_hop,
        backend=args.osd_backend,
        device=device,
    )
    sep = Separator(
        backend=args.sep_backend,
        checkpoint=(args.sep_checkpoint or None),
        device=device,
    )
    asr_model = _build_asr(args) if args.enable_asr else None

    # Start CPU monitor
    cpu_mon = CPUMonitor(interval=0.5)
    cpu_mon.start()

    ds = Libri2Mix8kDataset.load_test()
    total = len(ds)
    limit = args.max_files if args.max_files and args.max_files > 0 else total
    _log(
        f"Loaded dataset {Libri2Mix8kDataset.dataset_name} size={total}, processing={limit}"
    )

    # Aggregations
    osd_tp = osd_fp = osd_fn = 0
    gt_overlap_total_sec = 0.0
    pred_overlap_total_sec = 0.0

    # NEW: timing accumulators
    audio_total_sec = 0.0  # 所有混合音频总时长(去裁剪)
    osd_time_sec = 0.0  # OSD 推理总耗时
    sep_time_sec = 0.0  # 分离总耗时
    asr_time_sec = 0.0  # ASR 推理总耗时
    overlap_predicted_sec_for_sep = 0.0  # 进入分离调用的预测重叠时长（过滤后）

    sdr_list: List[float] = []
    sdri_list: List[float] = []

    details_csv = None
    writer = None
    if args.save_details:
        details_csv = (out_dir / "overlap_details.csv").open(
            "w", newline="", encoding="utf-8"
        )
        writer = csv.writer(details_csv)
        writer.writerow(
            ["wav", "seg_start", "seg_end", "dur", "si_sdr", "si_sdri", "perm_swapped"]
        )

    t0 = time.time()

    # ASR accumulators
    overlap_mix_refs: List[str] = []
    overlap_mix_hyps: List[str] = []
    overlap_sep_refs: List[str] = []
    overlap_sep_hyps: List[str] = []
    clean_refs: List[str] = []
    clean_hyps: List[str] = []

    for idx in range(limit):
        item = ds[idx]
        try:
            mix_p = str(item["mix_wav:FILE"])  # type: ignore[index]
            s1_p = str(item["s1_wav:FILE"])  # type: ignore[index]
            s2_p = str(item["s2_wav:FILE"])  # type: ignore[index]
        except Exception:
            mix_p = str(getattr(item, "mix_wav:FILE", ""))
            s1_p = str(getattr(item, "s1_wav:FILE", ""))
            s2_p = str(getattr(item, "s2_wav:FILE", ""))
        if not (
            mix_p
            and s1_p
            and s2_p
            and os.path.isfile(mix_p)
            and os.path.isfile(s1_p)
            and os.path.isfile(s2_p)
        ):
            continue
        mix, sr = load_audio(mix_p, target_sr=G_SAMPLE_RATE)
        s1, _ = load_audio(s1_p, target_sr=G_SAMPLE_RATE)
        s2, _ = load_audio(s2_p, target_sr=G_SAMPLE_RATE)
        m = min(len(mix), len(s1), len(s2))
        mix = mix[:m]
        s1 = s1[:m]
        s2 = s2[:m]
        dur = m / sr
        audio_total_sec += dur  # 统计总音频时长

        # Predicted segments (计时 OSD)
        t_osd_start = time.time()
        pred_segments = osd.analyze(mix, sr)
        osd_time_sec += time.time() - t_osd_start
        if not pred_segments:
            pred_segments = [(0.0, dur, False)]

        pred_mask = segments_to_mask(pred_segments, dur, args.osd_hop, args.osd_win)
        pred_overlap_total_sec += sum(e - s for s, e, f in pred_segments if f)

        gt_mask = build_gt_overlap_mask(
            s1, s2, sr, args.osd_win, args.osd_hop, args.activity_thr
        )
        gt_segments = masks_to_segments(gt_mask, args.osd_hop, args.osd_win, dur)
        gt_overlap_total_sec += sum(e - s for s, e in gt_segments)

        n = min(len(gt_mask), len(pred_mask))
        gmask = gt_mask[:n]
        pmask = pred_mask[:n]
        tp = int(np.sum(gmask & pmask))
        fp = int(np.sum(~gmask & pmask))
        fn = int(np.sum(gmask & ~pmask))
        osd_tp += tp
        osd_fp += fp
        osd_fn += fn

        # Separation SI-SDR
        for s, e, is_olap in pred_segments:
            if not is_olap:
                continue
            if (e - s) < args.min_overlap_dur:
                continue
            s_i = int(s * sr)
            e_i = int(e * sr)
            if e_i <= s_i:
                continue
            mix_chunk = mix[s_i:e_i]
            s1_chunk = s1[s_i:e_i]
            s2_chunk = s2[s_i:e_i]
            t_sep_start = time.time()
            w1, w2 = sep.separate(mix_chunk, sr)
            sep_time_sec += time.time() - t_sep_start
            overlap_predicted_sec_for_sep += e - s
            seg_sdr, seg_sdri = compute_sdr_improvement(
                mix_chunk, s1_chunk, s2_chunk, w1, w2
            )
            sdr_list.append(seg_sdr)
            sdri_list.append(seg_sdri)
            if writer:
                writer.writerow(
                    [
                        mix_p,
                        f"{s:.3f}",
                        f"{e:.3f}",
                        f"{(e - s):.3f}",
                        f"{seg_sdr:.3f}",
                        f"{seg_sdri:.3f}",
                        1 if seg_sdri < 0 else 0,
                    ]
                )

        # ASR 评估
        if args.enable_asr and asr_model is not None:
            rms1 = frame_rms(s1, sr, args.osd_win, args.osd_hop)
            rms2 = frame_rms(s2, sr, args.osd_win, args.osd_hop)
            peak = max(rms1.max(initial=0.0), rms2.max(initial=0.0), 1e-9)
            active1 = rms1 > peak * args.activity_thr
            active2 = rms2 > peak * args.activity_thr
            gt_overlap_mask = active1 & active2
            clean1_mask = active1 & ~active2
            clean2_mask = active2 & ~active1
            overlap_segments = masks_to_segments(
                gt_overlap_mask, args.osd_hop, args.osd_win, dur
            )
            clean1_segments = masks_to_segments(
                clean1_mask, args.osd_hop, args.osd_win, dur
            )
            clean2_segments = masks_to_segments(
                clean2_mask, args.osd_hop, args.osd_win, dur
            )

            for s_t, e_t in overlap_segments:
                if (e_t - s_t) < args.min_overlap_dur:
                    continue
                s_i = int(s_t * sr)
                e_i = int(e_t * sr)
                if e_i <= s_i:
                    continue
                mix_chunk = mix[s_i:e_i]
                s1_chunk = s1[s_i:e_i]
                s2_chunk = s2[s_i:e_i]
                t_asr_start = time.time()
                ref1_txt = _asr_infer(asr_model, sr, s1_chunk)
                ref2_txt = _asr_infer(asr_model, sr, s2_chunk)
                mix_hyp = _asr_infer(asr_model, sr, mix_chunk)
                w1, w2 = sep.separate(mix_chunk, sr)
                sep_time_sec += (
                    0.0  # 已在上面对分离计时，此处不重复；如需单独计时可再包
                )
                hyp1 = _asr_infer(asr_model, sr, w1)
                hyp2 = _asr_infer(asr_model, sr, w2)
                asr_time_sec += time.time() - t_asr_start
                cost_12 = _cer(ref1_txt, hyp1) + _cer(ref2_txt, hyp2)
                cost_21 = _cer(ref1_txt, hyp2) + _cer(ref2_txt, hyp1)
                hyp_pair = hyp2 + " " + hyp1 if cost_21 < cost_12 else hyp1 + " " + hyp2
                overlap_mix_refs.append(ref1_txt + " " + ref2_txt)
                overlap_mix_hyps.append(mix_hyp)
                overlap_sep_refs.append(ref1_txt + " " + ref2_txt)
                overlap_sep_hyps.append(hyp_pair)

            def _eval_clean(seg_list, src_wav):
                nonlocal asr_time_sec
                for s_t, e_t in seg_list:
                    if (e_t - s_t) < 0.05:
                        continue
                    s_i = int(s_t * sr)
                    e_i = int(e_t * sr)
                    if e_i <= s_i:
                        continue
                    ref_chunk = src_wav[s_i:e_i]
                    mix_chunk = mix[s_i:e_i]
                    t_asr = time.time()
                    ref_txt = _asr_infer(asr_model, sr, ref_chunk)
                    mix_txt = _asr_infer(asr_model, sr, mix_chunk)
                    asr_time_sec += time.time() - t_asr
                    clean_refs.append(ref_txt)
                    clean_hyps.append(mix_txt)

            _eval_clean(clean1_segments, s1)
            _eval_clean(clean2_segments, s2)

        if (idx + 1) % 20 == 0:
            _log(f"Processed {idx+1}/{limit}")

    if details_csv:
        details_csv.close()

    # ...existing code (summary build)...
    elapsed = time.time() - t0
    precision = osd_tp / (osd_tp + osd_fp) if (osd_tp + osd_fp) > 0 else 0.0
    recall = osd_tp / (osd_tp + osd_fn) if (osd_tp + osd_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = osd_tp / (osd_tp + osd_fp + osd_fn) if (osd_tp + osd_fp + osd_fn) > 0 else 0.0

    def _safe_stats(vals: List[float]):
        if not vals:
            return {"count": 0}
        arr = np.asarray(vals)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # NEW: RTF 计算
    def _div(a, b):
        return (a / b) if (b and b > 0) else 0.0

    rtf_total = _div(elapsed, audio_total_sec)
    rtf_asr = _div(asr_time_sec, audio_total_sec)
    rtf_sep_total = _div(sep_time_sec, audio_total_sec)
    rtf_sep_overlap = _div(sep_time_sec, overlap_predicted_sec_for_sep)
    rtf_osd = _div(osd_time_sec, audio_total_sec)

    eval_json = {
        "dataset": Libri2Mix8kDataset.dataset_name,
        "files_limit": limit,
        "elapsed_sec": round(elapsed, 3),
        "hop_sec": args.osd_hop,
        "win_sec": args.osd_win,
        "activity_thr": args.activity_thr,
        "min_overlap_dur": args.min_overlap_dur,
        "gt_overlap_total_sec": round(gt_overlap_total_sec, 3),
        "pred_overlap_total_sec": round(pred_overlap_total_sec, 3),
        "audio_total_sec": round(audio_total_sec, 3),
        "timing": {
            "time_wall_sec": round(elapsed, 3),
            "time_osd_sec": round(osd_time_sec, 3),
            "time_sep_sec": round(sep_time_sec, 3),
            "time_asr_sec": round(asr_time_sec, 3),
            "overlap_predicted_sec_for_sep": round(overlap_predicted_sec_for_sep, 3),
            "rtf_total": round(rtf_total, 4),
            "rtf_osd": round(rtf_osd, 4),
            "rtf_sep_total": round(rtf_sep_total, 4),
            "rtf_sep_overlap": round(rtf_sep_overlap, 4),
            "rtf_asr": round(rtf_asr, 4),
        },
        "osd": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "iou": round(iou, 4),
            "tp_frames": osd_tp,
            "fp_frames": osd_fp,
            "fn_frames": osd_fn,
        },
        "separation": {
            "si_sdr": _safe_stats(sdr_list),
            "si_sdri": _safe_stats(sdri_list),
        },
        "notes": "SI-SDR on predicted overlap segments; ASR metrics available when enable-asr. Includes timing & RTF.",
    }

    # CPU stats
    eval_json["cpu"] = cpu_mon.stop()

    if args.enable_asr:

        def _aggregate(refs: List[str], hyps: List[str]) -> Dict[str, float]:
            if not refs:
                return {"count": 0}
            wers = [_wer(r, h) for r, h in zip(refs, hyps)]
            cers = [_cer(r, h) for r, h in zip(refs, hyps)]
            return {
                "count": len(refs),
                "wer_mean": round(float(np.mean(wers)), 4),
                "wer_median": round(float(np.median(wers)), 4),
                "cer_mean": round(float(np.mean(cers)), 4),
                "cer_median": round(float(np.median(cers)), 4),
            }

        eval_json["asr"] = {
            "overlap_mixture": _aggregate(overlap_mix_refs, overlap_mix_hyps),
            "overlap_separated": _aggregate(overlap_sep_refs, overlap_sep_hyps),
            "clean": _aggregate(clean_refs, clean_hyps),
        }

    with (out_dir / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)

    _log(f"Done. Wrote evaluation to {out_dir / 'evaluation.json'}")
    if sdr_list:
        _log(
            f"SI-SDR mean={np.mean(sdr_list):.2f}dB, SI-SDRi mean={np.mean(sdri_list):.2f}dB"
        )
    _log(f"OSD precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} iou={iou:.3f}")


if __name__ == "__main__":
    main()
