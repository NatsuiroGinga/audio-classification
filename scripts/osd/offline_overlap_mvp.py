#!/usr/bin/env python3
"""Offline MVP for overlapping speech handling: OSD + Separation + SID/ASR.

This script:
- Loads enrollment/test lists (same format as benchmark_pipeline)
- For each test utterance: run OSD to locate overlap regions
- For non-overlap regions: run regular SID+ASR
- For overlap regions: run 2-speaker separation, then SID+ASR per branch
- Write segments.jsonl with per-segment outputs, plus summary.json with CER and timings

Dependencies (required):
- pyannote.audio for high-quality OSD
- asteroid for Conv-TasNet separation
"""
import argparse
import csv
import json
import os
import time
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Iterable, Optional, Tuple

import numpy as np
import torch

# torchaudio is required for PyTorch-based audio I/O
try:
    import torchaudio
    import torchaudio.functional as AF
except Exception as _e:
    torchaudio = None  # will raise with guidance when used

from src.model import SpeakerASRModels
from src.osd import OverlapAnalyzer, Separator


def read_list(path: str) -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            spk, wav = ln.split()
            d.setdefault(spk, []).append(wav)
    return d


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


# ---- Text normalization for CER ----
import re as _re

_cjk_re = _re.compile(r"[\u4e00-\u9fff]")
_alnum_re = _re.compile(r"[A-Za-z0-9]")


def normalize_for_cer(text: str) -> str:
    if not text:
        return ""
    text = text.replace(" ", "").strip()
    return "".join(ch for ch in text if _cjk_re.match(ch) or _alnum_re.match(ch))


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    r = list(ref)
    h = list(hyp)
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(1, len(r))


def load_refs(path: str, test_wavs: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Load reference texts and broadcast to all variant wavs of same core ID.

    Formats supported:
      1. TSV:  <wav_path>\t<ref_text>
      2. Utt-id + text: <utt_id><space><ref_text>
    """
    if not path:
        return {}
    refs: Dict[str, str] = {}

    def norm_wav_basename(b: str) -> str:
        parts = b.split("_")
        return "_".join(parts[:4]) if len(parts) >= 4 else b

    core_map: Dict[str, List[str]] = defaultdict(list)
    if test_wavs:
        for w in test_wavs:
            b = os.path.splitext(os.path.basename(w))[0]
            core = norm_wav_basename(b)
            core_map[core].append(w)

    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            if "\t" in ln:  # Mode 1 direct mapping
                wav, txt = ln.split("\t", 1)
                refs[wav] = txt.strip()
            else:  # Mode 2 core-id mapping
                parts = ln.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, txt = parts
                targets = core_map.get(utt_id)
                if not targets:
                    for k in core_map.keys():
                        if k.startswith(utt_id):
                            targets = core_map[k]
                            break
                if targets:
                    txt_clean = txt.strip()
                    for w in targets:
                        refs[w] = txt_clean
    return refs


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--speaker-file", required=True)
    p.add_argument("--test-list", required=True)
    p.add_argument("--model", required=True, help="Speaker embedding ONNX")
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
    p.add_argument("--threshold", type=float, default=0.5)
    # References for CER
    p.add_argument(
        "--ref-text-list", default="", help="Optional: <wav>\t<text> or <utt_id> <text>"
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
    return p.parse_args()


def main():
    args = parse_args()
    # Base output directory (e.g., test_overlap)
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    # Create timestamped subdirectory similar to test/<timestamp>
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_device = _provider_to_torch_device(args.provider)

    # Build models
    models = SpeakerASRModels(args)
    enroll_map = read_list(args.speaker_file)
    models.enroll_from_map(enroll_map, lambda f: load_audio(f, device=torch_device))

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

    test_map = read_list(args.test_list)
    # Prepare refs
    all_test_wavs: List[str] = [w for _, lst in test_map.items() for w in lst]
    refs = load_refs(args.ref_text_list, all_test_wavs) if args.ref_text_list else {}

    seg_jsonl = (out_dir / "segments.jsonl").open("w", encoding="utf-8")
    pred_csv = (out_dir / "segments.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(pred_csv)
    csv_writer.writerow(
        [
            "wav",
            "speaker_true",
            "start",
            "end",
            "kind",
            "stream",
            "speaker_pred",
            "score",
            "text",
        ]
    )
    n_segments = 0
    t0_all = time.time()
    cer_vals: List[float] = []
    cer_vals_base: List[float] = []
    seg_counts = 0
    for spk_true, wavs in test_map.items():
        for wav_path in wavs:
            samples, sr = load_audio(wav_path, device=torch_device)
            dur = len(samples) / sr
            # Baseline whole-utterance ASR for CER
            base_text = models.asr_infer(samples, sr)
            base_ref_raw = refs.get(wav_path, "")
            base_ref = normalize_for_cer(base_ref_raw) if base_ref_raw else ""
            base_hyp = normalize_for_cer(base_text)
            if base_ref:
                cer_vals_base.append(cer(base_ref, base_hyp))
            # OSD
            segs = osd.analyze(samples, sr)
            if not segs:
                segs = [(0.0, dur, False)]
            # Iterate segments
            stitched_text_parts: List[str] = []
            for s, e, is_olap in segs:
                s_i = int(s * sr)
                e_i = int(e * sr)
                chunk = samples[s_i:e_i]
                if e - s <= 0:
                    continue
                if not is_olap or (e - s) < args.min_overlap_dur:
                    # regular path
                    sid_t0 = time.time()
                    pred, score = models.identify(chunk, sr, args.threshold)
                    sid_t1 = time.time()
                    asr_t0 = time.time()
                    text = models.asr_infer(chunk, sr)
                    asr_t1 = time.time()
                    rec = {
                        "wav": wav_path,
                        "speaker_true": spk_true,
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "kind": "clean",
                        "stream": None,
                        "speaker_pred": pred,
                        "score": float(score),
                        "text": text,
                        "sid_time": round(sid_t1 - sid_t0, 3),
                        "asr_time": round(asr_t1 - asr_t0, 3),
                    }
                    seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    csv_writer.writerow(
                        [
                            wav_path,
                            spk_true,
                            f"{s:.3f}",
                            f"{e:.3f}",
                            "clean",
                            "",
                            pred,
                            f"{score:.3f}",
                            text,
                        ]
                    )
                    n_segments += 1
                    seg_counts += 1
                    stitched_text_parts.append(text)
                else:
                    # overlap → separate
                    w1, w2 = sep.separate(chunk, sr)
                    branches = [w1, w2]
                    best_text = ""
                    best_score = -1e9
                    for k, w in enumerate(branches):
                        sid_t0 = time.time()
                        pred, score = models.identify(w, sr, args.threshold)
                        sid_t1 = time.time()
                        asr_t0 = time.time()
                        text = models.asr_infer(w, sr)
                        asr_t1 = time.time()
                        rec = {
                            "wav": wav_path,
                            "speaker_true": spk_true,
                            "start": round(s, 3),
                            "end": round(e, 3),
                            "kind": "overlap",
                            "stream": k,
                            "speaker_pred": pred,
                            "score": float(score),
                            "text": text,
                            "sid_time": round(sid_t1 - sid_t0, 3),
                            "asr_time": round(asr_t1 - asr_t0, 3),
                        }
                        seg_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        csv_writer.writerow(
                            [
                                wav_path,
                                spk_true,
                                f"{s:.3f}",
                                f"{e:.3f}",
                                "overlap",
                                k,
                                pred,
                                f"{score:.3f}",
                                text,
                            ]
                        )
                        n_segments += 1
                        seg_counts += 1
                        # Stitching policy for utterance CER: pick higher score branch
                        if float(score) > best_score:
                            best_score = float(score)
                            best_text = text
                    stitched_text_parts.append(best_text)

            # CER for stitched text (utterance-level)
            stitched_text = "".join(stitched_text_parts)
            ref_raw = refs.get(wav_path, "")
            ref_norm = normalize_for_cer(ref_raw) if ref_raw else ""
            hyp_norm = normalize_for_cer(stitched_text)
            if ref_norm:
                cer_vals.append(cer(ref_norm, hyp_norm))

    seg_jsonl.close()
    pred_csv.close()
    elapsed = time.time() - t0_all
    summary = {
        "segments": n_segments,
        "elapsed_wall_sec": round(elapsed, 3),
        "cer_mean_stitched": (
            None if not cer_vals else round(float(np.mean(cer_vals)), 3)
        ),
        "cer_mean_baseline": (
            None if not cer_vals_base else round(float(np.mean(cer_vals_base)), 3)
        ),
        "delta_cer": (
            None
            if not (cer_vals and cer_vals_base)
            else round(float(np.mean(cer_vals)) - float(np.mean(cer_vals_base)), 3)
        ),
        "notes": "CER computed at utterance-level using stitched segments (overlap branch chosen by higher SID score).",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(
        f"Done. segments={n_segments}, elapsed={elapsed:.3f}s, out_dir={out_dir} (timestamp={timestamp})"
    )


if __name__ == "__main__":
    main()
