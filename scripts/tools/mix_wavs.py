#!/usr/bin/env python3
"""
Mix multiple single-speaker WAV files into one mixture with optional offsets and SNR/gain control.

Features
- Any number of sources
- Per-source start offsets (seconds)
- Relative SNRs (dB) w.r.t. a reference source, or per-source gains (dB)
- Auto resample to target sample rate
- Peak protection to avoid clipping
- Outputs mono mix (averaging channel if input is stereo)

Examples
1) Three speakers, align from t=0, set SNRs relative to source-0: src1 at -5 dB, src2 at -10 dB vs src0
   python3 mix_wavs.py s0.wav s1.wav s2.wav --out mix.wav --sr 16000 --snr 0,5,10

   Note: Positive numbers in --snr make the corresponding source quieter (relative to src0).

2) Add offsets (seconds): start s1 at 1.2s, s2 at 2.5s
   python3 mix_wavs.py s0.wav s1.wav s2.wav --out mix.wav --sr 16000 --snr 0,5,10 --offsets 0,1.2,2.5

3) Use absolute per-source gains (dB) instead of SNRs
   python3 mix_wavs.py s0.wav s1.wav --out mix.wav --sr 16000 --gains-db 0,-6

Outputs
- A single WAV file mix.wav at target sample rate.
"""
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    import soundfile as sf
except Exception as _e:  # pragma: no cover
    sf = None  # type: ignore

try:
    import torch
    import torchaudio
    import torchaudio.functional as AF
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    torchaudio = None  # type: ignore
    AF = None  # type: ignore


def _rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _to_mono(x: np.ndarray) -> np.ndarray:
    # Accept shapes: (T,), (C,T). Return (T,)
    if x.ndim == 2:
        if x.shape[0] > 1:
            x = np.mean(x, axis=0)
        else:
            x = x[0]
    return x.astype(np.float32, copy=False)


def _resample_np(wav: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return _to_mono(wav), sr
    if AF is None or torch is None:
        raise RuntimeError(
            "torchaudio is required for resampling; please install torchaudio."
        )
    t = torch.from_numpy(_to_mono(wav)).unsqueeze(0)  # (1,T)
    y = AF.resample(t, sr, target_sr).squeeze(0).contiguous().numpy()
    return y.astype(np.float32), target_sr


def _load_wav(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    # Prefer torchaudio for robust formats; fallback to soundfile
    if torchaudio is not None:
        wav_t, sr = torchaudio.load(path)
        wav = wav_t.numpy()
    elif sf is not None:
        wav, sr = sf.read(path, always_2d=True, dtype="float32")
        wav = wav.T  # (C,T)
    else:
        raise RuntimeError("Please install torchaudio or soundfile to load audio.")
    y, sr2 = _resample_np(wav, sr, target_sr)
    return y, sr2


def _parse_floats_csv(s: Optional[str], n: int, default: float) -> List[float]:
    if not s:
        return [default] * n
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals: List[float] = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            raise ValueError(f"Invalid float in list: {p}")
    if len(vals) == 1:
        return [vals[0]] * n
    if len(vals) != n:
        raise ValueError(f"List length mismatch: expected {n}, got {len(vals)}")
    return vals


def mix(
    inputs: List[str],
    out_path: str,
    sr: int = 16000,
    offsets: Optional[List[float]] = None,
    snr_dbs: Optional[List[float]] = None,
    gains_db: Optional[List[float]] = None,
    peak_limit: float = 0.98,
) -> None:
    n = len(inputs)
    if n == 0:
        raise ValueError("No input files provided")
    if offsets is None:
        offsets = [0.0] * n
    if len(offsets) != n:
        raise ValueError("offsets length must match number of inputs")
    if snr_dbs is not None and gains_db is not None:
        raise ValueError("Use either --snr or --gains-db, not both")

    # Load and resample
    waves: List[np.ndarray] = []
    lengths: List[int] = []
    for p in inputs:
        y, _ = _load_wav(p, sr)
        waves.append(y)
        lengths.append(y.shape[-1])

    # Compute placement indices with offsets (seconds)
    starts = [int(max(0.0, o) * sr) for o in offsets]
    total_len = max(s + l for s, l in zip(starts, lengths))

    # Compute scaling factors
    scales = np.ones(n, dtype=np.float64)
    if gains_db is not None:
        # Absolute per-source gains in dB
        if len(gains_db) != n:
            raise ValueError("gains-db length must match number of inputs")
        scales = 10.0 ** (np.asarray(gains_db, dtype=np.float64) / 20.0)
    elif snr_dbs is not None:
        if len(snr_dbs) != n:
            raise ValueError("snr length must match number of inputs")
        # SNRs are relative to source-0. Positive value makes the source quieter.
        # Desired RMS ratio: rms_i / rms_ref = 10^(-snr_i/20)
        ref_rms = _rms(waves[0])
        if ref_rms <= 0:
            ref_rms = 1e-3
        ratios = 10.0 ** (-np.asarray(snr_dbs, dtype=np.float64) / 20.0)
        # For i=0, snr_dbs[0] typically 0 => ratio 1.0
        rms_arr = np.asarray([max(_rms(w), 1e-6) for w in waves], dtype=np.float64)
        base_scales = ratios * (ref_rms / rms_arr)
        # We'll apply an extra global factor to respect peak_limit later
        scales = base_scales

    # Dry-run with global factor=1 to compute required attenuation for peak protection
    mix_tmp = np.zeros(total_len, dtype=np.float64)
    for i, w in enumerate(waves):
        s = starts[i]
        e = s + w.shape[-1]
        mix_tmp[s:e] += w.astype(np.float64) * scales[i]

    peak = float(np.max(np.abs(mix_tmp)) + 1e-12)
    if peak > peak_limit:
        g = peak_limit / peak
    else:
        g = 1.0

    mix_out = (mix_tmp * g).astype(np.float32)

    # Save
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if sf is None:
        raise RuntimeError(
            "soundfile is required to save WAV; please install soundfile."
        )
    sf.write(out_path, mix_out, sr)

    # Report
    ref = mix_out if np.any(mix_out) else None
    print(f"Saved mixture: {out_path}")
    print(f"- target SR     : {sr}")
    print(f"- duration (s)  : {len(mix_out)/sr:.3f}")
    print(f"- peak (before) : {peak:.4f}, global gain: {g:.4f}")
    print("- per-source scales (linear):", ", ".join(f"{s:.4f}" for s in scales))


def main():
    ap = argparse.ArgumentParser(description="Mix multiple WAVs into one mixture.")
    ap.add_argument("inputs", nargs="+", help="Input WAV files (mono or multi-channel)")
    ap.add_argument("--out", required=True, help="Output WAV file path")
    ap.add_argument("--sr", type=int, default=16000, help="Target sample rate (Hz)")
    ap.add_argument(
        "--offsets",
        default=None,
        help="Comma-separated seconds for each source (e.g., 0,1.2,2.5). If one value provided, applied to all.",
    )
    ap.add_argument(
        "--snr",
        default=None,
        help="Comma-separated SNRs (dB) relative to source-0 (e.g., 0,5,10). Positive makes sources quieter.",
    )
    ap.add_argument(
        "--gains-db",
        default=None,
        help="Comma-separated absolute per-source gains in dB (e.g., 0,-6,-12). Cannot be used with --snr.",
    )
    ap.add_argument(
        "--peak",
        type=float,
        default=0.98,
        help="Peak limit (linear 0..1) to avoid clipping after sum",
    )
    args = ap.parse_args()

    inputs = [str(p) for p in args.inputs]
    n = len(inputs)
    offsets = _parse_floats_csv(args.offsets, n, 0.0)
    snr_dbs = None
    gains_db = None
    if args.snr:
        snr_dbs = _parse_floats_csv(args.snr, n, 0.0)
    if args.gains_db:
        gains_db = _parse_floats_csv(args.gains_db, n, 0.0)

    mix(
        inputs=inputs,
        out_path=args.out,
        sr=int(args.sr),
        offsets=offsets,
        snr_dbs=snr_dbs,
        gains_db=gains_db,
        peak_limit=float(args.peak),
    )


if __name__ == "__main__":
    main()
