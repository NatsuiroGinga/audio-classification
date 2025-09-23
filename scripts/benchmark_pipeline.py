#!/usr/bin/env python3
"""Unified benchmarking script for Speaker ID + ASR pipeline.

Features:
- Enrollment from --speaker-file (same format: <spk> <wav>)
- Test from --test-list
- Measures per-utterance timings: sid_time, asr_time, total_time
- Computes RTF = total_time / audio_duration
- Collects CPU (process %) and GPU (nvidia-smi) utilization snapshots pre/post each utterance (optional)
- Outputs: JSONL (detail), CSV (predictions), summary JSON + text
- Optional reference text list for CER/WER (simple implementation; CER only if Chinese characters predominant)

GPU/CPU Monitoring:
- CPU: psutil.Process().cpu_percent(interval=None) sampled twice (before->after); per-utterance delta approximated.
- GPU: parses `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits` if available.
  If multiple GPUs, takes the first by default or specified via --gpu-index.

Reference text list usage:
- Mode A (original): file lines formatted as `<wav_path>\t<ref_text>` matching absolute/relative wav path in test list.
- Mode B (dataset transcription like test_transcription): lines formatted as `<utt_id><space><ref_text>` where `utt_id` equals wav basename without extension. Script will auto-map utt_id to actual wav path by scanning --test-list entries.
  Example: `3D_SPK_06850_010 我 爱 你` maps to any wav whose basename is `3D_SPK_06850_010.*` (wav). Provide this file via --ref-text-list pointing to dataset/transcription/test_transcription.

CER normalization:
- Before computing CER we remove all spaces and filter out punctuation, retaining only: CJK Unified Ideographs, basic Latin letters (a-zA-Z), digits 0-9.
- This mitigates inflated CER caused by per-character spaces in reference transcripts.

Progress bar:
- If tqdm is installed, a global progress bar ("Benchmark") over all test utterances is shown.
- If not installed, it silently falls back to normal iteration.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterable
import re

import numpy as np
import soundfile as sf
import sherpa_onnx
from src.model import SpeakerASRModels  # newly extracted class

# Progress bar (optional)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):  # fallback noop
        return it

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

# ----------------- Helpers -----------------

g_sample_rate = 16000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Speaker/ASR common
    p.add_argument('--speaker-file', required=True, help='Enrollment list <spk> <wav>')
    p.add_argument('--test-list', required=True, help='Test list <spk> <wav>')
    p.add_argument('--model', required=True, help='Speaker embedding ONNX')
    p.add_argument('--silero-vad-model', required=False, default='', help='(Unused here) VAD model path for future streaming extension')
    p.add_argument('--threshold', type=float, default=0.5, help='Speaker match threshold')
    p.add_argument('--num-threads', type=int, default=1)
    p.add_argument('--provider', type=str, default='cpu', help='cpu|cuda|coreml')
    p.add_argument('--debug', action='store_true')

    # ASR model (subset; keep simplest path: Paraformer OR SenseVoice OR Transducer)
    p.add_argument('--paraformer', default='', help='Paraformer model.onnx')
    p.add_argument('--sense-voice', default='', help='SenseVoice model.onnx')
    p.add_argument('--encoder', default='', help='Transducer encoder')
    p.add_argument('--decoder', default='')
    p.add_argument('--joiner', default='')
    p.add_argument('--tokens', default='', help='tokens.txt path')
    p.add_argument('--decoding-method', default='greedy_search')
    p.add_argument('--feature-dim', type=int, default=80)
    p.add_argument('--language', default='auto')

    # Reference text for WER/CER
    p.add_argument('--ref-text-list', default='', help='Optional: file with <wav>\t<ref_text> for CER/WER')

    # Output
    p.add_argument('--out-dir', default='test', help='Base output dir (will create timestamp subdir)')

    # GPU/CPU monitoring
    # GPU monitoring removed per user request

    # Enrollment / caching
    p.add_argument('--emb-cache-dir', default='', help='Cache directory for per-wav speaker embeddings (*.npy)')
    p.add_argument('--save-speaker-embeds', default='', help='Save aggregated speaker embeddings (npz) to this path')
    p.add_argument('--load-speaker-embeds', default='', help='Load precomputed speaker embeddings (npz) and skip raw wav enrollment')

    # CPU/GPU metrics options
    p.add_argument('--cpu-normalize', action='store_true', help='Normalize process CPU percent by logical core count (value / cores)')
    # Removed GPU sampling options
    p.add_argument('--plot-cpu', action='store_true', help='Plot per-utterance CPU usage line chart (requires matplotlib)')

    return p.parse_args()


def load_pairs(path: str) -> Dict[str, List[str]]:
    d = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) != 2:
                raise ValueError(f'Bad line: {ln}')
            spk, wav = parts
            d[spk].append(wav)
    return d


def load_audio(fname: str) -> Tuple[np.ndarray, int, float]:
    data, sr = sf.read(fname, always_2d=True, dtype='float32')
    data = data[:, 0]
    samples = np.ascontiguousarray(data)
    dur = len(samples) / sr if sr else 0.0
    if sr != g_sample_rate and len(samples) > 1:
        tgt_n = int(round(len(samples) * g_sample_rate / sr))
        if tgt_n > 1:
            old_idx = np.arange(len(samples), dtype=np.float64)
            new_idx = np.linspace(0, len(samples) - 1, tgt_n, dtype=np.float64)
            samples = np.interp(new_idx, old_idx, samples).astype(np.float32)
            sr = g_sample_rate
    return samples, sr, dur


def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# ---- Text normalization for CER ----
import re as _re
_cjk_re = _re.compile(r'[\u4e00-\u9fff]')
_alnum_re = _re.compile(r'[A-Za-z0-9]')

def normalize_for_cer(text: str) -> str:
    if not text:
        return ''
    text = text.replace(' ', '').strip()
    return ''.join(ch for ch in text if _cjk_re.match(ch) or _alnum_re.match(ch))

## SpeakerASRModels moved to src/model.py

class BenchmarkRunner:
    """执行测试并输出指标与文件。"""
    def __init__(self, args: argparse.Namespace, models: SpeakerASRModels):
        self.args = args
        self.models = models
        self.proc = psutil.Process(os.getpid()) if psutil else None
        self.detail_records: List[Dict[str, Any]] = []
        self.rows_csv: List[List[str]] = []
        self.metrics: Dict[str, Any] = {}
        self._durations: List[float] = []
        self._sid_times: List[float] = []
        self._asr_times: List[float] = []
        self._total_times: List[float] = []
        self._rtfs: List[float] = []
        self._cer_vals: List[float] = []
        # per-utterance CPU snapshots (before/after)
        self._cpu_before_seq: List[Optional[float]] = []
        self._cpu_after_seq: List[Optional[float]] = []
        # prime cpu_percent to avoid first-call 0.0 anomaly
        if self.proc:
            try:
                self.proc.cpu_percent(None)
            except Exception:
                pass
        self.total = 0
        self.correct = 0
        self.unknown = 0
        self.total_items = 0  # 全部待处理数量
        self._last_report_time = time.time()
        self._report_interval_sec = 5.0  # 至少每5秒打印一次

    def set_total_items(self, n: int):
        self.total_items = n

    # ---- Sampling helpers ----
    def sample_cpu(self) -> Optional[float]:
        if not self.proc:
            return None
        val = sample_cpu(self.proc)
        if val is None:
            return None
        if self.args.cpu_normalize:
            cores = os.cpu_count() or 1
            return val / cores
        return val

    # GPU sampling removed

    # ---- Core loop ----
    def process_one(self, spk_true: str, wav: str, refs: Dict[str, str]):
        samples, sr, dur = load_audio(wav)
        t0 = time.time()
        cpu_before = self.sample_cpu()
        sid_start = time.time()
        pred, score = self.models.identify(samples, sr, self.args.threshold)
        sid_end = time.time()
        asr_start = time.time()
        text = self.models.asr_infer(samples, sr)
        asr_end = time.time()
        cpu_after = self.sample_cpu()
        sid_time = sid_end - sid_start
        asr_time = asr_end - asr_start
        total_time = asr_end - t0
        rtf = total_time / dur if dur > 0 else 0.0
        self.total += 1
        if pred == spk_true:
            self.correct += 1
        elif pred == 'unknown':
            self.unknown += 1
        ref_txt_raw = refs.get(wav, '')
        ref_txt = normalize_for_cer(ref_txt_raw) if ref_txt_raw else ''
        hyp_norm = normalize_for_cer(text)
        cer_val = cer(ref_txt, hyp_norm) if ref_txt else float('nan')
        if not math.isnan(cer_val):
            self._cer_vals.append(cer_val)
        self._durations.append(dur)
        self._sid_times.append(sid_time)
        self._asr_times.append(asr_time)
        self._total_times.append(total_time)
        self._rtfs.append(rtf)
        row = [
            wav,
            spk_true,
            pred,
            f'{score:.3f}',
            text,
            f'{dur:.3f}',
            f'{sid_time:.3f}',
            f'{asr_time:.3f}',
            f'{total_time:.3f}',
            f'{rtf:.3f}',
        ]
        def _fmt(x):
            return '' if x is None else f'{x:.3f}'
        cpu_before_v = _fmt(cpu_before)
        cpu_after_v = _fmt(cpu_after)
        # 记录 CPU 序列
        self._cpu_before_seq.append(cpu_before)
        self._cpu_after_seq.append(cpu_after)
        cer_v = '' if math.isnan(cer_val) else f'{cer_val:.3f}'
        row.extend([
            cpu_before_v,
            cpu_after_v,
            cer_v,
        ])
        self.rows_csv.append(row)
        self.detail_records.append({
            'wav': wav,
            'speaker_true': spk_true,
            'speaker_pred': pred,
            'score': score,
            'text': text,
            'text_norm': hyp_norm,
            'ref_text': ref_txt_raw,
            'ref_text_norm': ref_txt,
            'dur_sec': round(dur, 3),
            'sid_time': round(sid_time, 3),
            'asr_time': round(asr_time, 3),
            'total_time': round(total_time, 3),
            'rtf': round(rtf, 3),
            'cpu_before': None if cpu_before is None else round(cpu_before, 3),
            'cpu_after': None if cpu_after is None else round(cpu_after, 3),
            'cer': None if math.isnan(cer_val) else cer_val,
        })
        # 进度打印（节流）
        now = time.time()
        if now - self._last_report_time >= self._report_interval_sec:
            done = self.total
            pct = (done / self.total_items * 100.0) if self.total_items else 0.0
            acc_tmp = self.correct / done if done else 0.0
            avg_rtf_tmp = float(np.mean(self._rtfs)) if self._rtfs else 0.0
            print(f"[Progress] {done}/{self.total_items} ({pct:.1f}%) acc={acc_tmp:.3f} avg_rtf={avg_rtf_tmp:.3f}")
            self._last_report_time = now

    def finalize(self, start_all: float, out_dir: Path, model_path: str, asr_type: str) -> Dict[str, Any]:
        acc = self.correct / self.total if self.total else 0.0
        elapsed = time.time() - start_all
        summary = {
            'total_utts': self.total,
            'train_speakers': len(self.models.enrolled),
            'correct': self.correct,
            'unknown': self.unknown,
            'accuracy': round(acc, 3),
            'avg_sid_time': round(float(np.mean(self._sid_times)), 3) if self._sid_times else 0.0,
            'avg_asr_time': round(float(np.mean(self._asr_times)), 3) if self._asr_times else 0.0,
            'avg_total_time': round(float(np.mean(self._total_times)), 3) if self._total_times else 0.0,
            'p95_rtf': round(float(np.percentile(self._rtfs,95)), 3) if self._rtfs else 0.0,
            'avg_rtf': round(float(np.mean(self._rtfs)), 3) if self._rtfs else 0.0,
            'cer_mean': None if not self._cer_vals else round(float(np.mean(self._cer_vals)), 3),
            'duration_audio_sum_sec': round(float(np.sum(self._durations)), 3),
            'elapsed_wall_sec': round(elapsed, 3),
            'threshold': self.args.threshold,
            'model': model_path,
            'asr_model_type': asr_type,
            'output_dir': str(out_dir),
        }
        self.metrics = summary
        return summary

    def write_outputs(self, out_dir: Path):
        detail_path = out_dir / 'detail.jsonl'
        pred_csv_path = out_dir / 'predictions.csv'
        summary_json_path = out_dir / 'summary.json'
        # CSV
        with pred_csv_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['wav','speaker_true','speaker_pred','score','text','dur_sec','sid_time','asr_time','total_time','rtf','cpu_pct_before','cpu_pct_after','cer'])
            for row in self.rows_csv:
                w.writerow(row)
        # JSONL
        with detail_path.open('w', encoding='utf-8') as f:
            for rec in self.detail_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # Summary
        with (out_dir / 'summary.json').open('w', encoding='utf-8') as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)
        with (out_dir / 'summary.txt').open('w', encoding='utf-8') as f:
            f.write('Benchmark Summary\n')
            for k,v in self.metrics.items():
                f.write(f'{k}: {v}\n')
        print(f'Written detail: {detail_path}')
        print(f'Written predictions: {pred_csv_path}')
        print(f'Written summary: {summary_json_path}')
        # CPU usage CSV & plot
        if self._cpu_after_seq and self.args.plot_cpu:
            cpu_csv = out_dir / 'cpu_usage.csv'
            with cpu_csv.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['index','cpu_before','cpu_after'])
                for i,(b,a) in enumerate(zip(self._cpu_before_seq, self._cpu_after_seq)):
                    w.writerow([i,
                        '' if b is None else f'{b:.3f}',
                        '' if a is None else f'{a:.3f}'
                    ])
            print(f'Written CPU usage CSV: {cpu_csv}')
            try:
                import matplotlib
                matplotlib.use('Agg')  # headless
                import matplotlib.pyplot as plt
                xs = list(range(len(self._cpu_after_seq)))
                plt.figure(figsize=(10,3))
                plt.plot(xs, [a if a is not None else float('nan') for a in self._cpu_after_seq], label='cpu_after', linewidth=1.0)
                plt.plot(xs, [b if b is not None else float('nan') for b in self._cpu_before_seq], label='cpu_before', linewidth=0.8, alpha=0.6)
                plt.xlabel('Utterance Index')
                plt.ylabel('CPU Usage' + (' (normalized)' if self.args.cpu_normalize else ' (%)'))
                plt.title('Per-utterance CPU Usage')
                plt.legend()
                plt.tight_layout()
                fig_path = out_dir / 'cpu_usage.png'
                plt.savefig(fig_path, dpi=150)
                plt.close()
                print(f'Written CPU usage plot: {fig_path}')
            except Exception as e:
                print(f'[plot-cpu] Skip plot (matplotlib not available or error: {e})')

# Reference text loading

def load_refs(path: str, test_wavs: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Load reference texts and broadcast to all variant wavs of same core ID.

    Formats supported:
      1. TSV:  <wav_path>\t<ref_text>
      2. Utt-id + text: <utt_id><space><ref_text>

    Broadcast logic (Mode 2):
      Many datasets have multiple device/distance variants of the same base utterance id,
      e.g. 3D_SPK_06154_003_Device03_..., 3D_SPK_06154_003_Device06_... . We derive a
      *core id* from the first 4 underscore-separated tokens (speaker + index). The
      transcription lines usually only list this core id (e.g. 3D_SPK_06154_003), so we
      assign the same reference text to ALL wavs whose normalized core id matches.

    Printed statistics (Mode 2):
      - ref_lines_total:  行数量（有效解析的转写行）
      - core_ids_matched: 成功在测试集中找到的 core id 数量
      - wavs_assigned:    被赋予参考文本的 wav 数量
      - test_wavs_total:  测试集中 wav 总数
      - coverage_wavs(%): wav 覆盖率 (wavs_assigned / test_wavs_total)
      - avg_variants_per_core: 平均每个 core id 覆盖的 wav 数
    """
    if not path:
        return {}
    refs: Dict[str, str] = {}

    def norm_wav_basename(b: str) -> str:
        parts = b.split('_')
        return '_'.join(parts[:4]) if len(parts) >= 4 else b

    # Map: core_id -> list[wav]
    core_map: Dict[str, List[str]] = defaultdict(list)
    if test_wavs:
        for w in test_wavs:
            b = os.path.splitext(os.path.basename(w))[0]
            core = norm_wav_basename(b)
            core_map[core].append(w)

    ref_lines_total = 0
    core_ids_matched = 0
    wavs_assigned = 0
    seen_core_ids: set = set()

    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.rstrip('\n')
            if not ln:
                continue
            if '\t' in ln:  # Mode 1 direct mapping
                wav, txt = ln.split('\t', 1)
                refs[wav] = txt.strip()
                wavs_assigned += 1
            else:  # Mode 2
                parts = ln.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, txt = parts
                ref_lines_total += 1
                # direct core id match
                targets = core_map.get(utt_id)
                if not targets:
                    # prefix fallback (rare) - find any core starting with utt_id
                    for k in core_map.keys():
                        if k.startswith(utt_id):
                            targets = core_map[k]
                            break
                if targets:
                    txt_clean = txt.strip()
                    for w in targets:
                        # last one wins if duplicates; acceptable
                        refs[w] = txt_clean
                    wavs_assigned += len(targets)
                    if utt_id not in seen_core_ids:
                        core_ids_matched += 1
                        seen_core_ids.add(utt_id)

    if test_wavs:
        test_wavs_total = len(list(test_wavs)) if not isinstance(test_wavs, list) else len(test_wavs)
        avg_variants = (wavs_assigned / core_ids_matched) if core_ids_matched else 0.0
        coverage = (wavs_assigned / test_wavs_total * 100.0) if test_wavs_total else 0.0
        print(
            f"[load_refs] ref_lines_total={ref_lines_total} core_ids_matched={core_ids_matched} "
            f"wavs_assigned={wavs_assigned} test_wavs_total={test_wavs_total} "
            f"coverage_wavs={coverage:.1f}% avg_variants_per_core={avg_variants:.2f}"
        )
    return refs

# Simple CER (character error rate) using Levenshtein distance

def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    r = list(ref)
    h = list(hyp)
    dp = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1):
        dp[i][0] = i
    for j in range(len(h)+1):
        dp[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost,
            )
    return dp[-1][-1] / len(r)

# GPU / CPU sampling

import shutil  # retained for other potential usage; GPU sampling removed

def sample_cpu(proc: 'psutil.Process') -> Optional[float]:  # type: ignore
    if psutil is None:
        return None
    # cpu_percent without interval gives delta since last call
    return proc.cpu_percent(interval=None)


# ----------------- Main Benchmark -----------------

def main():
    args = parse_args()
    start_all = time.time()
    # 构建模型 & enrollment
    spk_map = load_pairs(args.speaker_file)
    test_map = load_pairs(args.test_list)
    print(f"Loaded enrollment speakers: {len(spk_map)}")
    models = SpeakerASRModels(args)
    print(f"Model initialized: {models}")
    enroll_start = time.time()
    models.enroll_from_map(spk_map, load_audio)
    enroll_time = time.time() - enroll_start
    print(f"Enrollment completed for {len(models.enrolled)} speakers. time={enroll_time:.2f}s")
    # Collect all test wav full paths for reference mapping
    all_test_wavs: List[str] = []
    for _, _wavs in test_map.items():
        all_test_wavs.extend(_wavs)
    print(f"Loaded test utterances: {len(all_test_wavs)}")
    refs = load_refs(args.ref_text_list, all_test_wavs)
    print(f"refs: \n{refs}")
    os._exit(0)
    if refs:
        print(f"Loaded references for CER: {len(refs)} lines")
    # 输出目录
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = Path(args.out_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = BenchmarkRunner(args, models)
    # Flatten test utterances for a global progress bar
    flat_items = [(spk, wav) for spk, wavs in test_map.items() for wav in wavs]
    runner.set_total_items(len(flat_items))
    print("Start benchmarking ...")
    for spk_true, wav in tqdm(flat_items, desc='Benchmark', unit='utt'):
        runner.process_one(spk_true, wav, refs)
    asr_type = (
        'paraformer' if args.paraformer else (
            'sense_voice' if args.sense_voice else (
                'transducer' if args.encoder else 'unknown'
            )
        )
    )
    runner.finalize(start_all, out_dir, args.model, asr_type)
    # augment summary with enrollment & cpu mode
    runner.metrics['enrollment_time_sec'] = round(enroll_time, 3)
    runner.metrics['cpu_mode'] = 'normalized' if args.cpu_normalize else 'raw'
    runner.write_outputs(out_dir)
    print("Done. Summary saved to:", out_dir)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
