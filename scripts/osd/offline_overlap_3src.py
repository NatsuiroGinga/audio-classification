#!/usr/bin/env python3
"""
Offline OSD + 3-source Separation + ASR runner

- 将核心计算逻辑抽取到独立模块 overlap3_core.Overlap3Pipeline
- 本脚本仅负责参数解析与文件写入（JSONL/CSV/metrics/summary）
- 计时统计（time_*、rtf_*）均来自核心模块的计算时间，不包含本脚本的文件写入 I/O 时间
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import csv
import sys

# Make local directory importable to load overlap3_core without requiring a package
_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from overlap3_core import Overlap3Pipeline


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
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for reproducibility (>=0 to enable)",
    )
    # File-mode (bypass LibriMix) — process arbitrary mixture WAV(s)
    p.add_argument(
        "--input-wavs",
        nargs="+",
        default=None,
        help="Process given mixture WAV files directly (bypasses LibriMix). If set, --target-wav is required.",
    )
    p.add_argument(
        "--target-wav",
        default="",
        help="Enrollment audio WAV for the target speaker (REQUIRED in file mode).",
    )
    p.add_argument(
        "--refs-csv",
        default="",
        help="In file mode: CSV mapping mixture to reference sources with columns: mix,ref1,ref2[,ref3]. Enables separation quality evaluation.",
    )
    p.add_argument(
        "--ref-wavs",
        nargs="+",
        default=None,
        help="In file mode: reference source WAVs (2 or 3) when only a single mixture is provided. Enables separation quality evaluation.",
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

    # Target-speaker (auto from dataset)
    p.add_argument(
        "--spk-embed-model",
        required=True,
        help="Speaker embedding ONNX model path (REQUIRED)",
    )
    p.add_argument(
        "--sv-threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold (0~1)",
    )

    # Overlap handling
    p.add_argument("--min-overlap-dur", type=float, default=0.4)

    # Segment post-processing: make clean segments the complement of overlap (exclusive)
    p.add_argument(
        "--exclusive-segments",
        dest="exclusive_segments",
        action="store_true",
        help="Make clean segments the complement of merged overlap segments; ensures no time overlap between clean and overlap.",
    )
    p.add_argument(
        "--no-exclusive-segments",
        dest="exclusive_segments",
        action="store_false",
        help="Use raw window-expanded OSD labels; clean and overlap windows may overlap in time at boundaries.",
    )
    p.set_defaults(exclusive_segments=True)

    # Output / metrics
    p.add_argument("--out-dir", default="test/overlap3")
    p.add_argument("--enable-metrics", action="store_true")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--metrics-out", default="metrics.json")
    # Separation quality evaluation (optional)
    p.add_argument(
        "--eval-separation",
        action="store_true",
        help="Evaluate separation quality (SI-SDR / SI-SDRi) on predicted overlap segments using LibriMix references (K=3)",
    )
    p.add_argument(
        "--save-sep-details",
        action="store_true",
        help="Save per-overlap segment separation details CSV (sisdr/sisdri with PIT indices)",
    )
    p.add_argument(
        "--sep-details-out",
        default="overlap_sep_details.csv",
        help="Filename for per-segment separation evaluation CSV",
    )
    return p.parse_args()


def main():
    args = parse_args()

    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = base_out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    # Run compute-only pipeline
    pipeline = Overlap3Pipeline(args)
    result = pipeline.run()

    # Write per-segment outputs (I/O time is intentionally outside the pipeline)
    seg_jsonl_f = (out_dir / "segments.jsonl").open("w", encoding="utf-8")
    seg_csv_f = (out_dir / "segments.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(seg_csv_f)
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
            "target_src",
            "target_src_text",
        ]
    )
    for rec in result.segments:
        # JSONL
        seg_jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # CSV row
        csv_writer.writerow(
            [
                rec.get("wav", ""),
                f"{rec.get('start', 0):.3f}",
                f"{rec.get('end', 0):.3f}",
                rec.get("kind", ""),
                (rec.get("stream") if rec.get("stream") is not None else ""),
                rec.get("text", ""),
                f"{rec.get('asr_time', 0):.3f}",
                (rec.get("sv_score") if rec.get("sv_score") is not None else ""),
                rec.get("target_src", "") or "",
                rec.get("target_src_text", ""),
            ]
        )
    seg_jsonl_f.close()
    seg_csv_f.close()

    # Optional: write separation details CSV
    if getattr(args, "eval_separation", False) and getattr(
        args, "save_sep_details", False
    ):
        with (out_dir / args.sep_details_out).open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "wav",
                    "start",
                    "end",
                    "k_refs",
                    "sisdr",
                    "sisdri",
                    "selected_pred_indices",
                ]
            )
            for row in result.sep_details_rows:
                w.writerow(row)

    # Compose and write metrics/summary (metrics are compute-only)
    metrics: Any = result.metrics
    summary: Any = {
        "segments": metrics.get("segments_total"),
        "dataset": result.dataset_name,
        "subset": result.subset,
        "num_speakers": 3,
        "sample_rate": result.sample_rate,
        "processed_mixtures": result.processed_mixtures,
        "notes": "ASR only; overlap segments separated into 3 branches; no CER.",
        # Target hit/miss summary for convenience
        "target_hits_segments": metrics.get("segments_matched"),
        "target_misses_segments": metrics.get("segments_missed"),
        "target_hits_clean_segments": metrics.get("segments_clean"),
        "target_misses_clean_segments": metrics.get("segments_missed_clean"),
        "target_hits_overlap_segments": metrics.get("segments_overlap_streams"),
        "target_misses_overlap_segments": metrics.get("segments_missed_overlap"),
    }
    if getattr(args, "enable_metrics", False):
        with (out_dir / args.metrics_out).open("w", encoding="utf-8") as mf:
            json.dump(metrics, mf, ensure_ascii=False, indent=2)
        summary["metrics"] = metrics
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"Done. segments={metrics.get('segments_total')}, mixtures={result.processed_mixtures}, out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
