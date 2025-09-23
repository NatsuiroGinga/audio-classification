#!/usr/bin/env python3
"""
Generate a speaker list file for scene-2 audios, mirroring test-speaker.txt format.

Each line: <speaker_id> <absolute_path_to_wav>

Defaults:
- input dir: dataset/scenes/scene-2
- output file: dataset/scene-2-speaker.txt
- speaker id: SPK_0

Usage examples:
  python scripts/make_scene2_speaker.py
  python scripts/make_scene2_speaker.py --input-dir dataset/scenes/scene-2 --output dataset/scene-2-speaker.txt --speaker-id SPK_0
  python scripts/make_scene2_speaker.py --relative  # use relative paths instead of absolute
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def list_wavs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    # Match .wav (case-insensitive)
    wavs = sorted([p for p in input_dir.glob("*.wav")])
    return wavs


def write_speaker_list(
    wavs: list[Path],
    output_file: Path,
    speaker_id: str = "SPK_0",
    use_relative: bool = False,
):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()

    with output_file.open("w", encoding="utf-8") as f:
        for wav in wavs:
            path_str = str(wav if not use_relative else wav.relative_to(root))
            f.write(f"{speaker_id} {path_str}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate scene-2 speaker list file (like test-speaker.txt)"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dataset/scenes/scene-2"),
        help="Directory containing .wav files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/scene-2-speaker.txt"),
        help="Output speaker list file path",
    )
    parser.add_argument(
        "--speaker-id",
        type=str,
        default="SPK_0",
        help="Speaker ID to use for all entries",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use paths relative to CWD instead of absolute",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wavs = list_wavs(args.input_dir)
    if not wavs:
        raise SystemExit(f"No .wav files found under: {args.input_dir}")

    # Convert to absolute paths unless --relative
    wavs = [w.resolve() for w in wavs]
    write_speaker_list(wavs, args.output.resolve(), args.speaker_id, args.relative)
    print(
        f"Wrote {len(wavs)} entries to {args.output.resolve()} (speaker_id={args.speaker_id}, relative={args.relative})"
    )


if __name__ == "__main__":
    main()
