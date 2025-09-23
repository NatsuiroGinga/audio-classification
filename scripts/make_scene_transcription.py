#!/usr/bin/env python3
"""Generate a transcription file from a directory of wav files.

Each output line follows the format:

    <utt_id> <text>

Where both utt_id and text equal the audio filename without extension.

Default input directory:
    dataset/scenes/scene-2

Default output file:
    dataset/transcription/scene-2_transcription

Notes:
- Only files with the .wav extension (case-insensitive) are considered.
- Lines are sorted by the basename for reproducibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def list_wav_basenames(input_dir: Path) -> List[str]:
    """List basenames (without extension) for all .wav files in the directory.

    Args:
        input_dir: Directory containing .wav files.

    Returns:
        Sorted list of basenames (filename without extension).
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    basenames: List[str] = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".wav":
            continue
        name = p.stem.strip()
        if not name:
            continue
        basenames.append(name)

    basenames.sort()
    return basenames


def write_transcription(basenames: List[str], output_path: Path, encoding: str = "utf-8") -> None:
    """Write transcription lines where text equals the basename.

    Args:
        basenames: List of filename stems.
        output_path: Target file path to write.
        encoding: File encoding, default utf-8.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding=encoding, newline="\n") as f:
        for name in basenames:
            # Format: <utt_id> <text>
            f.write(f"{name} {name}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transcription where text equals the WAV filename (without extension).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../dataset/scenes/scene-2"),
        help="Directory containing .wav files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../dataset/transcription/scene-2_transcription"),
        help="Output transcription file path",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="Encoding used to write the output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    basenames = list_wav_basenames(args.input_dir)
    write_transcription(basenames, args.output, args.encoding)
    print(f"Collected {len(basenames)} items from {args.input_dir}")
    print(f"Written transcription to: {args.output}")


if __name__ == "__main__":
    main()
