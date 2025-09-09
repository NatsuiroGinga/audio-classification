#!/usr/bin/env python3
"""
Split a speaker list file into train/test sets.

Modes:
- speaker   : Split by speaker IDs (train/test speakers are disjoint)
- utterance : Stratified per speaker by utterances (default). Each speaker
              contributes some utterances to train and the rest to test.

Input file format (space-separated):
<speaker_id> <absolute_or_relative_audio_path>

Outputs:
- train file: lines whose speaker_id assigned to train
- test file: lines whose speaker_id assigned to test

Default split: 80% train / 20% test, deterministic with seed=42.

Example:
python ./split_speakers.py \
  --input ../dataset/speaker.txt \
  --train-out ../dataset/train-speaker.txt \
  --test-out ../dataset/test-speaker.txt \
  --train-ratio 0.8 \
  --seed 42 \
  --mode utterance
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split speaker list into train/test")
    p.add_argument("--input", "-i", type=Path, required=True, help="Path to input speaker.txt")
    p.add_argument("--train-out", type=Path, required=True, help="Output path for train-speaker.txt")
    p.add_argument("--test-out", type=Path, required=True, help="Output path for test-speaker.txt")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of speakers for train set (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--mode",
        type=str,
        choices=["speaker", "utterance"],
        default="utterance",
        help="Split mode: 'speaker' for disjoint speakers; 'utterance' for per-speaker stratified split",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert 0.0 < args.train_ratio < 1.0, "train-ratio must be in (0,1)"

    # Read all lines and collect unique speakers
    with args.input.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    # Group lines by speaker
    by_spk = {}
    for ln in lines:
        spk, *_ = ln.split(maxsplit=1)
        by_spk.setdefault(spk, []).append(ln)

    unique_speakers = sorted(by_spk.keys())
    if not unique_speakers:
        raise SystemExit("No speakers found in input file")

    rng = random.Random(args.seed)  # deterministic RNG

    if args.mode == "speaker":
        # Split speakers into train/test
        spk_shuffled = unique_speakers.copy()
        rng.shuffle(spk_shuffled)
        split_idx = int(len(spk_shuffled) * args.train_ratio)
        train_spk_set = set(spk_shuffled[:split_idx])
        test_spk_set = set(spk_shuffled[split_idx:])

        # Guard against empty split due to rounding
        if not train_spk_set and test_spk_set:
            spk = next(iter(test_spk_set))
            test_spk_set.remove(spk)
            train_spk_set.add(spk)
        if not test_spk_set and train_spk_set:
            spk = next(iter(train_spk_set))
            train_spk_set.remove(spk)
            test_spk_set.add(spk)

        train_lines = [ln for ln in lines if ln.split(maxsplit=1)[0] in train_spk_set]
        test_lines = [ln for ln in lines if ln.split(maxsplit=1)[0] in test_spk_set]

    else:  # utterance mode
        train_lines = []
        test_lines = []
        speakers_in_train = set()
        speakers_in_test = set()

        for spk in unique_speakers:
            utterances = by_spk[spk].copy()
            rng.shuffle(utterances)
            n = len(utterances)
            if n == 1:
                n_train = 1
            else:
                n_train = int(n * args.train_ratio)
                if n_train <= 0:
                    n_train = 1
                if n_train >= n:
                    n_train = n - 1

            train_part = utterances[:n_train]
            test_part = utterances[n_train:]

            if train_part:
                train_lines.extend(train_part)
                speakers_in_train.add(spk)
            if test_part:
                test_lines.extend(test_part)
                speakers_in_test.add(spk)

        # Safety: if somehow one side ended empty (e.g., very small dataset), rebalance a bit
        if not test_lines and train_lines:
            # move last utterance of the last speaker with >=2 train utterances
            for spk in reversed(unique_speakers):
                spk_trains = [ln for ln in train_lines if ln.split(maxsplit=1)[0] == spk]
                if len(spk_trains) >= 2:
                    mv = spk_trains[-1]
                    train_lines.remove(mv)
                    test_lines.append(mv)
                    speakers_in_test.add(spk)
                    break
        if not train_lines and test_lines:
            # move one utterance from test to train
            mv = test_lines[-1]
            test_lines.pop()
            train_lines.append(mv)
            speakers_in_train.add(mv.split(maxsplit=1)[0])

    # Ensure output dirs exist
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)

    # Write outputs
    with args.train_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + ("\n" if train_lines else ""))
    with args.test_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(test_lines) + ("\n" if test_lines else ""))

    # Print brief summary
    if args.mode == "speaker":
        print(f"Mode: speaker")
        print(f"Speakers: total={len(unique_speakers)}, train={len(train_spk_set)}, test={len(test_spk_set)}")
    else:
        # Count speakers that appear in each split
        spk_train = len({ln.split(maxsplit=1)[0] for ln in train_lines})
        spk_test = len({ln.split(maxsplit=1)[0] for ln in test_lines})
        print(f"Mode: utterance")
        print(f"Speakers: total={len(unique_speakers)}, in-train={spk_train}, in-test={spk_test}")

    print(f"Lines: train={len(train_lines)}, test={len(test_lines)}")
    print(f"Wrote: {args.train_out} and {args.test_out}")


if __name__ == "__main__":
    main()
