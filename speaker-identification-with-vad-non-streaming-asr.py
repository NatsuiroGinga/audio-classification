#!/usr/bin/env python3

"""
This script shows how to use Python APIs for speaker identification with
a microphone, a VAD model, and a non-streaming ASR model.

Please see also ./generate-subtitles.py

Usage:

(1) Prepare a text file containing speaker related files.

Each line in the text file contains two columns. The first column is the
speaker name, while the second column contains the wave file of the speaker.

If the text file contains multiple wave files for the same speaker, then the
embeddings of these files are averaged.

An example text file is given below:

    foo /path/to/a.wav
    bar /path/to/b.wav
    foo /path/to/c.wav
    foobar /path/to/d.wav

Each wave file should contain only a single channel; the sample format
should be int16_t; the sample rate can be arbitrary.

(2) Download a model for computing speaker embeddings

Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
to download a model. An example is given below:

    wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_zh_cnceleb_resnet34.onnx

Note that `zh` means Chinese, while `en` means English.

(3) Download the VAD model
Please visit
https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
to download silero_vad.onnx

For instance,

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

(4) Please refer to ./generate-subtitles.py
to download a non-streaming ASR model.

(5) Run this script

Assume the filename of the text file is speaker.txt.

python3 ./python-api-examples/speaker-identification-with-vad-non-streaming-asr.py \
  --silero-vad-model=/path/to/silero_vad.onnx \
  --speaker-file ./speaker.txt \
  --model ./wespeaker_zh_cnceleb_resnet34.onnx
"""
import argparse
import csv
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sherpa_onnx
import soundfile as sf

# sounddevice is optional for offline evaluation
try:  # pragma: no cover - optional dependency
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

g_sample_rate = 16000

def register_non_streaming_asr_model_args(parser):
    parser.add_argument(
        "--tokens",
        type=str,
        help="Path to tokens.txt",
    )

    parser.add_argument(
        "--encoder",
        default="",
        type=str,
        help="Path to the transducer encoder model",
    )

    parser.add_argument(
        "--decoder",
        default="",
        type=str,
        help="Path to the transducer decoder model",
    )

    parser.add_argument(
        "--joiner",
        default="",
        type=str,
        help="Path to the transducer joiner model",
    )

    parser.add_argument(
        "--paraformer",
        default="",
        type=str,
        help="Path to the model.onnx from Paraformer",
    )

    parser.add_argument(
        "--wenet-ctc",
        default="",
        type=str,
        help="Path to the CTC model.onnx from WeNet",
    )

    parser.add_argument(
        "--whisper-encoder",
        default="",
        type=str,
        help="Path to whisper encoder model",
    )

    parser.add_argument(
        "--whisper-decoder",
        default="",
        type=str,
        help="Path to whisper decoder model",
    )

    parser.add_argument(
        "--whisper-language",
        default="",
        type=str,
        help="""It specifies the spoken language in the input file.
        Example values: en, fr, de, zh, jp.
        Available languages for multilingual models can be found at
        https://github.com/openai/whisper/blob/main/whisper/tokenizer.py#L10
        If not specified, we infer the language from the input audio file.
        """,
    )

    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        choices=["transcribe", "translate"],
        type=str,
        help="""For multilingual models, if you specify translate, the output
        will be in English.
        """,
    )

    parser.add_argument(
        "--whisper-tail-paddings",
        default=-1,
        type=int,
        help="""Number of tail padding frames.
        We have removed the 30-second constraint from whisper, so you need to
        choose the amount of tail padding frames by yourself.
        Use -1 to use a default value for tail padding.
        """,
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Valid values are greedy_search and modified_beam_search.
        modified_beam_search is valid only for transducer models.
        """,
    )

    parser.add_argument(
        "--feature-dim",
        type=int,
        default=80,
        help="Feature dimension. Must match the one expected by the model",
    )

    parser.add_argument(
        "--sense-voice",
        default="",
        type=str,
        help="Path to sense voice model",
    )


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    register_non_streaming_asr_model_args(parser)

    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Language code (e.g., 'en', 'zh'), default: auto",
    )

    parser.add_argument(
        "--speaker-file",
        type=str,
        required=True,
        help="""Path to the speaker file. Read the help doc at the beginning of this
        file for the format.""",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the speaker embedding model file.",
    )

    parser.add_argument(
        "--silero-vad-model",
        type=str,
        required=True,
        help="Path to silero_vad.onnx",
    )

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of threads for neural network computation",
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="True to show debug messages",
    )

    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        help="Valid values: cpu, cuda, coreml",
    )

    parser.add_argument(
        "--test-list",
        type=str,
        default="dataset/test-speaker.txt",
        help="Path to test list (format: '<speaker> <wav_path>')",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="test",
        help="Directory to write predictions.csv and report.txt",
    )

    return parser.parse_args()


def assert_file_exists(filename: str):
    assert Path(filename).is_file(), (
        f"{filename} does not exist!\n"
        "Please refer to "
        "https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html to download it"
    )


def create_recognizer(args) -> sherpa_onnx.OfflineRecognizer:
    if args.encoder:
        assert len(args.paraformer) == 0, args.paraformer
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.encoder)
        assert_file_exists(args.decoder)
        assert_file_exists(args.joiner)

        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=args.encoder,
            decoder=args.decoder,
            joiner=args.joiner,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.paraformer:
        assert len(args.wenet_ctc) == 0, args.wenet_ctc
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.paraformer)

        recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=args.paraformer,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=g_sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.wenet_ctc:
        assert len(args.whisper_encoder) == 0, args.whisper_encoder
        assert len(args.whisper_decoder) == 0, args.whisper_decoder

        assert_file_exists(args.wenet_ctc)

        recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
            model=args.wenet_ctc,
            tokens=args.tokens,
            num_threads=args.num_threads,
            sample_rate=args.sample_rate,
            feature_dim=args.feature_dim,
            decoding_method=args.decoding_method,
            debug=args.debug,
        )
    elif args.whisper_encoder:
        assert_file_exists(args.whisper_encoder)
        assert_file_exists(args.whisper_decoder)

        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=args.whisper_encoder,
            decoder=args.whisper_decoder,
            tokens=args.tokens,
            num_threads=args.num_threads,
            decoding_method=args.decoding_method,
            debug=args.debug,
            language=args.whisper_language,
            task=args.whisper_task,
            tail_paddings=args.whisper_tail_paddings,
        )
    elif args.sense_voice:
        assert_file_exists(args.sense_voice)
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=args.sense_voice,
            tokens=args.tokens,
            num_threads=args.num_threads,
            use_itn=True,
            debug=args.debug,
            language=args.language
        )
    else:
        raise ValueError("Please specify at least one model")

    return recognizer


def load_speaker_embedding_model(args):
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.model,
        num_threads=args.num_threads,
        debug=args.debug,
        provider=args.provider,
    )
    if not config.validate():
        raise ValueError(f"Invalid config. {config}")
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    return extractor


def load_speaker_file(args) -> Dict[str, List[str]]:
    if not Path(args.speaker_file).is_file():
        raise ValueError(f"--speaker-file {args.speaker_file} does not exist")

    ans = defaultdict(list)
    with open(args.speaker_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}. Fields: {fields}")

            speaker_name, filename = fields
            ans[speaker_name].append(filename)
    return ans


def load_audio(filename: str) -> Tuple[np.ndarray, int]:
    data, sample_rate = sf.read(
        filename,
        always_2d=True,
        dtype="float32",
    )
    data = data[:, 0]  # use only the first channel
    samples = np.ascontiguousarray(data)
    # Ensure 16k sample rate to match extractor expectation
    if sample_rate != g_sample_rate and len(samples) > 1:
        target_n = int(round(len(samples) * g_sample_rate / sample_rate))
        if target_n > 1:
            old_idx = np.arange(len(samples), dtype=np.float64)
            new_idx = np.linspace(0, len(samples) - 1, target_n, dtype=np.float64)
            samples = np.interp(new_idx, old_idx, samples).astype(np.float32)
            sample_rate = g_sample_rate
    return samples, sample_rate


def compute_speaker_embedding(
    filenames: List[str],
    extractor: sherpa_onnx.SpeakerEmbeddingExtractor,
) -> np.ndarray:
    assert len(filenames) > 0, "filenames is empty"

    ans = None
    for filename in filenames:
        print(f"processing {filename}")
        samples, sample_rate = load_audio(filename)
        stream = extractor.create_stream()
        stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
        stream.input_finished()

        assert extractor.is_ready(stream)
        embedding = extractor.compute(stream)
        embedding = np.array(embedding)
        if ans is None:
            ans = embedding
        else:
            ans += embedding

    return ans / len(filenames) # type: ignore


def main():
    args = get_args()
    print(args)
    recognizer = create_recognizer(args)
    extractor = load_speaker_embedding_model(args)
    speaker_file = load_speaker_file(args)

    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    enrolled: Dict[str, np.ndarray] = {}
    for name, filename_list in speaker_file.items():
        embedding = compute_speaker_embedding(
            filenames=filename_list,
            extractor=extractor,
        )
        enrolled[name] = np.asarray(embedding, dtype=np.float32)
        status = manager.add(name, embedding)
        if not status:
            raise RuntimeError(f"Failed to register speaker {name}")

    # Pre-normalize enrolled embeddings for cosine scoring
    def _l2(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x)
        return x if n == 0 else x / n
    enrolled_norm = {k: _l2(v) for k, v in enrolled.items()}

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.25
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.sample_rate = g_sample_rate
    if not vad_config.validate():
        raise ValueError("Errors in vad config")

    window_size = vad_config.silero_vad.window_size

    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    samples_per_read = int(0.1 * g_sample_rate)  # 0.1 second = 100 ms

    print("Started offline evaluation from local wavs")

    # Offline evaluation: read test list and compute accuracy using manager.search
    test_list_path = Path(args.test_list)
    assert test_list_path.is_file(), f"{test_list_path} not found"
    print(f"Using test list: {test_list_path}")

    test_map = defaultdict(list)
    with open(test_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"Invalid line: {line}")
            spk, wav = fields
            test_map[spk].append(wav)

    total = 0
    correct = 0
    unknown_cnt = 0
    rows: List[Tuple[str, str, str, str, float]] = []  # (wav, true, pred, text, top1_score)

    for spk_true, wavs in test_map.items():
        for wav in wavs:
            samples, sample_rate = load_audio(wav)
            stream = extractor.create_stream()
            stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
            stream.input_finished()

            assert extractor.is_ready(stream)
            embedding = extractor.compute(stream)
            embedding = np.array(embedding, dtype=np.float32)
            emb_n = _l2(embedding)

            pred = manager.search(embedding, threshold=args.threshold)
            if not pred:
                pred = "unknown"

            # Now for non-streaming ASR on the same wav
            asr_stream = recognizer.create_stream()
            asr_stream.accept_waveform(sample_rate=sample_rate, waveform=samples)
            recognizer.decode_stream(asr_stream)
            text = asr_stream.result.text

            # Compute top-1 cosine score against all enrolled (for diagnostics)
            if enrolled_norm:
                names = list(enrolled_norm.keys())
                mat = np.stack([enrolled_norm[n] for n in names], axis=0)
                scores = mat @ emb_n
                top1_idx = int(np.argmax(scores))
                top1_score = float(scores[top1_idx])
            else:
                top1_score = float("nan")

            total += 1
            if pred == spk_true:
                correct += 1
            elif pred == "unknown":
                unknown_cnt += 1

            print(f"{total}: true={spk_true} pred={pred} text={text} file={Path(wav).name}")
            rows.append((str(wav), spk_true, pred, text, top1_score))

    acc = correct / total if total else 0.0
    print(f"Eval done. Accuracy: {acc:.4f} ({correct}/{total}), unknown: {unknown_cnt}")

    run_dir = write_eval_outputs(
        base_out_dir=Path(args.out_dir),
        rows=rows,
        train_speakers=len(enrolled),
        total=total,
        correct=correct,
        unknown_cnt=unknown_cnt,
        model=args.model,
        test_list_path=str(test_list_path),
        threshold=args.threshold,
    )
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")

def write_eval_outputs(
    *,
    base_out_dir: Path,
    rows: List[Tuple[str, str, str, str, float]],
    train_speakers: int,
    total: int,
    correct: int,
    unknown_cnt: int,
    model: str,
    test_list_path: str,
    threshold: float,
) -> Path:
    """Write predictions.csv 和 report.txt 到按时间戳创建的子目录下。

    返回创建的运行目录路径。
    """
    # e.g. 2025-09-09_14-03-27
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_out_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # predictions.csv
    pred_csv = run_dir / "predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["wav", "speaker_true", "speaker_pred", "text", "score"])
        for r in rows:
            w.writerow(r)

    # report.txt
    acc = (correct / total) if total else 0.0
    report_txt = run_dir / "report.txt"
    with report_txt.open("w", encoding="utf-8") as f:
        f.write("Speaker Identification Offline Evaluation\n")
        f.write(f"Train speakers: {train_speakers}\n")
        f.write(f"Test utterances: {total}\n")
        f.write(f"Top-1 accuracy: {acc:.4f} ({correct}/{total})\n")
        f.write(f"Unknown predicted: {unknown_cnt}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Test list: {test_list_path}\n")
        f.write(f"Threshold: {threshold}\n")

    return run_dir
