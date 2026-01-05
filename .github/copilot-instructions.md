# Copilot Instructions for Audio Classification

## Project Overview

This is an **audio classification & speech processing pipeline** project with four main branches:

1. **Speaker Identification**: VAD + Non-streaming ASR pipeline with speaker recognition
2. **Overlapped Speech Handling (OSD)**: Detects overlapped speech, separates speakers, then runs ASR
3. **3-Source Separation**: Libri3Mix experiments using OSD + 3-source separation + target speaker filtering
4. **Streaming ASR**: Real-time speech recognition with Partial/Final results, VAD-based segmentation

**Core Workflow**: Input mixed audio → VAD/OSD detection → source separation → speaker verification → ASR transcription

## Architecture & Key Components

### Model Components (`src/`)

- **`model.py`**: ASR factory (Paraformer/SenseVoice/Transducer), speaker embedding extraction, L2 normalization
- **`osd/osd.py`**: `OverlapAnalyzer` class wrapping pyannote.audio OSD pipeline (required dependency)
- **`osd/separation.py`**: `Separator` class using Asteroid Conv-TasNet for 2/3-source speech separation
- **`osd/dataset.py`**: Libri3Mix/LibriMix dataset handling
- **`osd/streaming_asr.py`**: `StreamingASR` and `VADStreamingASR` classes for incremental ASR with Partial/Final results

### Pipeline Runners (`scripts/`)

- **`benchmark_pipeline.py`**: Speaker ID + ASR benchmarking with enrollment/test, outputs metrics (accuracy, RTF, CER)
- **`osd/offline_overlap_3src.py`**: Offline 3-source OSD pipeline (file mode or LibriMix dataset mode)
- **`osd/overlap3_core.py`**: **Core compute logic** (OSD → separation → target filtering → ASR), returns `PipelineResult` dataclass
- **`osd/streaming_overlap3_core.py`**: Streaming variant with audio buffer and queue-based processing
- **`osd/optimized_streaming_overlap3_core.py`**: Optimized streaming pipeline using direct separation (no OSD)
- **`osd/vad_streaming_overlap3_core.py`**: VAD-based streaming pipeline with natural speech boundary detection
- **`osd/streaming_asr_pipeline.py`**: **Streaming ASR Pipeline** with Partial/Final result support
- **`osd/demo_streaming_asr.py`**: Demo script for streaming ASR visualization
- **`osd/evaluate_with_sources.py`**: Libri2Mix evaluation with optional ASR comparison

### Demo Scripts (`demo_video/`)

- **`demo_single.py`**: Single-file processing demo
- **`demo_streaming.py`**: Streaming processing demo with real-time output
- **`demo_streaming.sh`**: Shell wrapper supporting fixed-interval and VAD modes

## Critical Patterns & Conventions

### Separation of Concerns

- **Core compute** in `overlap3_core.Overlap3Pipeline.run()` excludes I/O timing
- **File I/O** (writing JSON/CSV) happens in caller (`offline_overlap_3src.py`)
- This separation allows accurate performance metrics (RTF, compute time) without I/O interference

### Speaker Embedding & Verification

- Uses Sherpa-ONNX speaker embedding extractor (`SpeakerEmbeddingManager`)
- L2-normalized cosine similarity with threshold (default 0.6) for target matching
- Can enroll from ground truth sources (dataset mode) or file-provided target (`--target-wav`)

### ASR Integration

Three ASR backends via `create_asr_model()`:

- Paraformer (single-model, fastest)
- SenseVoice (multilingual via Sherpa-ONNX)
- Transducer (Encoder-Decoder-Joiner, streaming-capable)

**Streaming ASR** (`StreamingASRPipeline`):

- Outputs **Partial results** during recognition (intermediate, may change)
- Outputs **Final results** when VAD detects sentence end (stable)
- Only VAD-based segmentation (no max duration forcing)
- Optimized: Partial results use sliding window (3s) to avoid O(n²) complexity

Use `--provider cuda|cpu|coreml` to control inference hardware.

### Output Formats

**3-Source Pipeline** (`offline_overlap_3src.py`):

- `segments.jsonl`: Per-segment results (start, end, kind, stream, text, sv_score)
- `segments.csv`: Same data, tabular format
- `summary.json`: Aggregated metrics (detection rate, separation quality, RTF)

**Benchmark Pipeline**:

- `detail.jsonl`: Per-utterance records (speaker, score, text, timings, RTF)
- `predictions.csv`: CSV version with additional CPU/GPU metrics
- `summary.json`: Accuracy, enrollment speakers, average timings

## Developer Workflows

### Installation

```bash
cd scripts
bash install.sh          # CPU by default
CPU=1 bash install.sh    # Force CPU-only
```

### Running 3-Source Pipeline (File Mode)

```bash
cd scripts/osd
python offline_overlap_3src.py \
  --input-wavs mix.wav \
  --target-wav target.wav \
  --spk-embed-model ../../models/speaker-recognition/model.onnx \
  --sense-voice ../../models/asr/model.onnx \
  --tokens ../../models/asr/tokens.txt \
  --provider cuda
```

### Running with LibriMix Dataset

```bash
python offline_overlap_3src.py \
  --librimix-root /path/to/LibriMix \
  --subset test \
  --sep-backend asteroid \
  --osd-backend pyannote
```

### Benchmarking Speaker ID + ASR

```bash
python benchmark_pipeline.py \
  --speaker-file speaker_list.txt \
  --test-list test_list.txt \
  --model models/speaker-recognition/model.onnx \
  --sense-voice models/asr/model.onnx \
  --tokens models/asr/tokens.txt
```

### Running Streaming ASR Demo

```bash
cd scripts
bash demo_streaming_asr.sh s1 0.3 0.5
# Args: <sample> <chunk_duration> <partial_interval>
```

### Running VAD-based Streaming Demo

```bash
cd demo_video
bash demo_streaming.sh s1 2.0 0.4 --vad
# --vad enables VAD-based segmentation
```

## External Dependencies & Integration

| Component         | Library          | Usage                               |
| ----------------- | ---------------- | ----------------------------------- |
| OSD               | `pyannote.audio` | Detects overlapped speech regions   |
| Separation        | `asteroid`       | Conv-TasNet speaker separation      |
| Speaker Embedding | `sherpa-onnx`    | ONNX-based speaker verification     |
| ASR               | `sherpa-onnx`    | SenseVoice/Paraformer transcription |
| Audio I/O         | `torchaudio`     | Loading/saving WAV files            |
| Monitoring        | `psutil`         | CPU/GPU metrics (optional)          |

## Testing & Evaluation

- **Resource Monitoring**: Use `--enable-metrics` to collect CPU/GPU utilization, wall-clock, and RTF (Real Time Factor)
- **RTF Calculation**: `total_compute_time / audio_duration`; <1.0 indicates real-time-capable processing
- **CER/WER**: Computed if `--ref-text-list` provided (character/word error rate vs. ground truth)

## Key File References

- Pipeline compute: [scripts/osd/overlap3_core.py](scripts/osd/overlap3_core.py)
- Streaming ASR pipeline: [scripts/osd/streaming_asr_pipeline.py](scripts/osd/streaming_asr_pipeline.py)
- VAD streaming pipeline: [scripts/osd/vad_streaming_overlap3_core.py](scripts/osd/vad_streaming_overlap3_core.py)
- Benchmark runner: [scripts/benchmark_pipeline.py](scripts/benchmark_pipeline.py)
- Model definitions: [src/model.py](src/model.py)
- Streaming ASR module: [src/osd/streaming_asr.py](src/osd/streaming_asr.py)
- OSD wrapper: [src/osd/osd.py](src/osd/osd.py)
- Separation module: [src/osd/separation.py](src/osd/separation.py)
