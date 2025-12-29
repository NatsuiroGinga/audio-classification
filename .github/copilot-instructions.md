# Audio Classification & Overlapped Speech Processing - Copilot Instructions

## Project Overview

This is an **audio speech processing pipeline** with three main components:

1. **Speaker Identification + VAD + ASR**: Main flow for speaker recognition with voice activity detection and non-streaming ASR (incomplete in favor of overlapped speech focus)
2. **Overlapped Speech Detection (OSD) MVP**: OSD + voice separation + ASR pipeline for overlapped speech scenarios, optimized for Libri2Mix/Libri3Mix datasets
3. **3-Speaker Separation**: Extended pipeline for 3-source speech separation using OSD + separation + target speaker filtering + ASR

**Core Insight**: The pipeline decomposes overlapped speech into: detect overlaps → separate sources → identify target speaker → transcribe (ASR). Most modules use ONNX/PyTorch inference for CPU/GPU efficiency.

## Architecture & Key Components

### Core Modules (`src/`)

**`src/model.py`** - ASR + Speaker Recognition Models

- Wraps `sherpa_onnx` for ASR (Paraformer, SenseVoice, or Transducer backends)
- Speaker embedding extraction & cosine similarity matching
- Global sample rate constant: `G_SAMPLE_RATE = 16000` (critical for all audio I/O)
- Key class: `SpeakerASRModels` handles multi-speaker setup

**`src/osd/osd.py`** - Overlapped Speech Detection (OSD)

- `OverlapAnalyzer` class enforces `pyannote.audio` backend (no fallback)
- Requires HuggingFace token via env vars: `PYANNOTE_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_TOKEN`
- Returns segments as `List[(start_sec, end_sec, is_overlap)]` covering full utterance
- Critical: Must initialize before use or raises `RuntimeError`

**`src/osd/separation.py`** - Voice Separation

- `Separator` class enforces Asteroid Conv-TasNet backend
- Supports 2-source (default) or 3-source (pre-trained on Libri3Mix) separation
- Auto-downloads weights from HuggingFace if checkpoint not provided
- Input/output: numpy arrays (torch used internally)

**`src/detection/model.py`** - Keyword Spotting (KWS) Models

- `KeywordSpotterModel` class wraps `sherpa_onnx.KeywordSpotter` for wake word detection
- Supports custom wake words in pinyin format (e.g., `n ǐ h ǎo zh ēn zh ēn @你好真真`)
- Key parameters: `keywords_threshold` (trigger sensitivity), `keywords_score`
- Factory function `create_kws_model()` auto-locates encoder/decoder/joiner from model dir

### Main Execution Scripts (`scripts/osd/`)

**`offline_overlap_3src.py`** - Production offline 3-speaker pipeline

- Entry point: parse LibriMix path + model paths → run `Overlap3Pipeline` core → write JSONL/CSV/metrics
- Metrics output: `metrics.json`, `results.jsonl`, `results.csv`, `summary.json`
- Key timing tracked: `time_osd`, `time_sep`, `time_asr` (excludes file I/O)
- Uses `--seed` for reproducible target speaker selection

**`offline_overlap_mvp.py`** - Simplified MVP for 2-speaker scenarios

**`streaming_overlap_3src.py`** - Streaming variant (incomplete, research phase)

**`evaluate_with_sources.py`** - Evaluation & metrics computation

- Compares ASR output against ground truth
- Computes SI-SDR for separation quality
- Outputs detailed match statistics (hit rates, missed segments)

### Helper Modules

**`batch_eval.py`** - Cross-run comparison & statistical aggregation

- Analyzes multiple result directories for RTF, accuracy, resource metrics
- Useful for benchmarking parameter variations

**`scripts/install.sh`** - Dependency installation

- Installs PyTorch (GPU/CPU via `CPU=1` env var)
- Downloads pre-trained models to `models/`
- Prompts for LibriMix dataset path (manual setup required)

## Critical Data Flows

### End-to-end OSD + 3-Separation + ASR Flow

```
Input Audio (16kHz, mono/stereo)
  ↓
[OSD] Detect overlap regions → List[(start, end, is_overlap)]
  ↓
[For each segment]:
  - If overlap: [Separate] 3-source separation → [SV Filter] cosine similarity → [ASR] transcribe
  - If clean: [SV Filter] check if target speaker (cosine sim ≥ threshold) → [ASR] transcribe
  ↓
Output: JSON/CSV with segment type, ASR text, SV scores, timings
```

### Model Initialization Pattern

All three core components follow similar lazy-loading + ONNX patterns:

- Models loaded at first use (not in **init**)
- Environment variables control backend selection (rarely needed—backends are forced)
- HuggingFace tokens critical for pyannote/Asteroid checkpoints
- Raises `RuntimeError` if dependencies missing (no silent fallbacks)

## Key Conventions & Project-Specific Patterns

### Sample Rate Contract

- **Always 16kHz**: hardcoded in `src/model.py` (`G_SAMPLE_RATE`)
- Audio I/O resamples to 16kHz automatically in dataset loaders
- Don't assume sample rate from file names; always check metadata

### ONNX/Sherpa-ONNX for ASR

- Input: numpy float32 waveform (single channel, 16kHz)
- Output: `sherpa_onnx.OfflineRecognitionResult` with `.text` property
- No streaming ASR inference in offline pipelines (sherpa-onnx provides both but only offline used in production)

### Environment Variable Injection Pattern

```python
# Common pattern: auto-inject HF token
if not self.auth_token:
    self.auth_token = (
        os.environ.get("PYANNOTE_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
```

Used in `OverlapAnalyzer`, `Separator`, and speaker embedding initialization.

### Metrics & Evaluation

- **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio): separation quality metric
- **Cosine Similarity**: target speaker filtering threshold (default 0.6)
- **RTF** (Real-Time Factor): `total_time / audio_duration`
- **Hit Rate**: fraction of ground-truth segments with matched ASR output

### Known Limitations (documented in `todo.md`)

1. **Clean segments**: Low match rate (~6%) due to short duration → unstable embeddings
2. **Short segment handling**: Recommend `--min-clean-sv-dur 0.8-1.2s` to skip very short clean segments
3. **Streaming incomplete**: Research branch, not production-ready

## Important File References

- **Datasets**: `dataset/` contains Libri2Mix metadata (speaker lists, scene info)
- **Models**: `models/speaker-recognition/`, `models/asr/` (auto-downloaded by `install.sh`)
- **Results**: Output written to timestamped dirs like `2025-11-26_14-51-01/` with JSON/CSV/metrics
- **Caching**: `cache/` stores pre-computed embeddings (numpy `.npy` files) for faster re-runs

## Build, Test & Development Commands

### Install (one-time setup)

```bash
cd scripts
bash install.sh
```

Set `CPU=1 bash install.sh` for CPU-only (slower but no GPU required).

### Run offline 3-speaker pipeline

```bash
cd scripts/osd
python3 offline_overlap_3src.py \
  --librimix-root /path/to/LibriMix \
  --spk-embed-model ../../models/speaker-recognition/your_model.onnx \
  --sense-voice ../../models/asr/your_asr.onnx \
  --tokens ../../models/asr/tokens.txt \
  --provider cuda --max-files 100
```

### Evaluate results

```bash
python3 scripts/osd/evaluate_with_sources.py \
  --results-dir path/to/results/dir \
  --librimix-root /path/to/LibriMix
```

### Batch comparison

```bash
python3 batch_eval.py result_dir_1 result_dir_2 result_dir_3
```

## When Modifying Core Logic

1. **OSD changes**: Edit `src/osd/osd.py::OverlapAnalyzer.analyze()` → test with `offline_overlap_3src.py --max-files 5`
2. **Separation changes**: Edit `src/osd/separation.py::Separator.separate()` → verify SI-SDR doesn't degrade
3. **ASR changes**: Edit `src/model.py::create_asr_model()` → ensure output still provides `.text` attribute
4. **New metrics**: Add fields to `metrics.json` structure in `scripts/osd/overlap3_core.py` (main pipeline logic)

All changes must preserve the (start_sec, end_sec, is_overlap) segment contract flowing between modules.

## Keyword Spotting (KWS) Module

### Overview

The KWS module (`src/detection/`) provides wake word detection using sherpa-onnx Zipformer Transducer models.

### Available Models

- `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`: Chinese KWS, 3.3M params
- `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`: Chinese-English KWS, 3M params

### Wake Word Format (Pinyin)

```python
# Format: pinyin tokens (space-separated) + @ + display text
KEYWORD_NIHAO_ZHENZHEN = "n ǐ h ǎo zh ēn zh ēn @你好真真"
```

### Key Evaluation Metrics

1. **FRR (False Rejection Rate)**: Miss rate = N_miss / N_target × 100%
   - Target: FRR < 5% when SNR > 10dB
2. **FA/Hr (False Alarms per Hour)**: N_false / T_hours
   - Target: < 1 FA/Hr (general) or < 1 FA/24Hr (strict)
3. **RTF (Real-Time Factor)**: T_process / T_audio
   - Target: RTF < 0.1 for edge devices, RTF < 1.0 for real-time

### Usage Examples

```bash
# Quick test with single audio file
cd scripts/detection
python test_nihao_zhenzhen.py --wav /path/to/test.wav

# Full benchmark with positive/negative datasets
python benchmark_kws.py \
  --model-dir ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
  --positive-dir /path/to/positive_samples \
  --negative-dir /path/to/negative_samples \
  --keyword "n ǐ h ǎo zh ēn zh ēn @你好真真"
```

### Model Parameter Tuning

- `keywords_threshold`: Higher = harder to trigger (default 0.25)
- `keywords_score`: Weight for keyword matching (default 1.0)
- `use_int8`: Use INT8 quantized models for faster inference

### Generating keywords.txt from keywords_raw.txt

Use `sherpa-onnx-cli text2token` or the provided Python scripts to convert human-readable keywords to model-compatible format:

```bash
# Method 1: Using sherpa-onnx-cli (for pure Chinese)
sherpa-onnx-cli text2token \
  --tokens models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
  --tokens-type ppinyin \
  keywords_raw.txt keywords.txt

# Method 2: Using generate_keywords.py (recommended)
cd scripts/detection
python generate_keywords.py --keyword "你好真真" \
  --tokens-file ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt

# Method 3: For zh-en mixed models (with Lexicon)
python generate_keywords_zh_en.py \
  --model-dir ../../models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 \
  --keyword "HELLO WORLD"
```

**Keywords format in keywords_raw.txt:**

```
你好真真 @你好真真
小爱同学 :2.0 #0.6 @小爱同学
HELLO WORLD @HELLO_WORLD
```

- `:score` - boosting score (optional)
- `#threshold` - trigger threshold (optional)
- `@text` - display text (recommended)
