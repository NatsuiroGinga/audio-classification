# Audio Classification & Overlapped Speech Processing - Copilot Instructions

## Project Overview

This is an **audio speech processing pipeline** with three main components:

1. **Speaker Identification + VAD + ASR**: Main flow for speaker recognition with voice activity detection and non-streaming ASR (incomplete in favor of overlapped speech focus)
2. **Overlapped Speech Detection (OSD) MVP**: OSD + voice separation + ASR pipeline for overlapped speech scenarios, optimized for Libri2Mix/Libri3Mix datasets
3. **3-Speaker Separation**: Extended pipeline for 3-source speech separation using OSD + separation + target speaker filtering + ASR
4. **Keyword Spotting (KWS)**: Wake word detection using Zipformer models with Chinese/English support

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
- Returns `KeywordDetection` objects with timestamp and confidence scores

**`src/detection/decoy_filter.py`** - Homophone False Positive Defense

- `DecoyFilter` class removes homophones/decoy words post-KWS detection
- Predefined sets for Chinese keywords (e.g., `DECOY_KEYWORDS_NIHAO_ZHENZHEN`)
- Uses exact string matching against tone/phoneme variations
- Tracks interception statistics for monitoring

**`src/detection/verifier.py`** - ASR-Based Keyword Verification

- `KeywordVerifier` class provides two-stage confirmation: KWS detection → ASR verification
- Compares ASR transcript against target keyword using text similarity + pinyin matching
- Resolves homophones by converting to pinyin (tone-insensitive)
- Configurable fuzzy matching thresholds (default 0.8 text similarity)
- Integrates with `src/model.py` ASR backend for transcription

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

### Environment Setup

**CRITICAL**: Always activate the `default` conda environment before running any terminal commands in this project:

```bash
conda activate default
```

All Python scripts, model inference, and audio processing operations depend on packages installed in the `default` environment. Failure to activate the environment will result in import errors or missing dependencies.

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
   - **FA_homophone** (homophones): Detections on confusable tone/sound variants (key metric)
3. **RTF (Real-Time Factor)**: T_process / T_audio
   - Target: RTF < 0.1 for edge devices, RTF < 1.0 for real-time

### False Positive Defense: Two-Stage Filtering

KWS models are susceptible to homophones (words with similar pronunciation but different meanings). The codebase provides **two independent filtering strategies**:

#### Strategy 1: DecoyFilter (Fast, Deterministic)

**Use Case**: When homophones are well-defined and static.

```python
from src.detection.decoy_filter import DecoyFilter, DECOY_KEYWORDS_NIHAO_ZHENZHEN

# Initialize with predefined homophones
filter = DecoyFilter(DECOY_KEYWORDS_NIHAO_ZHENZHEN, log_intercepted=True)

# Post-KWS filtering
detections = kws_model.detect(audio, sr)
filtered = filter.filter(detections)  # Removes exact matches

stats = filter.get_stats()  # {"total_intercepted": N, "intercepted_keywords": [...]}
```

**Advantages**: O(1) lookup, no ASR required, perfect precision  
**Disadvantages**: Only catches exact text matches (homophones must be pre-defined)

**Homophone Sets** (for "你好真真"):

- Tone variants: "你好镇镇" (zhèn 4th), "你好诊诊" (zhěn 3rd), "你好振振" (zhèn 4th)
- Nasal variants: "你好正正", "你好争争", "你好整整", "你好征征"
- Initial consonant variants: "你好认认" (r vs zh), "你好曾曾" (z vs zh)

#### Strategy 2: KeywordVerifier (Accurate, ASR-Based)

**Use Case**: When homophones are unknown or too numerous to pre-define.

```python
from src.detection.verifier import KeywordVerifier, VerifierConfig

config = VerifierConfig(
    keyword_text="你好真真",
    asr_model="models/asr/sensevoice.onnx",
    similarity_threshold=0.8,  # Text similarity required
    use_pinyin_match=True,  # Handle tone variations
)
verifier = KeywordVerifier(config)

# Post-KWS verification
detections = kws_model.detect(audio, sr)
for detection in detections:
    is_valid = verifier.verify(audio_segment, detection.start_time)  # ASR transcription + matching
    if is_valid:
        # Confirmed detection
```

**Verification Flow**:

1. ASR transcribes the audio segment
2. Compares transcript against target keyword
3. Uses **text similarity** (difflib) for direct match
4. Falls back to **pinyin matching** (tone-insensitive) for same-sound variations

**Advantages**: Handles unknown homophones, flexible scoring  
**Disadvantages**: Requires ASR inference (~0.5-1s per segment), slower

### Combined Workflow (Production)

```python
# 1. KWS detection (fast)
detections, rtf = kws_model.detect(audio, sr)

# 2. Decoy filtering (instant, high precision)
decoy_filter = create_nihao_zhenzhen_filter()
detections = decoy_filter.filter(detections)

# 3. ASR verification (slow, high recall)
verifier = KeywordVerifier(config)
confirmed = []
for detection in detections:
    if verifier.verify(audio, detection.start_time):
        confirmed.append(detection)

# 4. Report statistics
print(f"KWS: {len(detections_raw)}, Decoy filtered: {decoy_filter.get_stats()}, Verified: {len(confirmed)}")
```

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

# Parameter optimization (find best threshold + boost)
python scripts/detection/param_optimization.py --output-dir test/detection

# Demo with parameter presets (low-FRR, balanced, zero-FA)
python demo_wakeword.py /path/to/test.wav --config balanced
```

### Model Parameter Tuning

- `keywords_threshold`: Higher = harder to trigger (default 0.25)
  - Range: 0.0-1.0; typical sweep: [0.3, 0.4, 0.5, 0.6, 0.7]
  - **Trade-off**: Lower threshold → FRR↓, FA↑
- `keywords_score`: Weight for keyword matching (default 1.0)
  - Boost value (boost score): 0.3-1.5 range
- `use_int8`: Use INT8 quantized models for faster inference (RTF ↓ ~15%)

**Parameter Selection Guide** (from ablation studies):

| Config   | Boost | Threshold | FRR   | FA_homophone | Use Case        |
| -------- | ----- | --------- | ----- | ------------ | --------------- |
| default  | 1.0   | 0.25      | 2.1%  | 32           | balanced        |
| low-frr  | 1.5   | 0.45      | 1.4%  | 29           | minimize misses |
| balanced | 0.7   | 0.60      | 3.5%  | 15           | general         |
| zero-fa  | 0.3   | 0.40      | 13.9% | 0            | strict (demo)   |

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

## Code Patterns & Development Practices

### KWS Detection Pipeline Architecture

**Detection module layout** (`scripts/detection/`):

```
scripts/detection/
├── Core Evaluation:
│   ├── benchmark_kws.py           # Main eval: FRR/FA_homophone/RTF metrics
│   ├── ablation_experiment.py     # Comparisons: KWS vs KWS+verifier vs KWS+decoy
│   └── evaluate_with_sources.py   # Pair audio + ground truth evaluation
├── Parameter Optimization:
│   ├── param_optimization.py      # Grid search: threshold × boost sweep
│   ├── param_optimization_with_decoy.py  # Joint optimization with filter
│   └── optimize_params_refined.py # Advanced search with statistical confidence
├── Demo & Testing:
│   ├── test_nihao_zhenzhen.py    # Single-file detection test
│   ├── demo_wakeword.py          # Interactive demo with presets (default/low-frr/balanced/zero-fa)
│   └── demo_decoy_filter.py      # Decoy interception demo
├── Generation Tools:
│   ├── generate_keywords.py       # Text → pinyin → keywords.txt (Chinese)
│   └── generate_keywords_zh_en.py # Mixed zh-en keyword generation
└── Dataset Utilities:
    ├── data_generator.py         # Create synthetic test datasets
    └── analyze_misclassified.py  # Debug FP/FN patterns
```

**Key Script Usage Patterns**:

1. **Quick Iteration**:

   ```bash
   python test_nihao_zhenzhen.py --wav test_audio.wav
   ```

2. **Parameter Tuning** (find optimal threshold × boost):

   ```bash
   python param_optimization.py --output-dir results/
   # Outputs: threshold_results.json, param_optimization.log
   ```

3. **Comparative Evaluation** (measure FRR/FA improvement):

   ```bash
   python ablation_experiment.py
   # Tests: baseline, int8, int8+verifier, int8+decoy filters
   ```

4. **Production Validation** (FRR/FA against labeled datasets):
   ```bash
   python benchmark_kws.py \
     --model-dir models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
     --positive-dir /path/to/positive --negative-dir /path/to/negative
   # Outputs: benchmark_result.json with FRR%, FA/Hr, RTF metrics
   ```

### Lazy Loading Model Pattern

All core models (ASR, OSD, Separator, KWS) use lazy initialization:

```python
class MyModel:
    def __init__(self, config):
        self.config = config
        self._model = None  # NOT loaded in __init__

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()  # Load on first access
        return self._model
```

**Why**: Avoid blocking startup time when model not used, reduces memory until needed.

### Segment Contract Pattern

All segment-based logic preserves this tuple contract flowing through the pipeline:

```python
# Type: List[Tuple[float, float, bool]]
# (start_time_sec, end_time_sec, is_overlap_flag)
segments = [
    (0.0, 1.5, False),   # clean segment
    (1.5, 3.2, True),    # overlapped segment
    (3.2, 4.8, False),
]
```

**Critical**: When modifying OSD, Separator, or ASR modules, preserve this contract or pipeline breaks.

### NumPy/Torch Interop Pattern

```python
# Audio I/O always uses numpy (float32, -1.0..1.0 range)
audio_np = np.load("audio.npy")

# Internal torch processing (optional device placement)
audio_torch = torch.from_numpy(audio_np).to(device)
output_torch = model(audio_torch)

# Always convert back to numpy for downstream
result_np = output_torch.cpu().numpy()
```

**Convention**: Public APIs accept/return numpy; internal methods may use torch.

### Embedding Caching Pattern

Speaker embeddings are cached to `cache/` as `.npy` files with predictable naming:

```python
# Cache key format: "{audio_filename_stem}.npy"
# Example: "3D_SPK_06154_001_Device03_Distance00_Dialect00.npy"
cache_path = os.path.join("cache", f"{stem}.npy")
if os.path.exists(cache_path):
    embedding = np.load(cache_path)  # Use cached
else:
    embedding = model.encode(audio)   # Compute & save
    np.save(cache_path, embedding)
```

**Benefit**: Re-runs with same audio avoid re-computing embeddings.

## Common Development Scenarios

### Scenario 1: Fixing Low Clean Segment Match Rate

**Problem**: Clean (non-overlapped) segments have ~6% match rate (see `todo.md`).

**Root causes**:

- Short segment duration → unstable speaker embeddings
- Fixed cosine similarity threshold too strict for short audio

**Solutions** (in priority order):

1. Add `--min-clean-sv-dur` parameter (0.8-1.2s) to skip micro-segments
2. Per-segment dynamic thresholding based on duration
3. Sliding window aggregation before embedding extraction

**Files to modify**: `scripts/osd/overlap3_core.py`, argument parser in `offline_overlap_3src.py`

### Scenario 2: Adding New ASR Backend

**Current**: Supports Paraformer, SenseVoice, Transducer via sherpa-onnx.

**To add new backend**:

1. Verify backend available in `sherpa-onnx` release
2. Update `src/model.py::create_asr_model()` to instantiate new backend
3. **Critical**: Ensure return type has `.text` attribute (required by pipeline)
4. Test with `offline_overlap_3src.py --max-files 1`
5. Update `README.md` with new model path examples

### Scenario 3: Optimizing for Edge Devices

**Approach**: Reduce RTF (real-time factor) for 3-speaker separation.

**Levers**:

- Use INT8 quantized ASR models (`--use-int8-asr`)
- Reduce OSD confidence threshold to merge short silence gaps
- Batch N audio files instead of per-file processing
- Profile with `--max-files 10` to find bottleneck

**Files**: `scripts/osd/offline_overlap_3src.py`, `src/model.py`, `src/osd/separation.py`

### Scenario 4: Handling Multi-Speaker Clean Segments

**Current limitation**: Clean segments assume single target speaker.

**Extension path**:

1. In OSD output, add `num_speakers` field (estimated via spectral entropy)
2. For clean segments with 2+ speakers, apply 2-way separation instead of skipping
3. Run SV filter post-separation to pick target stream

**Files**: `src/osd/osd.py::OverlapAnalyzer.analyze()`, `overlap3_core.py`

## Dependency & Environment Notes

### Critical Environment Variables

```bash
# HuggingFace token (REQUIRED for OSD + Separation)
export HF_TOKEN="hf_xxx"  # or HUGGINGFACE_TOKEN or PYANNOTE_TOKEN

# GPU Provider (optional, defaults to CPU)
export ONNX_PROVIDER="cuda"  # or "tensorrt", "openvino", "cpu"

# LibriMix dataset path
export LIBRIMIX_ROOT="/path/to/LibriMix"
```

### Optional Plugins

- **onnxruntime-gpu**: For ONNX model acceleration (auto-installed by `install.sh`)
- **pytorch-cuda**: GPU support for torch (set `CPU=1` to skip)
- **sherpa-onnx**: Wraps ASR models, speech endpoints (auto-installed)

### Troubleshooting Model Loading

```python
# If model fails to load:
1. Check HF_TOKEN is valid: `huggingface-cli login`
2. Verify cache dir writable: `ls -la ~/.cache/huggingface/`
3. For OSD: confirm pyannote-audio installed: `pip show pyannote.audio`
4. For Separator: verify asteroid installed: `pip show asteroid`
```

## Performance Profiling

### Generate Timing Breakdown

Results include per-stage timing (exclude file I/O):

```json
{
  "time_osd": 2.315, // OSD detection phase
  "time_sep": 2.782, // Voice separation phase
  "time_asr": 13.635, // ASR transcription phase
  "time_total": 18.732
}
```

**Identify bottleneck**: If `time_asr >> time_osd + time_sep`, consider ASR model swap.

### Batch Comparison

```bash
python batch_eval.py result_dir_1 result_dir_2 result_dir_3
```

Outputs aggregate stats across multiple runs (useful for before/after optimization).

## Known Limitations & TODOs

Detailed tracking in [todo.md](../../todo.md):

1. **Clean Segment Instability** (HIGH PRIORITY)

   - Short duration + fixed threshold → 6% match rate
   - **Fix**: Adaptive thresholding + minimum duration filter

2. **Streaming Incomplete** (RESEARCH)

   - `streaming_overlap_3src.py` not production-ready
   - Buffering/latency not optimized

3. **Multi-Speaker Clean** (FUTURE)

   - Currently assume clean segments = 1 speaker
   - Need 2-way separation fallback for polyphone clean segments

4. **Memory Footprint** (OPTIMIZATION)
   - 3-way separation requires ~3GB peak memory
   - Consider chunk-wise processing for <1GB devices

## Testing & Debugging Best Practices

### Quick Sanity Check

```bash
# Test OSD in isolation (2 seconds of audio should return in <5 seconds)
cd scripts/osd
python3 offline_overlap_3src.py --librimix-root /path/to/LibriMix --max-files 1 --seed 42

# Verify all model paths are correct
python3 -c "from overlap3_core import Overlap3Pipeline; print('OSD initialized')"
```

### Debug Mode Development

```bash
# Run with detailed logging (set environment)
export LOG_LEVEL=DEBUG
python3 offline_overlap_3src.py --max-files 5 --provider cpu

# Output files will appear in timestamped dir with detailed metrics per-file
# Check results.jsonl for segment-level timing & scores
```

### Profiling & Bottleneck Analysis

```python
# Add timing instrumentation in core modules:
import time
start = time.perf_counter()
# ... operation ...
elapsed = time.perf_counter() - start
print(f"Operation took {elapsed:.3f}s")
```

**Key files to instrument**:

- `src/osd/osd.py::OverlapAnalyzer.analyze()` - OSD detection time
- `src/osd/separation.py::Separator.separate()` - Voice separation time
- `src/model.py::recognize()` - ASR inference time

### Common Errors & Fixes

| Error                               | Cause                            | Fix                                                                              |
| ----------------------------------- | -------------------------------- | -------------------------------------------------------------------------------- |
| `RuntimeError: OSD not initialized` | Missing HF_TOKEN                 | Set `export HF_TOKEN="hf_..."`                                                   |
| `KeyError: 'text'` in ASR output    | Wrong ASR backend used           | Verify `--paraformer`, `--sense-voice` or `--encoder` + `--decoder` + `--joiner` |
| `Low separation SI-SDR (<5 dB)`     | Wrong checkpoint or input format | Check `--sep-checkpoint` or use default Conv-TasNet-3src                         |
| `Memory OOM during separation`      | Dataset too large                | Use `--max-files 10` to limit batch size                                         |

### Writing New Tests

```python
# Template for module-level test
import pytest
from pathlib import Path
import numpy as np

def test_module_basic():
    """Smoke test: module initializes without error"""
    # Arrange
    config = {...}

    # Act
    obj = MyModule(config)

    # Assert
    assert obj is not None
    assert obj.is_ready()

def test_audio_contract():
    """Verify audio I/O respects 16kHz, float32, [-1, 1] range"""
    # Arrange
    test_audio = np.random.randn(16000).astype(np.float32)  # 1 sec @ 16kHz

    # Act
    result = module.process(test_audio)

    # Assert
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert np.all(result >= -1.0) and np.all(result <= 1.0)
```

Place tests in `test/` directory with naming convention `test_*.py`.

## File Organization & Naming Conventions

```
audio-classification/
├── src/                          # Core modules (no executable scripts here)
│   ├── model.py                  # ASR + speaker embedding models
│   ├── detection/                # KWS module
│   │   ├── model.py             # KeywordSpotterModel class
│   │   ├── decoy_filter.py      # DecoyFilter for homophone blocking
│   │   ├── verifier.py          # KeywordVerifier for ASR-based confirmation
│   │   └── __init__.py
│   └── osd/                      # OSD + separation modules
│       ├── osd.py               # OverlapAnalyzer class
│       └── separation.py         # Separator class
├── scripts/                      # Entry points & CLI utilities
│   ├── osd/
│   │   ├── offline_overlap_3src.py      # Main 3-speaker pipeline
│   │   ├── offline_overlap_mvp.py       # 2-speaker MVP
│   │   ├── overlap3_core.py             # Shared pipeline logic (NOT an entry point)
│   │   └── evaluate_with_sources.py     # Evaluation & metrics
│   ├── detection/
│   │   ├── benchmark_kws.py             # FRR/FA/RTF benchmarking (MAIN EVAL)
│   │   ├── ablation_experiment.py       # Comparative study: KWS vs KWS+verifier vs KWS+decoy
│   │   ├── test_nihao_zhenzhen.py       # Single-file KWS test
│   │   ├── demo_wakeword.py             # Interactive demo with parameter presets
│   │   ├── demo_decoy_filter.py         # Decoy filter interception demo
│   │   ├── param_optimization.py        # Parameter grid search (threshold × boost)
│   │   ├── param_optimization_with_decoy.py  # Joint optimization including filter
│   │   ├── generate_keywords.py         # Keyword format conversion (Chinese)
│   │   ├── generate_keywords_zh_en.py   # Mixed zh-en keyword generation
│   │   ├── data_generator.py            # Synthetic dataset creation
│   │   └── analyze_misclassified.py     # Debug false positives/negatives
│   └── install.sh                       # One-time dependency setup
├── test/                         # Unit & integration tests
│   ├── test_*.py                # Test naming: test_{module}_{scenario}.py
│   └── detection/               # KWS test data & metadata
│       ├── decoy_keywords.txt  # Decoy word definitions
│       └── metadata.json       # Test dataset metadata
├── cache/                        # Auto-generated embedding cache
│   └── *.npy                    # Naming: {audio_stem}.npy
└── results/                      # Output results (auto-created per run)
    └── {timestamp}/             # Results dir: YYYY-MM-DD_HH-MM-SS/
        ├── results.jsonl        # Per-file segment results
        ├── results.csv          # CSV export of results.jsonl
        ├── metrics.json         # Aggregated metrics (SI-SDR, match rate, RTF)
        └── summary.json         # High-level summary stats
```

**Naming Conventions**:

- **Entry-point scripts** (in `scripts/`): `{feature}_{phase}.py` (e.g., `offline_overlap_3src.py`)
- **Core modules** (in `src/`): lowercase, no dashes, single domain per file
- **Test files**: `test_{module}_{scenario}.py` (e.g., `test_osd_overlap_detection.py`)
- **Cache files**: `{audio_stem}.npy` (preserve original filename stem)
- **Output dirs**: Timestamped `YYYY-MM-DD_HH-MM-SS/` format

## Extending the Pipeline: Integration Patterns

### Adding a New Speech Processing Stage

If you need to insert a new processing stage (e.g., diarization, speaker counting):

1. **Implement module in `src/`**:

   ```python
   # src/new_stage.py
   class NewStageModel:
       def __init__(self, model_path=None):
           self._model = None
           self.model_path = model_path

       @property
       def model(self):
           if self._model is None:
               self._model = self._load_model()  # Lazy load
           return self._model

       def process(self, audio_np: np.ndarray) -> Dict[str, Any]:
           """Input: 16kHz float32 numpy array
           Output: dict with results (preserve segment contract downstream)
           """
           pass
   ```

2. **Update pipeline in `scripts/osd/overlap3_core.py`**:

   ```python
   from src.new_stage import NewStageModel

   class Overlap3Pipeline:
       def __init__(self, ...):
           self.new_stage = NewStageModel(...)

       def run(self, audio, segments):
           # Insert after OSD, before separation
           new_stage_result = self.new_stage.process(audio)
           # Continue with existing pipeline...
   ```

3. **Test integration**:
   ```bash
   python offline_overlap_3src.py --max-files 1 --seed 42
   ```

### Swapping Model Backends

**To replace an existing model** (e.g., ASR, OSD, Separator):

1. Ensure new backend has compatible I/O:

   - Input: numpy array (16kHz, float32)
   - Output: must match existing interface (e.g., ASR returns `.text` attribute)

2. Update factory function in relevant module:

   ```python
   # src/model.py
   def create_asr_model(config):
       if config.get("use_new_backend"):
           return NewASRBackend(...)  # New code
       else:
           return SenseVoiceModel(...)  # Fallback
   ```

3. Add CLI argument to entry point:

   ```python
   # scripts/osd/offline_overlap_3src.py
   p.add_argument("--use-new-asr-backend", action="store_true")
   ```

4. Test with baseline data:
   ```bash
   python offline_overlap_3src.py --max-files 10 --use-new-asr-backend
   # Compare metrics.json with previous run
   ```

### Adding New Metrics or Output Fields

1. **Define metric computation** in `overlap3_core.py`:

   ```python
   def compute_custom_metric(results_list):
       """Aggregate metric across all processed files"""
       return sum(...) / len(results_list)
   ```

2. **Add to output structure**:

   ```python
   summary_json = {
       "total_files": ...,
       "custom_metric": compute_custom_metric(all_results),  # New field
   }
   json.dump(summary_json, open("summary.json", "w"))
   ```

3. **Update evaluation script** (`evaluate_with_sources.py`) if metric depends on ground truth.

## Integration Checklist Before Committing

When adding new features or modifying core logic:

- [ ] **Contracts preserved**: Segment tuple `(start, end, is_overlap)` unchanged
- [ ] **Audio contract**: All I/O respects 16kHz, float32, [-1, 1] range
- [ ] **Lazy loading**: Models not loaded in `__init__`, only on first use
- [ ] **Error handling**: Raises `RuntimeError` with clear message if dependencies missing
- [ ] **Caching**: Speaker embeddings (if any) stored in `cache/` with predictable naming
- [ ] **Logging**: Key timings (`time_*`) recorded and exported to `metrics.json`
- [ ] **Tests**: Added `test/test_*.py` with basic smoke tests
- [ ] **Documentation**: Updated relevant docstrings & README
- [ ] **Backward compatibility**: Existing CLI args still work with defaults

## Quick Reference: Common Commands

```bash
# IMPORTANT: Always activate the default environment first
conda activate default

# Full setup
cd scripts && bash install.sh

# Run with LibriMix dataset (recommended for testing)
cd scripts/osd
python offline_overlap_3src.py --librimix-root /path/to/LibriMix --max-files 100 --seed 42

# Run with custom audio files
python offline_overlap_3src.py \
  --input-wavs mix1.wav mix2.wav \
  --target-wav speaker_enrollment.wav \
  --ref-wavs source1.wav source2.wav source3.wav \
  --refs-csv mappings.csv

# Evaluate results
python evaluate_with_sources.py --results-dir ../results/2025-11-26_14-51-01 --librimix-root /path/to/LibriMix

# Batch benchmark
python ../../batch_eval.py result_dir_1 result_dir_2 result_dir_3

# KWS testing
cd scripts/detection
python test_nihao_zhenzhen.py --wav /path/to/audio.wav
python benchmark_kws.py --model-dir ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
  --positive-dir ./samples/positive --negative-dir ./samples/negative
```

## Debugging Tips

**Issue: Module import fails**

```python
# Add to top of script for local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

**Issue: ONNX model not found**

- Check model path: `ls -la models/asr/`
- Verify `--paraformer` or `--sense-voice` flags point to correct `.onnx` files
- Use absolute paths if relative paths don't work

**Issue: GPU memory exceeded**

- Reduce `--max-files` (batch size)
- Use `--provider cpu` to force CPU inference
- Profile with `nvidia-smi` during execution

**Issue: Low match rate on new dataset**

- Start with `--max-files 5` for quick iteration
- Check ground truth format matches expected `.txt` or LibriMix structure
- Use `--seed 42` for reproducibility across runs
- Verify audio sample rate is 16kHz (`ffprobe -show_streams audio.wav`)
