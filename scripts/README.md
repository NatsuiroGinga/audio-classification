# 运行脚本（scripts/）

本目录包含所有的运行脚本，包括基准测试、三源分离、流式处理等。

## 📁 目录结构

```
scripts/
├── install.sh                        # 依赖安装脚本
├── benchmark_pipeline.py            # 说话人识别基准测试
├── batch_eval.py                    # 批量评估脚本
├── compare_methods.py               # 方法对比脚本
├── osd/                             # 三源分离模块
│   ├── offline_overlap_3src.py      # 离线 3-源分离管线
│   ├── overlap3_core.py             # 核心计算逻辑
│   ├── streaming_overlap_3src.py    # 流式 3-源分离
│   ├── streaming_overlap3_core.py   # 流式分离核心逻辑
│   ├── optimized_streaming_overlap3_core.py  # 优化版本
│   ├── vad_streaming_overlap3_core.py        # VAD 版本
│   ├── streaming_asr_pipeline.py    # 流式 ASR 管线
│   ├── evaluate_with_sources.py     # 分离质量评估
│   ├── demo_streaming_asr.py        # 流式 ASR 演示
│   ├── test_overlap_3src.sh         # 测试包装脚本
│   ├── streaming_overlap_3src.sh    # 流式分离包装脚本
│   └── demo_streaming.sh            # 演示包装脚本
├── tools/                           # 工具脚本
│   ├── mix_wavs.py                  # 混合音频
│   └── mix_wavs.sh                  # 混合包装脚本
└── stream/                          # 流式处理相关脚本
```

## 🚀 主要脚本说明

### install.sh - 依赖安装

**用途**：自动安装项目依赖和下载模型。

```bash
# GPU 版本（默认）
bash install.sh

# CPU-only 版本
CPU=1 bash install.sh

# 自定义 CUDA 版本
CUDA=11.8 bash install.sh
```

**功能**：
- ✓ 安装 Python 依赖（pytorch, torchaudio, onnxruntime 等）
- ✓ 下载预训练模型
  - 说话人嵌入（3dspeaker）
  - ASR 模型（SenseVoice）
  - VAD 模型（Silero）
- ✓ 提示准备 LibriMix/Libri3Mix 数据集
- ✓ 验证安装是否成功

### benchmark_pipeline.py - 说话人识别基准

**用途**：评估说话人识别性能（精度、RTF、CER）。

```bash
python benchmark_pipeline.py \
  --speaker-file speaker_list.txt \
  --test-list test_list.txt \
  --model /path/to/speaker_model.onnx \
  --sense-voice /path/to/asr_model.onnx \
  --tokens /path/to/tokens.txt \
  --output-dir test/benchmark \
  --enable-metrics \
  --num-threads 2
```

**参数**：
- `--speaker-file`：说话人列表（每行一个说话人 ID 和对应音频路径）
- `--test-list`：测试列表（说话人 ID，音频路径，参考文本）
- `--model`：说话人嵌入模型路径
- `--sense-voice`：ASR 模型路径
- `--tokens`：ASR 令牌表
- `--enrollment-ratio`：注册数据比例（默认 0.5）
- `--sv-threshold`：说话人验证阈值（默认 0.6）
- `--provider`：推理后端（cuda/cpu）
- `--enable-metrics`：启用 CPU/GPU 监控

**输出**：
- `detail.jsonl`：逐个样例的结果
- `predictions.csv`：CSV 格式结果
- `summary.json`：汇总统计

### osd/offline_overlap_3src.py - 离线三源分离

**用途**：离线处理三源分离，支持文件模式和数据集模式。

#### 文件模式示例

```bash
python offline_overlap_3src.py \
  --input-wavs mix.wav \
  --target-wav target.wav \
  --spk-embed-model ../../models/speaker-recognition/model.onnx \
  --sense-voice ../../models/asr/model.onnx \
  --tokens ../../models/asr/tokens.txt \
  --provider cuda \
  --output-dir test/overlap3
```

#### 数据集模式示例

```bash
python offline_overlap_3src.py \
  --librimix-root /path/to/LibriMix \
  --subset test \
  --num-sources 3 \
  --sep-backend asteroid \
  --osd-backend pyannote \
  --eval-separation \
  --max-files 100
```

**主要参数**：
- `--input-wavs`：输入混合音频（文件模式）
- `--target-wav`：目标说话人音频（文件模式）
- `--librimix-root`：LibriMix 数据集路径（数据集模式）
- `--sep-backend`：分离后端（asteroid/custom）
- `--sep-checkpoint`：自定义分离模型路径
- `--osd-backend`：OSD 后端（pyannote/silero）
- `--spk-embed-model`：说话人嵌入模型（必需）
- `--sense-voice`：ASR 模型（必需）
- `--tokens`：ASR 令牌表（必需）
- `--sv-threshold`：说话人验证阈值（默认 0.6）
- `--provider`：推理后端（cuda/cpu/coreml）
- `--eval-separation`：计算分离质量（SI-SDR）
- `--max-files`：最多处理文件数

**输出**：
```
test/overlap3/<timestamp>/
├── segments.jsonl           # 分段记录
├── segments.csv            # CSV 版本
├── summary.json            # 汇总统计
├── metrics.json            # 性能指标
└── overlap_sep_details.csv # 分离细节（可选）
```

### osd/overlap3_core.py - 三源分离核心

**用途**：核心计算逻辑（排除 I/O）。

**主要类**：
- `Overlap3Pipeline`：三源分离管线
  - 调用 OSD 检测重叠
  - 调用分离器分离源
  - 说话人验证
  - ASR 转录

```python
from scripts.osd.overlap3_core import Overlap3Pipeline
import torchaudio

pipeline = Overlap3Pipeline(
    osd_model='pyannote',
    sep_model='asteroid',
    spk_embed_model='/path/to/speaker_model.onnx',
    asr_model='/path/to/asr_model.onnx',
    tokens='/path/to/tokens.txt'
)

# 运行管线
mixture, sr = torchaudio.load('mix.wav')
target, _ = torchaudio.load('target.wav')

result = pipeline.run(
    mixture=mixture,
    sr=sr,
    target_embedding=target,
    sv_threshold=0.6
)

print(result.text)       # ASR 文本
print(result.sv_score)   # 说话人验证得分
print(result.rtf)        # 实时因子
```

### osd/streaming_overlap_3src.py - 流式三源分离

**用途**：实时处理流式音频的三源分离。

```bash
python streaming_overlap_3src.py \
  --input-wavs stream.wav \
  --target-wav target.wav \
  --chunk-size 1600 \
  --stride 800 \
  --buffer-size 4800 \
  --provider cuda
```

**关键参数**：
- `--chunk-size`：处理块大小（样本数）
- `--stride`：滑动步长
- `--buffer-size`：缓冲区大小
- `--use-vad`：是否使用 VAD 分段
- `--vad-threshold`：VAD 判断阈值

### osd/streaming_asr_pipeline.py - 流式 ASR

**用途**：实时语音转文本，支持多语言。

```bash
python streaming_asr_pipeline.py \
  --model /path/to/model.onnx \
  --tokens /path/to/tokens.txt \
  --input audio.wav \
  --chunk-duration 0.32 \
  --partial-interval 0.1 \
  --provider cuda
```

**特点**：
- ✓ 输出 Partial 结果（中间结果，可能改变）
- ✓ 输出 Final 结果（最终结果，不再改变）
- ✓ 仅使用 VAD 分段（无强制最大时长）
- ✓ 优化的滑动窗口，避免 O(n²) 复杂度

### tools/mix_wavs.py - 音频混合

**用途**：混合多个音频文件。

```bash
python mix_wavs.py \
  --sources src1.wav src2.wav src3.wav \
  --output mixture.wav \
  --snr 10 \
  --randomize
```

### batch_eval.py - 批量评估

**用途**：评估多次运行的结果。

```bash
python batch_eval.py \
  --results-dir test/overlap3 \
  --output batch_analysis.json \
  --format jsonl
```

**输出**：聚合统计，包括：
- 平均 RTF、CER、SI-SDR
- 检测率、识别率
- 计时分布

## 📋 常见工作流

### 1. 快速测试（文件模式）

```bash
cd scripts/osd
bash test_overlap_3src.sh
```

### 2. 数据集评估（Libri3Mix）

```bash
cd scripts/osd
python offline_overlap_3src.py \
  --librimix-root /data/LibriMix \
  --subset test \
  --max-files 100
```

### 3. 批量处理

```bash
python batch_eval.py \
  --results-dir test/overlap3 \
  --output results_summary.json
```

### 4. 性能基准

```bash
python benchmark_pipeline.py \
  --speaker-file speakers.txt \
  --test-list test.txt \
  --enable-metrics
```

### 5. 流式演示

```bash
cd scripts/osd
bash demo_streaming_asr.sh s1 0.3 0.5
# 参数: <sample> <chunk_duration> <partial_interval>
```

## 🔧 性能优化建议

### 计算加速
```bash
# 使用 GPU
--provider cuda

# 增加线程数
--num-threads 4

# 启用分离优化
--optimized-sep
```

### 内存优化
```bash
# 减小分离 batch
--sep-batch-size 1

# 启用梯度检查点（如果支持）
--checkpointing
```

### 输出优化
```bash
# 跳过不必要的输出
--skip-waveforms
--skip-details

# 压缩输出
--compress
```

## 📊 输出解释

### segments.jsonl 字段

```json
{
  "wav": "test_s1.wav",
  "start": 0.0,
  "end": 5.2,
  "kind": "clean",
  "stream": 0,
  "text": "speech recognition test",
  "sv_score": 0.92,
  "asr_time": 0.35,
  "target_src": 0,
  "target_src_text": "speech recognition test"
}
```

### metrics.json 字段

```json
{
  "total_duration": 120.5,
  "num_segments": 250,
  "time_osd_sec": 15.2,
  "time_sep_sec": 45.8,
  "time_asr_sec": 28.5,
  "time_compute_total_sec": 89.5,
  "rtf": 0.74,
  "detection_rate": 0.35,
  "avg_si_sdr": 8.5
}
```

---

**更新**：2026-01-09  
**作者**：NatsuiroGinga
