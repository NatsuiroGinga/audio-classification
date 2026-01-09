# 三源分离模块（scripts/osd/）

本目录包含三源（3-Source）语音分离的核心实现，包括离线和流式处理方案。

## 📁 核心脚本

### offline_overlap_3src.py - 离线处理管线

**功能**：检测重叠区间、分离语音、验证说话人、进行 ASR 转录。

#### 文件模式快速开始

```bash
python offline_overlap_3src.py \
  --input-wavs /path/to/mix.wav \
  --target-wav /path/to/target.wav \
  --spk-embed-model ../../models/speaker-recognition/model.onnx \
  --sense-voice ../../models/asr/model.onnx \
  --tokens ../../models/asr/tokens.txt \
  --provider cuda \
  --output-dir test/overlap3
```

#### 数据集模式（Libri3Mix）

```bash
python offline_overlap_3src.py \
  --librimix-root /data/LibriMix \
  --subset test \
  --num-sources 3 \
  --sep-backend asteroid \
  --osd-backend pyannote \
  --max-files 1000 \
  --eval-separation
```

**关键参数**：

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `--input-wavs` | 输入混合音频（文件模式）| None |
| `--target-wav` | 目标说话人音频 | None |
| `--librimix-root` | Libri3Mix 根路径（数据集模式）| None |
| `--subset` | 数据集子集 (train/test/dev) | test |
| `--num-sources` | 源数 | 3 |
| `--sep-backend` | 分离后端 (asteroid/custom) | asteroid |
| `--sep-checkpoint` | 自定义分离模型 | None |
| `--osd-backend` | OSD 后端 (pyannote) | pyannote |
| `--spk-embed-model` | 说话人嵌入模型 | ✓ 必需 |
| `--sense-voice` | ASR 模型 | ✓ 必需 |
| `--tokens` | ASR 令牌表 | ✓ 必需 |
| `--sv-threshold` | 说话人验证阈值 | 0.6 |
| `--min-overlap-dur` | 最小重叠时长 (秒) | 0.2 |
| `--eval-separation` | 计算 SI-SDR | False |
| `--provider` | 推理后端 | cuda |
| `--output-dir` | 输出目录 | test/overlap3 |
| `--max-files` | 最多处理文件数 | -1 (全部) |

**输出文件**：

```
test/overlap3/<timestamp>/
├── segments.jsonl          # 每行一个分段
├── segments.csv            # CSV 版本
├── summary.json            # 汇总统计
├── metrics.json            # 性能指标（如果 --eval-separation）
└── overlap_sep_details.csv # 分离细节（如果启用）
```

### overlap3_core.py - 核心计算引擎

**作用**：实现 OSD → 分离 → 说话人验证 → ASR 的完整流程。

**主要类**：`Overlap3Pipeline`

```python
from scripts.osd.overlap3_core import Overlap3Pipeline

pipeline = Overlap3Pipeline(
    osd_backend='pyannote',
    sep_backend='asteroid',
    spk_embed_model='/path/to/model.onnx',
    asr_model='/path/to/asr.onnx',
    tokens='/path/to/tokens.txt',
    device='cuda'
)

# 运行完整流程
result = pipeline.run(
    mixture,           # 混合音频张量
    sr=16000,
    target_embedding,  # 目标说话人嵌入
    sv_threshold=0.6
)

print(result.text)            # ASR 转录文本
print(result.sv_score)        # 说话人验证得分
print(result.compute_time)    # 计算耗时
```

**数据流**：
```
混合音频 → OSD检测 → 提取重叠区间 → 分离 → 说话人验证 → ASR → 结果
```

### streaming_overlap_3src.py - 流式处理

**功能**：实时处理音频流的三源分离。

```bash
python streaming_overlap_3src.py \
  --input-wavs stream.wav \
  --target-wav target.wav \
  --chunk-size 1600 \
  --stride 800 \
  --use-vad \
  --provider cuda
```

**流式处理特点**：
- 块处理（chunk-based）：避免一次性加载整个音频
- 缓冲区管理：维持滑动窗口
- Partial/Final 结果：中间和最终输出

### streaming_asr_pipeline.py - 流式 ASR

**功能**：实时语音识别，支持中间结果。

```bash
python streaming_asr_pipeline.py \
  --model /path/to/model.onnx \
  --tokens /path/to/tokens.txt \
  --input audio.wav \
  --chunk-duration 0.32 \
  --partial-interval 0.1
```

**输出示例**：
```
[Partial] 0.32s: 你好
[Partial] 0.64s: 你好世界
[Final]   0.95s: 你好世界
```

### vad_streaming_overlap3_core.py - VAD 版本

**功能**：使用 VAD 进行自然分段的流式处理。

优势：
- 自动检测说话边界
- 无需手动设置分段时长
- 更符合自然语音节奏

```bash
python -c "from scripts.osd.vad_streaming_overlap3_core import VADStreamingOverlap3Pipeline"
```

### optimized_streaming_overlap3_core.py - 优化版本

**功能**：直接分离（跳过 OSD），加速流式处理。

适用场景：
- 高实时性要求
- 已知音频中大部分是重叠部分

## 🔄 工作流对比

### 离线 vs 流式

| 特点 | 离线 | 流式 |
|------|------|------|
| 延迟 | 高（需等待全部音频） | 低（边处理边输出） |
| 内存 | 需加载全部音频 | 块处理，内存小 |
| 精度 | 可能更高（上下文充足） | 略低（上下文限制） |
| 应用 | 批量处理、评估 | 实时应用 |

### OSD vs 直接分离

| 方法 | OSD 分离 | 直接分离 |
|-----|---------|---------|
| 精度 | 更高（只分离重叠部分）| 全局分离 |
| 速度 | 慢（多一步 OSD）| 快（直接分离） |
| 适用 | 重叠片段较少 | 全是重叠 |

## 📊 评估指标

### 分离质量：SI-SDR

```
SI-SDR (dB) = 10 log10(s_target^2 / e_noise^2)

典型范围：5~15 dB
- < 5 dB：效果差
- 5-10 dB：一般
- 10-15 dB：较好
- > 15 dB：优秀
```

### 说话人验证：余弦相似度

```
similarity = embed1 · embed2 / (||embed1|| * ||embed2||)

范围：[-1, 1]
- > 0.6：很可能是同一说话人
- 0.5-0.6：边界
- < 0.5：不同说话人
```

### 实时因子：RTF

```
RTF = 计算时间 / 音频时长

RTF < 1.0：可实时处理
RTF > 1.0：超实时（需优化或更强硬件）
```

## 🎯 使用建议

### 1. 快速原型测试

```bash
# 使用文件模式，测试单个样例
bash test_overlap_3src.sh
```

### 2. 完整数据集评估

```bash
# 使用 Libri3Mix，启用分离评估
python offline_overlap_3src.py \
  --librimix-root /data/LibriMix \
  --eval-separation \
  --max-files 100
```

### 3. 实时应用

```bash
# 使用流式 ASR + VAD
python streaming_asr_pipeline.py \
  --use-vad \
  --partial-results
```

### 4. 性能基准

```bash
# 启用详细监控
python offline_overlap_3src.py \
  --input-wavs test.wav \
  --enable-metrics \
  --log-file bench.log
```

## ⚙️ 性能优化

### 加速策略

```bash
# 1. 使用 GPU
--provider cuda

# 2. 增加线程数
--num-threads 4

# 3. 跳过不必要的计算
--skip-ref-eval          # 跳过参考评估
--skip-waveforms         # 跳过保存波形
--skip-separation-eval   # 跳过 SI-SDR 计算

# 4. 使用优化版本
python optimized_streaming_overlap3_core.py
```

### 内存优化

```bash
# 1. 减小分离 batch
--sep-batch-size 1

# 2. 流式处理
python streaming_overlap_3src.py

# 3. 清理缓存
--clear-cache-every 100
```

## 🐛 常见问题

**Q: OSD 检测不准确**
- A: 调整最小重叠时长 `--min-overlap-dur`
- A: 检查音频质量和增益

**Q: 分离效果差**
- A: 使用自定义分离模型 `--sep-checkpoint`
- A: 检查音频SNR（信噪比）

**Q: 说话人验证失败**
- A: 调整阈值 `--sv-threshold` (0.5~0.7)
- A: 检查目标说话人音频质量

**Q: 处理速度慢**
- A: 使用 GPU (`--provider cuda`)
- A: 使用优化版本 (`optimized_streaming_overlap3_core.py`)
- A: 减小音频采样率

## 📈 性能指标参考

基于 V100 GPU，16k 采样率，单说话人目标：

| 阶段 | 耗时 | 占比 |
|-----|------|------|
| OSD | 12% | 少量音频检查 |
| 分离 | 60% | 主要计算量 |
| ASR | 25% | 转录 |
| 说话人验证 | 3% | 嵌入提取 |
| **总计** | **100%** | **RTF ≈ 0.5** |

## 🔗 相关模块

- [overlap3_core.py](overlap3_core.py) - 核心计算
- [../model.py](../src/model.py) - ASR 和说话人嵌入
- [../../src/osd/osd.py](../../src/osd/osd.py) - OSD 实现
- [../../src/osd/separation.py](../../src/osd/separation.py) - 分离实现

---

**更新**：2026-01-09  
**作者**：NatsuiroGinga
