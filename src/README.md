# 源代码模块（src/）

本目录包含项目的核心源代码模块，实现了 ASR、说话人嵌入、重叠检测、语音分离等功能。

## 📁 目录结构

```
src/
├── model.py                  # ASR 工厂 + 说话人嵌入管理
├── osd/                      # 重叠检测与语音分离模块
│   ├── __init__.py
│   ├── osd.py               # OSD（重叠检测）包装器
│   ├── separation.py        # 语音分离（Conv-TasNet）
│   ├── dataset.py           # 数据集处理（Libri3Mix 等）
│   └── streaming_asr.py     # 流式 ASR 实现
└── detection/               # 其他检测模块（如说话人检测）
```

## 🔧 核心模块说明

### model.py - ASR 工厂与说话人嵌入

**主要类：**

#### `ASRModel`

自动选择 ASR 后端的工厂类。

```python
from src.model import create_asr_model

# 创建 ASR 模型
asr = create_asr_model(
    model_type='sense-voice',  # 或 'paraformer', 'transducer'
    model_path='/path/to/model.onnx',
    tokens_path='/path/to/tokens.txt',
    provider='cuda'  # 'cuda', 'cpu', 'coreml'
)

# 推理
result = asr.recognize('audio.wav')
print(result)  # {'text': '转录文本', 'confidence': 0.95}
```

**支持的 ASR 后端：**

- **Paraformer**：单一模型，推理速度快，适合实时场景
- **SenseVoice**：多语言支持，通过 Sherpa-ONNX 运行
- **Transducer**：Encoder-Decoder-Joiner 框架，支持流式识别

#### `SpeakerEmbeddingManager`

说话人嵌入提取与验证。

```python
from src.model import SpeakerEmbeddingManager

manager = SpeakerEmbeddingManager(
    model_path='/path/to/speaker_model.onnx',
    provider='cuda'
)

# 提取嵌入
embed = manager.extract_embedding('audio.wav')
print(embed.shape)  # (192,) - L2 正规化

# 计算相似度
similarity = manager.compute_similarity(embed1, embed2)
print(similarity)  # 0.85 - 余弦相似度

# 验证说话人
is_match = manager.verify(embed, reference_embed, threshold=0.6)
print(is_match)  # True/False
```

### osd/ - 重叠检测与分离模块

#### osd.py - OverlapAnalyzer

使用 pyannote.audio 进行重叠检测。

```python
from src.osd.osd import OverlapAnalyzer

analyzer = OverlapAnalyzer(
    use_auth_token='your_huggingface_token'
)

# 检测重叠
diarization = analyzer.diarize('audio.wav')
# diarization: Annotation 对象，包含扬声器时间戳和标签

# 提取重叠区间
overlaps = analyzer.get_overlaps(diarization)
for start, end in overlaps:
    print(f"重叠: {start:.2f}s - {end:.2f}s")
```

#### separation.py - Separator

语音分离模块，支持 Conv-TasNet。

```python
from src.osd.separation import Separator
import torchaudio

separator = Separator(
    model_name='conv_tasnet',  # 或自定义 checkpoint
    n_sources=3,               # 分离源数
    sample_rate=16000,
    device='cuda'
)

# 分离音频
waveform, sr = torchaudio.load('mixed.wav')
separated = separator.separate(waveform)
print(separated.shape)  # (3, 1, samples) - 三个分离的源

# 保存分离结果
for i, src in enumerate(separated):
    torchaudio.save(f'source_{i}.wav', src, sr)

# 评估分离质量（如果有参考）
si_sdr = separator.compute_si_sdr(separated, references)
print(f"SI-SDR: {si_sdr:.2f} dB")
```

#### dataset.py - Libri3Mix 数据集处理

处理 Libri3Mix/LibriMix 数据集。

```python
from src.osd.dataset import Libri3MixDataset

dataset = Libri3MixDataset(
    root='/path/to/LibriMix',
    sample_rate=16000,
    num_sources=3,
    max_samples=100
)

for sample in dataset:
    mixture = sample['mixture']      # 混合音频
    sources = sample['sources']      # [src1, src2, src3]
    speakers = sample['speakers']    # 说话人ID列表

    print(f"混合时长: {mixture.shape[0]/16000:.2f}s")
```

#### streaming_asr.py - StreamingASR

流式 ASR 实现，支持中间结果和最终结果。

```python
from src.osd.streaming_asr import StreamingASR, VADStreamingASR

# 创建流式 ASR
asr = StreamingASR(
    model_path='/path/to/model.onnx',
    tokens_path='/path/to/tokens.txt',
    chunk_size=640,  # 样本数
    stride=320,      # 滑动步长
    provider='cuda'
)

# 模拟音频流
import torchaudio
waveform, sr = torchaudio.load('audio.wav')

# 处理流
for i in range(0, waveform.shape[1], 320):
    chunk = waveform[:, i:i+640]

    # 获取中间结果（Partial）
    partial_result = asr.process_chunk(chunk, is_final=False)
    if partial_result:
        print(f"中间: {partial_result['text']}")

    # 语音结束时获取最终结果
    if i == waveform.shape[1] - 1:
        final_result = asr.process_chunk(chunk, is_final=True)
        print(f"最终: {final_result['text']}")

# VAD 版本（自动分段）
vad_asr = VADStreamingASR(
    model_path='/path/to/model.onnx',
    tokens_path='/path/to/tokens.txt',
    vad_model='silero',  # VAD 模型
    provider='cuda'
)

# 处理，自动按 VAD 分段
results = vad_asr.process_stream(waveform)
for result in results:
    print(f"{result['start']:.2f}s - {result['end']:.2f}s: {result['text']}")
```

### detection/ - 检测模块

其他检测功能的实现（如关键词检测、说话人检测等）。

## 📊 数据流与集成

### 典型工作流

```
输入音频
    ↓
[VAD] → 检测语音片段
    ↓
[OSD] → 检测重叠区间
    ↓
[分离] → 分离源音频（重叠部分）
    ↓
[说话人验证] → 筛选目标说话人
    ↓
[ASR] → 转录文本
    ↓
输出结果（JSON/CSV）
```

### 关键参数配置

**ASR 模型：**

```python
# 支持模型列表
{
    'sense-voice': '/models/asr/sense-voice.onnx',
    'paraformer': '/models/asr/paraformer.onnx',
    'transducer': '/models/asr/transducer.onnx'
}
```

**推理后端：**

```python
provider_options = {
    'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
    'cpu': ['CPUExecutionProvider'],
    'coreml': ['CoreMLExecutionProvider', 'CPUExecutionProvider']
}
```

**说话人验证阈值：**

```python
sv_threshold = 0.6  # 余弦相似度，推荐 0.5~0.7
```

## 🧪 使用示例

### 完整的三源分离流程

```python
from src.model import create_asr_model, SpeakerEmbeddingManager
from src.osd.osd import OverlapAnalyzer
from src.osd.separation import Separator
import torchaudio

# 初始化各模块
osd = OverlapAnalyzer()
separator = Separator(n_sources=3, device='cuda')
asr = create_asr_model('sense-voice', provider='cuda')
spk_manager = SpeakerEmbeddingManager(provider='cuda')

# 加载音频
mixture, sr = torchaudio.load('mixed.wav')
target_audio, _ = torchaudio.load('target_spk.wav')

# 检测重叠
diarization = osd.diarize(str(mixture))
overlaps = osd.get_overlaps(diarization)

# 处理重叠片段
for start, end in overlaps:
    # 分离
    start_idx = int(start * sr)
    end_idx = int(end * sr)
    overlap_seg = mixture[:, start_idx:end_idx]

    separated = separator.separate(overlap_seg)

    # 目标筛选
    target_embed = spk_manager.extract_embedding(target_audio)

    best_src = None
    best_score = 0
    for i, src in enumerate(separated):
        src_embed = spk_manager.extract_embedding(src)
        score = spk_manager.compute_similarity(target_embed, src_embed)
        if score > best_score:
            best_score = score
            best_src = i

    # ASR
    if best_src is not None and best_score > 0.6:
        result = asr.recognize(separated[best_src])
        print(f"分离源 {best_src}: {result['text']} (SV: {best_score:.2f})")
```

## 📚 相关资源

- [ASRModel 详细文档](../scripts/osd/overlap3_core.py)
- [Sherpa-ONNX 文档](https://github.com/k2-fsa/sherpa-onnx)
- [pyannote.audio 文档](https://github.com/pyannote/pyannote-audio)
- [Asteroid 文档](https://github.com/asteroid-team/asteroid)

## 🔗 模块依赖

```
model.py
├── onnxruntime     # ONNX 推理
├── sherpa-onnx     # ASR + 说话人嵌入
└── numpy/torch     # 数值计算

osd/
├── pyannote.audio  # 重叠检测
├── asteroid        # 语音分离
├── torchaudio      # 音频处理
└── julius          # 音频过滤
```

## ⚙️ 性能提示

1. **模型加载优化**

   - 预加载模型并重用，避免重复加载
   - 使用 GPU 推理加速（provider='cuda'）

2. **批处理**

   - 流式 ASR 支持批处理多个音频块
   - 分离模块可批处理多个混合

3. **内存管理**
   - 大音频文件分块处理
   - 及时释放不需要的张量

---

**更新**：2026-01-09  
**作者**：NatsuiroGinga
