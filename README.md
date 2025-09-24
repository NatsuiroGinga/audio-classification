# 项目说明

本项目实现了一个基于 VAD 和非流式 ASR 的说话人识别系统，提供基线 Benchmark 与“重叠说话”离线 MVP（OSD + 分离 + SID/ASR + CER）。

## 依赖安装与快速运行

### 安装依赖与下载模型/数据

- 进入脚本目录并执行安装脚本：

```bash
cd scripts
bash install.sh
```

- 脚本会安装：
  - sherpa-onnx（ASR 与说话人）、soundfile/sounddevice/numpy
  - asteroid（分离，必需）
  - pyannote.audio（OSD，必需）
  - huggingface_hub（自动下载分离 checkpoint）

### 运行主程序与基线测试

- 主程序（非流式 SID+ASR）：

```bash
bash run.sh
```

- 基线 Benchmark：

```bash
bash test.sh
```

### 运行“重叠说话”离线 MVP（OSD + 分离）

- 进入 OSD 脚本目录并运行：

```bash
cd scripts/osd
bash ./test_overlap.sh
```

输出位于 `../test_overlap`：

- segments.jsonl / segments.csv：分段级别的 SID/ASR 结果
- summary.json：CER 对比（baseline vs. stitched）与用时统计

### Hugging Face Token（仅在需要时）

- 若 pyannote 的 `pyannote/overlapped-speech-detection` 需要认证，请在环境中设置以下任一变量（按优先级读取）：
  - `PYANNOTE_TOKEN` 或 `HF_TOKEN` 或 `HUGGINGFACE_TOKEN`
- 代码会自动将该 token 注入 OverlapAnalyzer（无需手动传参）。

### 分离模型 checkpoint（自动下载，可覆盖）

- 默认行为：若未提供 `--sep-checkpoint`，程序会自动从 Hugging Face 下载一个公开可用的 Conv-TasNet 权重并缓存：
  - Repo：`mpariente/ConvTasNet_WHAM_sepclean`
  - File：`pytorch_model.bin`
- 如需替换默认来源，可通过环境变量覆盖：

```bash
export ASTEROID_SEP_REPO_ID="YourRepo/YourModel"
export ASTEROID_SEP_FILENAME="your_weights.bin"
```

- 如已有本地 checkpoint，可在运行脚本时显式传入：

```bash
python3 ./offline_overlap_mvp.py \
  ...其它参数... \
  --sep-checkpoint /path/to/conv_tasnet_checkpoint.pt
```

## 目录结构

### 脚本

在 scripts 目录下包含以下脚本：

1. install.sh 用于安装依赖、模型、数据集等
2. run.sh 运行主程序（非流式 SID+ASR）
3. test.sh 运行基线 benchmark
4. osd/offline_overlap_mvp.py 离线 OSD + 分离 + SID/ASR + CER 对比
5. osd/test_overlap.sh 一键运行 OSD MVP
6. generate-speaker-text.sh 生成说话人文本
7. split_speakers.py 划分 train/test 数据集
8. version.py 打印环境信息

### 主程序

`speaker-identification-with-vad-non-streaming-asr.py`
