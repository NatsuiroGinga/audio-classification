# 项目说明

本项目包含：

1. 基于 VAD + 非流式 ASR 的说话人识别主流程（含 Benchmark）。
2. “重叠说话”离线 MVP：当前版本已精简为 **OSD（重叠检测）+ 语音分离 + ASR（无说话人识别 / 无 CER）**，面向 **Libri2Mix 8k test** 混合语音集的快速实验。
3. 三说话人（3-src）语音分离快速实验仓库：基于 OSD + 三路分离 + 目标说话人筛选 + ASR 的离线管线，面向 Libri3Mix / 自建三说话人混合的快速实验。

## 依赖安装与快速运行

### 安装依赖与下载模型/数据

- 进入脚本目录并执行安装脚本：

```bash
cd scripts
bash install.sh
```

- 脚本会：
  - 安装 Python 依赖（默认 GPU/CUDA 版本，来源于 requirements.txt；如需 CPU 请在执行前设置 `CPU=1`）
  - 下载默认模型：
    - 说话人嵌入（3dspeaker 16k ONNX）到 `models/speaker-recognition/`
    - SenseVoice（Sherpa-ONNX，多语 ASR）到 `models/asr/`
  - 提示你准备 LibriMix/Libri3Mix 数据集并设置 `LIBRIMIX_ROOT`（因 LibriMix 数据集过大，所以不自动下载）

可选：CPU-only 环境

# 三说话人（3-src）语音分离快速实验仓库

本仓库聚焦于离线“三说话人”语音分离的研究与工程化验证：用 OSD（重叠检测）挑出重叠片段，
对重叠片段做三路（n_src=3）分离，然后对分离后的支路做目标说话人筛选与 ASR。文档以 3-src 为中心，
并保留必要的运行示例与评估说明，适合在 Libri3Mix / 自建三说话人混合上做快速实验。

核心思想：

- 使用 OSD（pyannote）识别重叠区间；
- 对重叠片段运行三路分离（Asteroid Conv-TasNet 或自定义 checkpoint）；
- 使用说话人嵌入（Sherpa-ONNX）对分离支路进行目标筛选，只有通过 SV 筛选的支路才进行 ASR；
- 输出每个片段（clean / overlap）对应的 ASR 文本、SV 得分与元信息，便于下游分析与评估。

## 快速开始（三说话人模式）

### 安装依赖（推荐在脚本目录下执行）

```bash
cd scripts
bash install.sh
```

可选（CPU-only）：

```bash
CPU=1 bash install.sh
```

安装脚本会尝试下载常用模型（说话人嵌入、Sherpa-ONNX ASR 等），并提示准备 LibriMix/Libri3Mix 数据集。

### 一键运行（文件模式示例）

脚本 `scripts/osd/test_overlap_3src.sh` 提供了文件/数据两种运行模式的包装：

示例（文件模式，单混合 + 指定 target）：

```bash
cd scripts/osd
bash ./test_overlap_3src.sh
```

包装脚本会将默认的 `INPUT_WAVS` 与 `TARGET_WAV` 转为 `--input-wavs` / `--target-wav` 并调用 `offline_overlap_3src.py`。

示例（直接运行 Python，Libri3Mix 数据集模式）：

```bash
python3 ./offline_overlap_3src.py \
  --librimix-root /abs/path/to/LibriMix \
  --spk-embed-model ../../models/speaker-recognition/your_spk.onnx \
  --sense-voice ../../models/asr/your_asr.onnx \
  --tokens ../../models/asr/tokens.txt \
  --provider cuda --num-threads 2 \
  --osd-backend pyannote --sep-backend asteroid \
  --max-files 100 --min-overlap-dur 0.2 \
  --seed 123
```

关键参数：

- `--sep-backend`：分离后端（默认 asteroid）；
- `--sep-checkpoint`：自定义分离模型 checkpoint；
- `--spk-embed-model`：说话人嵌入 ONNX 模型路径（必需）；
- `--sv-threshold`：目标筛选的余弦阈值，默认 0.6；
- `--input-wavs` / `--target-wav`：文件模式下直接提供混合与目标音频（便于调试单样例）。

## 输出与评估指标（面向 3-src）

运行结束后会生成时间戳目录（默认 `test/overlap3/<timestamp>/`），主要文件：

- `segments.jsonl`：每行一个分段记录（clean / overlap），包含字段：wav、start、end、kind、stream、text、asr_time、sv_score、target_src、target_src_text；
- `segments.csv`：同信息的 CSV 备份，便于快速查看；
- `overlap_sep_details.csv`（可选）：当开启 `--eval-separation` 且 `--save-sep-details` 时，保存每个重叠片段的 SI-SDR / SI-SDRi（PIT 后）与选路信息；
- `metrics.json`（可选）：聚合统计（总时长、分段数、time_osd_sec、time_sep_sec、time_asr_sec、time_compute_total_sec、rtf 等），注意计时仅覆盖“核心计算”部分（不含写盘 I/O）；
- `summary.json`：总体摘要（包含 target hit/miss 统计）。

分离质量评估：支持在文件模式下提供参考源（refs CSV / ref_wavs）进行 SI-SDR/PIT 评估，支持 K=2 或 3 的参考情况。

## 数据集（建议 / 常用）

- Libri3Mix / LibriMix：用于多说话人分离实验（推荐准备 Libri3Mix 以做原生 3-src 评估）；
- LibriSpeech：作为源语音语料；
- THCHS30（清华大学中文语音数据集）：http://www.openslr.org/resources/18/data_thchs30.tgz，可用于中文 ASR 训练或快速中文 domain 适配测试。

## 关于计时与对齐（重要提示）

- 核心计算已封装在 `scripts/osd/overlap3_core.py::Overlap3Pipeline`，`offline_overlap_3src.py` 仅负责写文件并将 `Overlap3Pipeline` 的输出落盘；因此 `metrics.json` 中的 `time_compute_total_sec` 与各阶段时间不包含写文件耗时。
- 若你在输出中发现 `text` 与 `target_src_text` 不一致，最常见原因：
  1. 混音时对源做了偏移（offset），导致按混音时间直接从 `target_wav` 切片会错位；
  2. 分离支路残留干扰或分支选择错误（SV 分数偏低）；
  3. OSD 切分在词边界处分割导致 ASR 差异。

建议：在 refs CSV 中记录 per-source offset 或启用/实现自动对齐（GCC-PHAT 互相关的全局或局部对齐），并根据需要调整 `--sv-threshold` 以减少误选。

## 小贴士与扩展

- 若你的工作聚焦中文语音分离，建议：
  - 使用 THCHS30 对 ASR 做微调或选择更适合中文的 ASR 模型；
  - 在文件模式下提供 refs（`--ref-wavs` 或 `--refs-csv`），并在必要时在 CSV 中扩展 `offset` 字段以确保对齐；
- 若需要将 I/O 时间也计入总体延迟统计，可在调用脚本外部测量并合并到自定义报告中。

---

更多细节与高级用法请查看 `scripts/osd` 下的各脚本（`test_overlap_3src.sh`, `offline_overlap_3src.py`, `overlap3_core.py`）
