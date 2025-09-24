# 项目说明

本项目包含：

1. 基于 VAD + 非流式 ASR 的说话人识别主流程（含 Benchmark）。
2. “重叠说话”离线 MVP：当前版本已精简为 **OSD（重叠检测）+ 语音分离 + ASR（无说话人识别 / 无 CER）**，面向 **Libri2Mix 8k test** 混合语音集的快速实验。

> 说明：早期版本包含 SID（说话人识别）与 CER（参考文本对齐）比较，已在最新接口中移除；历史脚本中 `--speaker-file`, `--test-list` 等参数已废弃。若你查看旧日志或提交记录看到相关字段，属正常演进差异。

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

### 运行“重叠说话”离线 MVP（OSD + 分离 + ASR, 无 SID / 无 CER）

进入 OSD 脚本目录并运行：

```bash
cd scripts/osd
bash ./test_overlap.sh
```

环境变量可选：

```bash
export MAX_FILES=50         # 限制处理多少个 Libri2Mix 混合（0 或未设表示全部）
export HF_TOKEN=xxxxxxxx    # 若 pyannote 需要鉴权
```

脚本会：

1. 加载 Libri2Mix 8k test split（需已下载或由 dataset 代码自动处理）。
2. 对每个混合音频执行重叠检测（pyannote）。
3. 对标记为 overlap 的片段使用 Conv-TasNet（asteroid）做 2 路分离。
4. 对 clean 片段直接 ASR；对 overlap 片段的每个分离分支分别 ASR。
5. 生成时间戳子目录：`test_overlap/<YYYY-MM-DD_HH-MM-SS>/`。

输出文件（均位于上述时间戳目录内）：

- `segments.jsonl`：逐片段 JSON 行；字段见下。
- `segments.csv`：同内容 CSV 形式，便于快速查看 / 表格处理。
- `summary.json`：总体统计（段数、耗时、数据集名、处理的混合数量等）。

segments 记录字段说明：

| 字段        | 说明                                                    |
| ----------- | ------------------------------------------------------- |
| wav         | 原始混合音频的文件路径                                  |
| start / end | 片段起止时间 (秒, 浮点, 3 位小数)                       |
| kind        | `clean`（未判定重叠或时长低于最小重叠门限）或 `overlap` |
| stream      | `null`（clean 片段）或 0/1（分离后的两个分支）          |
| text        | ASR 输出文本（解码方法默认 greedy_search）              |
| asr_time    | 该片段（或分支）ASR 解码耗时（秒，浮点）                |

summary.json 主要字段：

| 字段               | 说明                                        |
| ------------------ | ------------------------------------------- |
| segments           | 输出的总片段（含分离分支）计数              |
| elapsed_wall_sec   | 整个流程的墙钟时间                          |
| dataset            | 数据集标识（Libri2Mix8kTest）               |
| processed_mixtures | 实际处理的混合音频数量（受 MAX_FILES 限制） |
| sample_rate_target | 统一重采样的采样率（默认 16000）            |
| notes              | 额外说明（当前为 ASR only / 无 CER）        |

与旧版本差异：

- 不再输出 speaker_true / speaker_pred / score 等列。
- 不再计算 CER（缺乏对应参考转写或实验聚焦于结构管线）。
- CLI 中 `--speaker-file`, `--test-list`, `--ref-text-list` 已删除。
- CLI 中 `--threshold` 仅为兼容占位，可忽略。

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
4. `osd/offline_overlap_mvp.py` 离线 OSD + 分离 + **ASR（无 SID / 无 CER）** 管线
5. `osd/test_overlap.sh` 一键运行最新 ASR-only Overlap MVP
6. generate-speaker-text.sh 生成说话人文本
7. split_speakers.py 划分 train/test 数据集
8. version.py 打印环境信息

### 主程序

`speaker-identification-with-vad-non-streaming-asr.py`
