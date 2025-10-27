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
3. 对标记为 overlap 的片段使用 Conv-TasNet（asteroid）做 n_src 路分离（默认 2，可设 3）。
4. 对 clean 片段直接 ASR；对 overlap 片段的每个分离分支分别 ASR。
5. 生成时间戳子目录：`test_overlap/<YYYY-MM-DD_HH-MM-SS>/`。

输出文件（均位于上述时间戳目录内）：

- `segments.jsonl`：逐片段 JSON 行；字段见下。
- `segments.csv`：同内容 CSV 形式，便于快速查看 / 表格处理。
- `summary.json`：总体统计（段数、耗时、数据集名、处理的混合数量等）。

segments 记录字段说明：

| 字段        | 说明                                                                                 |
| ----------- | ------------------------------------------------------------------------------------ |
| wav         | 原始混合音频的文件路径                                                               |
| start / end | 片段起止时间 (秒, 浮点, 3 位小数)                                                    |
| kind        | `clean`（未判定重叠或时长低于最小重叠门限）或 `overlap`                              |
| stream      | `null`（clean 片段）或 0/1（两路分离）/ 0/1/2（三路分离）                            |
| text        | ASR 输出文本（解码方法默认 greedy_search）                                           |
| asr_time    | 该片段（或分支）ASR 解码耗时（秒，浮点）                                             |
| sv_score    | 说话人相似度分数（若启用目标说话人筛选时会给出，否则为空）                           |
| matched     | 是否判定为目标说话人（1/0；未启用筛选时恒为 1）                                      |
| match       | 匹配标签（当使用目标说话人筛选时为 `target` 或 `unknown`；三路分离脚本会写入该字段） |

summary.json 主要字段：

| 字段               | 说明                                        |
| ------------------ | ------------------------------------------- |
| segments           | 输出的总片段（含分离分支）计数              |
| elapsed_wall_sec   | 整个流程的墙钟时间                          |
| dataset            | 数据集标识（Libri2Mix8kTest）               |
| processed_mixtures | 实际处理的混合音频数量（受 MAX_FILES 限制） |
| sample_rate_target | 统一重采样的采样率（默认 16000）            |
| notes              | 额外说明（当前为 ASR only / 无 CER）        |

与目标说话人筛选相关的命中统计（在三路分离脚本或启用筛选时提供）：

| 字段                     | 说明                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| segments_seen_clean      | 经过筛选统计的 clean 段片段数                                        |
| segments_seen_overlap    | 经过筛选统计的 overlap 段片段数                                      |
| segments_matched         | 命中目标说话人的片段数（clean + overlap）                            |
| segments_missed          | 未命中目标说话人的片段数（clean + overlap）                          |
| audio_seen_clean_sec     | clean 段累计时长（秒）                                               |
| audio_seen_overlap_sec   | overlap 段累计时长（秒）                                             |
| audio_matched_sec        | 命中目标累计时长（秒）                                               |
| audio_missed_sec         | 未命中目标累计时长（秒）                                             |
| target_hit_rate_segments | 片段命中率 = segments_matched / (segments_matched + segments_missed) |

与旧版本差异：

- 不再输出 speaker_true / speaker_pred / score 等列。
- 不再计算 CER（缺乏对应参考转写或实验聚焦于结构管线）。
- CLI 中 `--speaker-file`, `--test-list`, `--ref-text-list` 已删除。
- CLI 中 `--threshold` 仅为兼容占位，可忽略。

#### 三路分离（LibriMix/Libri3Mix，支持目标说话人筛选，可选）

我们提供了三路分离脚本与一键运行脚本，基于 torchaudio 的 LibriMix 数据集接口，默认采样率 16k：

```bash
cd scripts/osd
export LIBRIMIX_ROOT=/abs/path/to/LibriMix   # 指向包含 Libri3Mix/Libri2Mix 的上级目录
bash ./test_overlap_3src.sh
```

可选：启用目标说话人筛选（只对目标的分支做 ASR；clean 段也会筛选）：

```bash
export SPK_EMBED_MODEL=../../models/speaker-recongition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
export ENROLL_WAVS=/path/to/target1.wav,/path/to/target2.wav  # 可逗号分隔或多次传参
export SV_THRESHOLD=0.6
bash ./test_overlap_3src.sh
```

说明：

- 三路脚本为 `offline_overlap_3src.py`，会对 overlap 片段分离出 3 路，并基于声纹注册（`SpeakerEmbeddingManager`）仅保留与目标匹配的一路进行 ASR；若无匹配则跳过该段。
- 若未提供注册信息，脚本将对三路全部执行 ASR（与两路逻辑一致）。
- 三路脚本在 `segments.csv` 中会额外写入 `match` 列（`target`/`unknown`），并在 `summary.json` 中增加命中/未命中统计字段（见上表）。

### 评估（带源语音，对 OSD/分离/可选 ASR 进行量化）

我们提供评估脚本用于在 Libri2Mix 8k 上对“预测的重叠片段”做质量评估：

```bash
cd scripts/osd
# 快速启动（推荐用包装脚本，会根据环境拼好参数）
./eval_overlap_sources.sh

# 或直接运行 Python（可自行传参）
python3 evaluate_with_sources.py \
  --max-files 30 \
  --osd-backend pyannote --sep-backend asteroid \
  --activity-thr 0.03 --min-overlap-dur 0.4
```

可选开启 ASR 对比（仅用于“重叠 vs 分离后 vs clean”伪参考对比）：

```bash
export ENABLE_ASR=1   # 在 eval_overlap_sources.sh 中生效
./eval_overlap_sources.sh
```

输出目录：`test_overlap_eval/<YYYY-MM-DD_HH-MM-SS>/`

- `evaluation.json`：聚合指标（见下）
- `overlap_details.csv`：每个“预测重叠片段”的 SI-SDR 明细（及可选 ASR 明细）

evaluation.json 主要字段说明：

- `osd`: OSD 精度/召回/F1/IoU（基于能量门限构造的 GT 掩码）
- `separation`: 在“预测重叠片段”上计算的 SI-SDR 与 SI-SDRi（Permutation Invariant）
- `asr`（可选 ENABLE_ASR）:
  - `overlap_mixture` / `overlap_separated` / `clean` 的 WER/CER 分布聚合
- `timing`: 分阶段耗时与 RTF
  - `rtf_total = time_wall_sec / audio_total_sec`
  - `rtf_osd`、`rtf_sep_*`、`rtf_asr` 为各阶段相对实时比
- `cpu`: 进程 CPU 使用率（已“归一化”到 0–100%）
  - `cpu_avg_percent` / `cpu_peak_percent`: 归一化后百分比（约等于原始值除以逻辑核数，最高截断 100）
  - `cpu_avg_percent_raw` / `cpu_peak_percent_raw`: 原始 psutil 聚合值（可能大于 100%）
  - `cpu_logical_cores`: 逻辑核数，`normalized: true` 表示已做归一化
- `gpu`（若已启用 GPU 监控）: 利用率/显存统计（脚本内使用 pynvml/torch 采集）

提示：

- 评估脚本仅在“预测为重叠”的片段上计算分离质量（SI-SDR），未预测到的 GT 重叠不计入。
- 由于域不匹配，默认的 Conv-TasNet 在中文语音上可能出现 SI-SDRi 为负的情况；可更换更合适的分离模型或做全局分离策略优化。

#### n_src 与三参考（Libri3Mix）评估

- 通过包装脚本设置 `SEP_NSRC` 传入 `--sep-nsrc`（默认 2，支持 3）：

```bash
cd scripts/osd
export SEP_NSRC=3
# 可选：export SEP_CHECKPOINT=/abs/path/to/pytorch_model.bin
bash ./eval_overlap_sources.sh
```

- 若数据样本包含 `s3_wav:FILE` 字段（Libri3Mix），评估脚本会自动按三参考 PIT 计算分离段的 SI-SDR/SI-SDRi：
  - GT 重叠帧定义为“≥2 路活动”。
  - 从 N 路分离输出中选择最优三路（组合 × 排列遍历）并与 3 个参考对齐。
  - `overlap_details.csv` 中新增 `k_refs` 列显示参考数（2 或 3），以及 `selected_pred_indices` 记录 PIT 选择的预测路索引（例如 `0;2;4`）。
  - ASR 的 `overlap_separated` 仍仅在 2 路情况下统计；在 `sep_nsrc!=2` 或三参考时将跳过并在 JSON 中给出原因。

### Hugging Face Token（仅在需要时）

- 若 pyannote 的 `pyannote/overlapped-speech-detection` 需要认证，请在环境中设置以下任一变量（按优先级读取）：
  - `PYANNOTE_TOKEN` 或 `HF_TOKEN` 或 `HUGGINGFACE_TOKEN`
- 代码会自动将该 token 注入 OverlapAnalyzer（无需手动传参）。

### 分离模型 checkpoint（自动下载，可覆盖）

- 默认行为：若未提供 `--sep-checkpoint`，程序会自动从 Hugging Face 下载一个公开可用的 Conv-TasNet 权重并缓存：
  - 当 `n_src=2`：Repo `mpariente/ConvTasNet_WHAM_sepclean`，File `pytorch_model.bin`
  - 当 `n_src=3`：Repo `JorisCos/ConvTasNet_Libri3Mix_sepclean_16k`，File `pytorch_model.bin`
- 如需替换默认来源，可通过环境变量覆盖：

```bash
export ASTEROID_SEP_REPO_ID_2="YourRepo/YourModelFor2Src"     # 覆盖 2 路仓库
export ASTEROID_SEP_FILENAME_2="your_2src_weights.bin"        # 覆盖 2 路文件名
export ASTEROID_SEP_REPO_ID_3="YourRepo/YourModelFor3Src"     # 覆盖 3 路仓库
export ASTEROID_SEP_FILENAME_3="your_3src_weights.bin"        # 覆盖 3 路文件名
```

- 如已有本地 checkpoint，可在运行脚本时显式传入：

```bash
python3 ./offline_overlap_mvp.py \
  ...其它参数... \
  --sep-checkpoint /path/to/conv_tasnet_checkpoint.pt
```

## 所使用的模型与文档/下载地址

重叠语音检测（OSD）

- 模型：pyannote/overlapped-speech-detection
- 代码/文档：<https://github.com/pyannote/pyannote-audio>
- 模型页（HF）：<https://huggingface.co/pyannote/overlapped-speech-detection>
- 说明：多数 pyannote 模型需要 HF Token 授权

语音分离（2 说话人）

- 框架：Asteroid（Conv-TasNet）<https://github.com/asteroid-team/asteroid>
- 默认权重（自动下载）：mpariente/ConvTasNet_WHAM_sepclean
  - 模型页（HF）：<https://huggingface.co/mpariente/ConvTasNet_WHAM_sepclean>

语音分离（3 说话人）

- 默认权重（自动下载）：JorisCos/ConvTasNet_Libri3Mix_sepclean_16k（当 n_src=3）
  - 模型页（HF）：<https://huggingface.co/JorisCos/ConvTasNet_Libri3Mix_sepclean_16k>

ASR（可选，多分支）

- SenseVoice（Sherpa-ONNX 多语）
  - Sherpa-ONNX 文档：<https://k2-fsa.github.io/sherpa/>
  - 发布页集合（含多模型）：<https://github.com/k2-fsa/sherpa-onnx/releases>
- Paraformer（FunASR 系列）
  - FunASR 项目：<https://github.com/alibaba-damo-academy/FunASR>
  - ModelScope 示例（中文 Paraformer）：<https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary>
- RNN-T（encoder/decoder/joiner）
  - 同 Sherpa-ONNX 发布页（手动指定 --encoder/--decoder/--joiner 时使用）

数据集（评估用）

- Libri2Mix/LibriMix：<https://github.com/JorisCos/LibriMix>
- LibriSpeech（源语料）：<https://www.openslr.org/12>

相关工具/库

- huggingface_hub（权重下载）：<https://github.com/huggingface/huggingface_hub>
- torchaudio（音频 I/O/重采样）：<https://pytorch.org/audio/stable/>
- psutil（CPU 监控）：<https://github.com/giampaolo/psutil>
- pynvml（GPU 监控）：<https://pypi.org/project/pynvml/>

许可与合规：

- pyannote 多为研究许可，请确认商业使用限制。
- FunASR/Paraformer 权重各自附带 LICENSE，请在部署前复核。
- WHAM/LibriMix/LibriSpeech 遵循各自数据协议，请按条款使用。

## 目录结构

### 脚本

在 scripts 目录下包含以下脚本：

1. install.sh 用于安装依赖、模型、数据集等
2. run.sh 运行主程序（非流式 SID+ASR）
3. test.sh 运行基线 benchmark
4. `osd/offline_overlap_mvp.py` 离线 OSD + 分离 + **ASR（无 SID / 无 CER）** 管线
5. `osd/test_overlap.sh` 一键运行最新 ASR-only Overlap MVP
6. `osd/offline_overlap_3src.py` 三路分离版（LibriMix/Libri3Mix；支持目标说话人筛选）
7. `osd/test_overlap_3src.sh` 一键运行三路分离（需设置 `LIBRIMIX_ROOT`）
8. generate-speaker-text.sh 生成说话人文本
9. split_speakers.py 划分 train/test 数据集
10. version.py 打印环境信息

### 主程序

`speaker-identification-with-vad-non-streaming-asr.py`
