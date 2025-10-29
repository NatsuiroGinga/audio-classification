## 表现解读

- 重叠段命中率很高
  - overlap：命中 94 / 看见 101 ≈ 93.1%。这符合“先分离 →SV 选择目标一路”的预期，说明三路分离 + 目标筛选在重叠场景下较稳。
- clean 段命中率仍然偏低
  - clean：命中 6 / 看见 98 ≈ 6.1%，和上次结果趋势一致（clean 段普遍短、静态，短时嵌入不稳定，SV 容易 miss）。
  - 匹配到的 clean 总时长仅 5.1s，但看见的 clean 音频时长有 76.04s，说明多数 clean 段被判为非目标。
- 分离质量稳定
  - K=3 的分离质量评估：SI-SDR 平均 10.16 dB，SI-SDRi 提升 13.54 dB，和上次一致，说明分离模型输出稳定。
- 资源与性能
  - time_osd≈2.315s、time_sep≈2.782s、time_asr≈13.635s，总体瓶颈不在 OSD/分离，ASR 占用较低；CPU 多核占用正常，GPU 显存极低（仅 256MB 预留）。

## 仍然存在的问题

- clean 段命中偏低是主要短板
  - 这会导致大量“看见但未输出”的 clean 片段（segments_missed_clean=92），可见在 SV 阈值与短段稳定性上还有提升空间。

## 建议的下一步优化（按收益排序）

1. 为 clean 段增加“最小 SV 判定时长”

- 新增参数 --min-clean-sv-dur（建议 0.8~1.2s），不足该时长的 clean 段不做 SV 判定（既不计 seen 也不计 miss），或先累积/拼接后再判定。
- 预期可显著降低 clean 的误拒（假阴性），提升整体命中率，避免短片段导致嵌入噪声。

2. 提供稳健的短段打分策略

- 对短段 embedding 采用多窗采样并取均值/中位数打分，或做轻量时序聚合（例如滑窗拼接到至少 1s 再判）。
- 或对 clean 段单独使用较低的 SV 阈值（例如 --sv-threshold-clean），与 overlap 段区分。

3. 复现性与分析性

- 你现在可以通过 --seed 固定目标源的随机选择，稳定回归评测。
- 如需分析未命中，建议加一个 --emit-unmatched 选项，将 miss 的段输出到独立 CSV/JSONL（带 match=unknown），默认关闭。

4. 小的可读性改进

- 把 metrics 中的 segments_overlap_streams 重命名为 segments_overlap，避免和 separated_streams 混淆（功能不变）。
