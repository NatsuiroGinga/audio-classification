# Decoy Keywords 参数优化计划

## 当前状态

### 测试配置

- **目标关键词**: "你好真真" (nǐ hǎo zhēn zhēn)
- **诱饵关键词**: 11 个谐音变体
  - Level 1 (声调): 镇镇(zhèn), 诊诊(zhěn), 振振(zhèn)
  - Level 2 (韵尾): 正正(zhèng), 争争(zhēng), 整整(zhěng), 征征(zhēng)
  - Level 3 (声母): 认认(rèn), 曾曾(zéng), 怎怎(zěn), 真真(rěn)
- **当前参数**: 统一 boost=2.0, threshold=0.45

### 测试结果

- **正样本 (144)**:
  - TP (检测到目标): 121
  - FP_decoy (误识别为诱饵): 18 (12.5%)
  - FN (漏检): 5
  - **FRR**: 3.47%
- **谐音负样本 (36)**:
  - 被拦截: 24
  - 漏过: 10
  - **Decoy intercept rate**: 70.6%
- **真实负样本 (144)**:
  - FA_true: 0
  - **误报率**: 0.00%

## 优化目标

1. **FRR ≤ 2%** (当前 3.47%)
2. **Decoy intercept rate ≥ 95%** (当前 70.6%)
3. **FA_true = 0** (当前满足)

## 问题分析

### 问题 1: FRR 过高 (3.47% > 2%)

**根本原因**:

- 诱饵关键词竞争力过强，抢走了目标关键词的检测
- 18 个正样本被误识别为诱饵，说明目标关键词缺乏竞争优势

**解决方向**:

- **提高目标关键词 boost** (2.5-3.0)
- **降低诱饵关键词 boost** (1.5-1.8)
- **差异化参数策略**: 目标关键词使用更高的 boost

### 问题 2: Decoy intercept rate 过低 (70.6% < 95%)

**根本原因**:

- 10 个谐音样本漏过拦截，说明诱饵关键词对谐音不够敏感
- 可能原因:
  1. threshold 过高 (0.45)，导致诱饵不易触发
  2. 诱饵 boost 不足，导致检测置信度不够

**解决方向**:

- **降低 threshold** (0.40-0.42)
- **保持或提高诱饵 boost** (如果使用差异化策略，则适度降低)

### 问题 3: 正样本误分类 (12.5%)

**需要分析**:

- 哪些诱饵关键词导致误分类？
- 误分类样本的特征 (时长、能量、音色)?
- 是否特定语速/音调更容易误分类？

**解决方向**:

- 通过 `analyze_misclassified.py` 分析模式
- 根据分析结果调整特定诱饵的参数

## 优化策略

### 策略 1: 微调当前配置 (快速测试)

**假设**: 降低 threshold 可以提高 intercept rate

**执行**:

```bash
# 修改 test/detection/decoy_keywords_clean.txt
# 将所有关键词的 threshold 从 0.45 改为 0.40

# 重新生成 keywords.txt
cd scripts/detection
python generate_keywords.py \
  --raw-file ../../test/detection/decoy_keywords_clean.txt \
  --tokens-file ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt \
  --output-file ../../test/detection/decoy_keywords.txt

# 重新测试
python param_optimization_with_decoy.py
```

**预期**:

- Intercept rate 提升 5-10%
- FRR 可能略微上升 (需要观察)

### 策略 2: 网格搜索 - 统一参数模式 (全面搜索)

**执行**:

```bash
cd scripts/detection

# 统一参数搜索 (所有关键词使用相同的 boost/threshold)
python optimize_decoy_params.py \
  --mode uniform \
  --output uniform_search_results.json

# 预计测试 25 种配置，耗时约 15-25 分钟
```

**搜索范围**:

- boost: [1.5, 1.8, 2.0, 2.2, 2.5]
- threshold: [0.40, 0.42, 0.45, 0.47, 0.50]

**预期**:

- 找到一组最优的统一参数
- 可能无法同时满足所有目标 (trade-off between FRR and intercept rate)

### 策略 3: 网格搜索 - 差异化参数模式 (精准优化)

**执行**:

```bash
cd scripts/detection

# 差异化参数搜索 (目标和诱饵使用不同的 boost)
python optimize_decoy_params.py \
  --mode differential \
  --output differential_search_results.json

# 预计测试 12 种配置，耗时约 8-15 分钟
```

**搜索范围**:

- target_boost: [2.2, 2.5, 2.8, 3.0]
- decoy_boost: [1.5, 1.8, 2.0]
- threshold: 固定 0.40

**预期**:

- 通过给目标关键词更高的 boost，降低 FRR
- 同时保持诱饵对谐音的敏感度，维持 intercept rate
- 更有可能同时满足所有目标

## 执行计划

### Phase 1: 问题诊断 (5 分钟)

```bash
cd scripts/detection

# 分析误分类样本
python analyze_misclassified.py \
  --output-json misclassified_analysis.json

# 查看哪些诱饵导致最多误分类
# 查看哪些谐音样本漏过拦截
```

### Phase 2: 快速测试 (10 分钟)

```bash
# 测试 threshold=0.40
# 修改 decoy_keywords_clean.txt 后重新生成和测试
python generate_keywords.py --raw-file ... --output-file ...
python param_optimization_with_decoy.py
```

### Phase 3: 差异化搜索 (15 分钟)

```bash
# 执行差异化参数搜索
python optimize_decoy_params.py --mode differential

# 查看结果，选择最优配置
cat differential_search_results.json
```

### Phase 4: (可选) 统一搜索 (25 分钟)

```bash
# 如果差异化搜索未找到满意结果，执行统一搜索
python optimize_decoy_params.py --mode uniform
```

### Phase 5: 应用最优配置

```bash
# 根据搜索结果，更新 decoy_keywords_clean.txt
# 重新生成 decoy_keywords.txt
# 最终验证
```

## 评估标准

### 优先级 1 (必须满足)

- ✅ FA_true = 0 (不能误报真实负样本)

### 优先级 2 (核心目标)

- ✅ FRR ≤ 2% (漏检率)
- ✅ Decoy intercept rate ≥ 95% (谐音拦截率)

### 优先级 3 (次要优化)

- 正样本误分类率 ≤ 5% (FP_decoy / total_positive)
- 推理速度 RTF < 0.1 (实时性)

## 预期结果

### 最优场景

- FRR: 1.5-2.0%
- Intercept rate: 95-100%
- FA_true: 0
- 配置: target_boost=2.5-2.8, decoy_boost=1.5-1.8, threshold=0.40

### 可接受场景

- FRR: 2.0-2.5%
- Intercept rate: 90-95%
- FA_true: 0
- 配置: 可能需要在 FRR 和 intercept rate 之间权衡

### 回退方案

如果无法同时满足 FRR 和 intercept rate:

1. **优先保证 intercept rate ≥ 95%** (避免谐音误触发)
2. **接受 FRR ≤ 3%** (可以通过其他环节补偿，如二次确认)

## 工具说明

### optimize_decoy_params.py

- 自动化网格搜索工具
- 支持统一/差异化两种模式
- 自动生成 keywords.txt 文件
- 输出 JSON 格式结果

### analyze_misclassified.py

- 误分类样本分析工具
- 按诱饵关键词分组
- 提供音频特征统计
- 输出 JSON 格式详细信息

### param_optimization_with_decoy.py

- 单配置测试工具
- 用于验证最终选定的配置
- 输出详细指标

## 后续工作

1. **扩展到其他唤醒词**: 将 decoy 策略应用到"小爱同学"、"你好问问"
2. **真实环境测试**: 在噪声环境、边缘设备上验证
3. **长时运行测试**: 测试 FA/Hr 指标 (每小时误报次数)
4. **用户体验优化**: 根据实际使用反馈调整参数
