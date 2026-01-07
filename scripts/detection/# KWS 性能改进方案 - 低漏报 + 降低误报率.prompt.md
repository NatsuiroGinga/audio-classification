# KWS 性能改进方案 - 低漏报 + 降低误报率

## 当前状态分析

### 现有配置性能对比

| 配置       | Boost | Threshold | FRR   | 谐音误报 | 策略             |
| ---------- | ----- | --------- | ----- | -------- | ---------------- |
| 低漏报优先 | 1.5   | 0.45      | 1.4%  | 29/36    | 宽松检测，高召回 |
| 零误报优先 | 0.3   | 0.40      | 13.9% | 0/36     | 严格检测，低误报 |
| 平衡方案   | 0.7   | 0.60      | 9.7%  | 20/36    | 中等权衡         |

### 问题定义

**目标**: 最低漏报率（FRR ≤ 2%） + 最低误报率（尤其是谐音误报）

**核心矛盾**: KWS 单阶段检测无法同时满足低 FRR 和低 FA

- 宽松阈值 → 低 FRR，高 FA（谐音触发）
- 严格阈值 → 高 FRR，低 FA（漏检真实唤醒词）

## 改进策略：二次验证架构

### 方案 1: KWS + ASR 二次验证（谐音强制映射）

#### 架构流程

```
音频输入
  ↓
[KWS 阶段] 宽松阈值检测 (boost=1.5, threshold=0.45)
  ↓ 触发
[ASR 阶段] 对触发片段重新解码
  ↓
[决策阶段]
  - 如果 ASR = "你好真真" → 接受
  - 如果 ASR = 谐音词（镇镇/珍珍/阵阵） → 强制映射为"你好真真"，接受
  - 其他情况 → 拒绝
```

#### 优势

- **低 FRR**: 使用宽松 KWS 阈值，确保真实唤醒词不会漏检
- **低 FA**: ASR 二次验证过滤掉非目标词
- **谐音友好**: 将常见谐音映射为目标词，提升用户体验

#### 技术实现要点

**1. ASR 模型选择**

- **SenseVoice** (推荐)
  - 优势: 高精度，支持情感/事件检测
  - 路径: `models/asr/sense-voice-chinese`
  - 已集成在 `src/model.py`

**2. 触发片段提取**

- 使用 KWS 返回的 `timestamps` 确定起止时间
- 前后各扩展 0.2-0.3 秒作为上下文
- 示例: `timestamps = [0.12, 0.16, ..., 0.80]` → 提取 `[0.0-1.1]` 秒

**3. 谐音词白名单**

```python
HOMOPHONE_WHITELIST = {
    "你好镇镇", "你好珍珍", "你好阵阵", "你好甄甄",
    "李浩真真", "泥豪真真", "尼好真真",
}

TARGET_KEYWORD = "你好真真"
```

**4. 决策逻辑**

```python
def verify_keyword(asr_text: str, target: str, homophones: set) -> bool:
    # 清理空格和标点
    clean_text = clean_asr_output(asr_text)

    # 完全匹配
    if target in clean_text:
        return True

    # 谐音匹配
    for homophone in homophones:
        if homophone in clean_text:
            return True  # 强制映射为目标词

    return False
```

### 方案 2: KWS + 音调分类器

#### 架构流程

```
音频输入
  ↓
[KWS 阶段] 检测到"真/镇/珍/甄"
  ↓
[音调分类器] 提取基频（F0），判断声调
  - ēn (第一声): 平
  - én (第二声): 升
  - ěn (第三声): 降升
  - èn (第四声): 降
  ↓
[决策] 如果是第一声（ēn）→ 接受
```

#### 优势

- 更精细的声调区分能力
- 无需完整 ASR 解码，延迟更低

#### 劣势

- 需要训练音调分类器
- 对噪声敏感
- 实现复杂度高

## 推荐实施路径

### Phase 1: 快速验证（1-2 天）

**目标**: 验证二次 ASR 验证方案的有效性

1. **实现 KWS + ASR 二次验证**

   - 创建 `src/detection/verifier.py`（已有雏形）
   - 集成 SenseVoice ASR
   - 实现谐音白名单匹配

2. **测试集评估**

   - 正样本（144 个）
   - 谐音负样本（36 个）
   - 正常负样本（其他词汇）

3. **性能指标**
   - FRR（False Rejection Rate）
   - FA（False Alarm）- 区分谐音 FA 和其他 FA
   - 延迟（KWS + ASR 总延迟）
   - RTF（Real-Time Factor）

### Phase 2: 优化与调优（2-3 天）

1. **延迟优化**

   - 缓存 ASR 模型，避免重复加载
   - 异步 ASR 解码
   - 提前截取音频片段（在 KWS 解码时并行）

2. **准确率优化**

   - 调整谐音白名单范围
   - 实验不同的 ASR 模型（Paraformer vs SenseVoice）
   - 模糊匹配阈值（编辑距离）

3. **鲁棒性测试**
   - 噪声环境测试
   - 不同说话人测试
   - 边界情况（极短/极长音频）

### Phase 3: 生产部署（1-2 天）

1. **流式支持**

   - 将二次验证整合到流式检测流程
   - 处理实时音频流

2. **资源优化**

   - 内存占用优化
   - CPU/GPU 利用率优化
   - 多实例并发支持

3. **监控与日志**
   - 检测统计（触发次数、验证通过率）
   - 性能指标监控
   - 调试日志

## 实施细节

### 文件结构

```
src/detection/
├── model.py              # 现有 KWS 模型
├── verifier.py           # 新增：二次验证器
│   ├── KeywordVerifier   # 主验证类
│   ├── ASRVerifier       # ASR 解码器封装
│   └── HomophoneMapper   # 谐音映射逻辑
└── pipeline.py           # 新增：完整检测流程

scripts/detection/
├── test_verifier.py      # 验证器单元测试
├── evaluate_verifier.py  # 在测试集上评估
└── benchmark_verifier.py # 性能基准测试
```

### 关键代码示例

#### KeywordVerifier 接口

```python
class KeywordVerifier:
    def __init__(
        self,
        kws_model_dir: str,
        asr_model_dir: str,
        target_keyword: str = "你好真真",
        homophone_whitelist: Optional[Set[str]] = None,
        kws_boost: float = 1.5,
        kws_threshold: float = 0.45,
    ):
        self.kws = create_kws_model(kws_model_dir, boost=kws_boost, threshold=kws_threshold)
        self.asr = create_asr_model(asr_model_dir)
        self.target = target_keyword
        self.homophones = homophone_whitelist or DEFAULT_HOMOPHONES

    def detect(self, audio: np.ndarray, sample_rate: int) -> Optional[VerifiedResult]:
        # Stage 1: KWS 检测
        kws_result = self.kws.detect(audio, sample_rate)
        if not kws_result:
            return None

        # Stage 2: 提取触发片段
        segment = extract_segment(audio, kws_result.timestamps, sample_rate)

        # Stage 3: ASR 二次验证
        asr_text = self.asr.decode(segment, sample_rate)

        # Stage 4: 决策
        if self._verify_text(asr_text):
            return VerifiedResult(
                keyword=self.target,
                kws_result=kws_result,
                asr_text=asr_text,
                is_homophone=asr_text in self.homophones,
            )

        return None

    def _verify_text(self, asr_text: str) -> bool:
        clean_text = self._clean_text(asr_text)
        return self.target in clean_text or any(h in clean_text for h in self.homophones)
```

#### 性能评估脚本

```python
def evaluate_verifier(verifier, test_data):
    tp, fp, fn = 0, 0, 0
    homophone_accepted = 0

    for sample in test_data["positive"]:
        result = verifier.detect(sample["audio"], sample["sr"])
        if result:
            tp += 1
        else:
            fn += 1

    for sample in test_data["negative_homophone"]:
        result = verifier.detect(sample["audio"], sample["sr"])
        if result:
            homophone_accepted += 1
            fp += 1

    for sample in test_data["negative_normal"]:
        result = verifier.detect(sample["audio"], sample["sr"])
        if result:
            fp += 1

    frr = fn / len(test_data["positive"]) * 100
    fa_rate = fp / (len(test_data["negative_homophone"]) + len(test_data["negative_normal"])) * 100

    print(f"FRR: {frr:.1f}%")
    print(f"FA Rate: {fa_rate:.1f}%")
    print(f"Homophone Accepted: {homophone_accepted}/{len(test_data['negative_homophone'])}")
```

## 预期性能提升

### 目标指标

- **FRR**: ≤ 2% (vs 现在 1.4%，基本持平)
- **谐音误报**: 完全接受（强制映射）→ 0% 真正误报
- **其他误报**: < 5% (vs 现在未知)
- **延迟**: < 200ms (KWS ~30ms + ASR ~150ms)
- **RTF**: < 0.2x

### 风险与缓解

**风险 1: ASR 解码失败**

- 缓解: 如果 ASR 失败，降级到纯 KWS 结果
- 监控: 记录 ASR 失败率

**风险 2: 延迟过高**

- 缓解: 异步处理，提前截取音频
- 监控: P99 延迟

**风险 3: ASR 误识别**

- 缓解: 使用高精度 ASR 模型（SenseVoice）
- 监控: ASR 准确率

## 下一步行动

1. **立即执行**（今天）

   - [ ] 实现 `src/detection/verifier.py` 基础框架
   - [ ] 创建评估脚本 `scripts/detection/evaluate_verifier.py`
   - [ ] 在小规模数据集上测试（10 个正样本 + 10 个负样本）

2. **明天执行**

   - [ ] 完整测试集评估（144 + 36 + 正常负样本）
   - [ ] 性能分析与调优
   - [ ] 延迟优化

3. **后天执行**
   - [ ] 流式支持
   - [ ] 生产部署准备
   - [ ] 文档与示例

## 附录：谐音词库

### 已知谐音词（来自测试集）

- 你好镇镇（zhèn zhèn）
- 你好珍珍（zhēn zhēn，但声调不同）
- 李浩真真（lǐ hào zhēn zhēn）
- 泥豪真真（ní háo zhēn zhēn）

### 潜在扩展

- 你好甄甄、你好阵阵、你好臻臻
- 拼音相近但不同字的组合

### 处理策略

- **激进策略**（用户要求）：所有包含"真/镇/珍/甄/阵/臻"的组合都映射为"你好真真"
- **保守策略**：仅映射测试集中出现的高频谐音词
- **推荐**: 从保守策略开始，根据实际使用反馈扩展
