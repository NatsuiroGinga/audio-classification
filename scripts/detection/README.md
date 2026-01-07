## KWS (关键词唤醒) 模块使用指南

### 📁 目录结构

```
scripts/detection/
├── __init__.py
├── core/                    # 核心功能脚本
│   ├── benchmark_kws.py     # 📊 主评估脚本（FRR/FAR/RTF）
│   ├── test_nihao_zhenzhen.py # 🧪 快速单文件测试
│   └── demo_wakeword.py     # 🎯 交互式演示（参数预设）
└── utils/                   # 工具脚本
    ├── data_generator.py    # 🎤 TTS 生成测试数据
    ├── merge_test_data.py   # 📦 合并数据集
    ├── generate_keywords.py # 🔤 中文关键词格式转换
    └── generate_keywords_zh_en.py # 🔤 中英文混合关键词转换
```

### 🚀 常见任务

#### 1. 快速测试单个音频文件

```bash
cd scripts/detection/core
python test_nihao_zhenzhen.py --wav /path/to/audio.wav
```

#### 2. 完整评估（FRR/FAR/RTF）

```bash
cd scripts/detection/core
python benchmark_kws.py \
  --model-dir ../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 \
  --positive-dir /path/to/positive/samples \
  --negative-dir /path/to/negative/samples \
  --keywords-file ../../test/detection/decoy_keywords.txt
```

#### 3. 生成测试数据

```bash
cd scripts/detection/utils
python data_generator.py \
  --keyword "你好真真" \
  --num-positive 144 \
  --num-negative 540 \
  --output-dir ../../dataset/kws_test_data
```

#### 4. 合并多个数据集

```bash
cd scripts/detection/utils
python merge_test_data.py \
  --dataset1 ../../dataset/original \
  --dataset2 ../../dataset/expanded \
  --output-dir ../../dataset/merged
```

#### 5. 关键词格式转换

```bash
cd scripts/detection/utils
# 中文
python generate_keywords.py --keyword "你好真真" \
  --tokens-file ../../models/.../tokens.txt

# 中英文混合
python generate_keywords_zh_en.py --keyword "HELLO WORLD"
```

#### 6. 交互式演示

```bash
cd scripts/detection/core
python demo_wakeword.py /path/to/test.wav --config balanced
```

### 📊 当前优化配置

**Target Keyword (你好真真)**

- boost: 2.0
- threshold: 0.20

**Decoy Keywords (移除三声)**

- boost: 1.0
- threshold: 0.20
- 包含: 镇镇(4 声), 正正(4 声), 争争(1 声), 认认, 曾曾, 怎怎

### 📈 性能指标 (已优化)

| 指标 | 数值   | 说明          |
| ---- | ------ | ------------- |
| FRR  | 1.39%  | 低漏报 ✅     |
| FAR  | 7.46%  | 中等误报 ⚠️   |
| RTF  | 0.0171 | 实时性优秀 ✅ |

**说明**: 排除了模型无法区分的"你好"谐音 (泥豪真真/李浩真真, 84 个样本)

### 🔧 核心模块 (src/detection/)

- `src/detection/model.py` - KWS 模型包装器
- `src/detection/decoy_filter.py` - 谐音过滤器 (8 个 decoy 关键词)
- `src/detection/verifier.py` - ASR 验证器 (已禁用)
