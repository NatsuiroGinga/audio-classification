"""KWS 优化配置文件

统一管理关键词唤醒的所有配置参数。
包含：模型参数、关键词定义、decoy 配置等。
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# =====================================================
# 目标关键词配置
# =====================================================


@dataclass
class TargetKeywordConfig:
    """目标关键词配置"""

    pinyin: str = "n ǐ h ǎo zh ēn zh ēn"
    text: str = "你好真真"
    boost: float = 2.0
    threshold: float = 0.20

    def to_kws_format(self) -> str:
        """转换为 KWS 关键词格式"""
        return f"{self.pinyin} :{self.boost} #{self.threshold} @{self.text}"


# =====================================================
# Decoy 关键词配置（谐音词）
# =====================================================

# 优化后的 decoy 关键词
# 说明：
# - 移除了三声变体 (诊诊/整整)，因为陕西方言声音会误识别
# - 保留四声、一声、声母变体
# - 所有 decoy 使用统一参数: boost=1.0, threshold=0.20
DECOY_KEYWORDS = {
    "你好镇镇": ("zh èn zh èn", 1.0, 0.20),  # 四声
    "你好正正": ("zh èng zh èng", 1.0, 0.20),  # 四声
    "你好争争": ("zh ēng zh ēng", 1.0, 0.20),  # 一声
    "你好认认": ("r èn r èn", 1.0, 0.20),  # 声母变体
    "你好曾曾": ("c éng c éng", 1.0, 0.20),  # 声母变体
    "你好怎怎": ("z ěn z ěn", 1.0, 0.20),  # 声母变体
}

# 已排除的 decoy（因与某些 TTS 声音产生干扰）
EXCLUDED_DECOY_KEYWORDS = {
    "你好诊诊": "第三声，陕西方言声音会误识别",
    "你好整整": "第三声，陕西方言声音会误识别",
}

# 无法区分的同音字（KWS 声学识别无法区分）
INDISTINGUISHABLE_HOMOPHONES = {
    "目标": {"珍": "第一声", "甄": "第一声", "臻": "第一声"},
    "说明": "一声同音字与目标词'真'音频波形完全相同，KWS 无法区分，无需处理",
}

# 模型无法区分的"你好"变体
UNRESOLVABLE_NIHAO_VARIANTS = {
    "泥豪真真": "83% 被误识别为目标词（KWS 模型局限）",
    "李浩真真": "60% 被误识别为目标词（KWS 模型局限）",
    "说明": "这是 KWS 模型的声学局限，无法通过 decoy 或 ASR 解决",
}


# =====================================================
# KWS 模型配置
# =====================================================


@dataclass
class KWSModelConfig:
    """KWS 模型配置"""

    model_dir: str = "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    use_int8: bool = True  # 使用 INT8 量化
    epoch: int = 12  # 平均轮数
    avg: int = 2  # 模型平均
    keywords_threshold: float = 0.25  # 全局阈值（可被 keywords.txt 覆盖）


# =====================================================
# 性能指标（已优化）
# =====================================================


@dataclass
class PerformanceMetrics:
    """性能指标"""

    frr: float = 0.0139  # 漏报率 1.39%
    far: float = 0.0746  # 误报率 7.46%（排除不可区分样本后）
    rtf: float = 0.0171  # 实时因子

    # 测试数据
    positive_samples: int = 144  # 正样本数
    negative_samples: int = 456  # 负样本数（排除84个后）
    excluded_samples: int = 84  # 排除的不可区分样本

    # TTS 声音覆盖
    voices_count: int = 8  # TTS 声音种类数
    snr_levels: tuple = (5, 10, 15, 20, 30, "clean")  # SNR 水平


# =====================================================
# 测试数据集配置
# =====================================================


@dataclass
class TestDatasetConfig:
    """测试数据集配置"""

    positive_dir: str = "dataset/kws_test_data_merged/positive"
    negative_dir: str = "dataset/kws_test_data_merged/negative"
    metadata_file: str = "dataset/kws_test_data_merged/metadata.json"


# =====================================================
# 助手函数
# =====================================================


def get_decoy_keywords_kws_format() -> str:
    """获取 KWS 格式的 decoy 关键词"""
    lines = []
    for keyword, (pinyin, boost, threshold) in DECOY_KEYWORDS.items():
        line = f"n ǐ h ǎo {pinyin} :{boost} #{threshold} @{keyword}"
        lines.append(line)
    return "\n".join(lines)


def generate_keywords_txt(target: TargetKeywordConfig) -> str:
    """生成 keywords.txt 内容"""
    target_line = target.to_kws_format()
    decoy_lines = get_decoy_keywords_kws_format()
    return target_line + "\n" + decoy_lines


# =====================================================
# 全局配置实例
# =====================================================

TARGET_KEYWORD = TargetKeywordConfig()
MODEL_CONFIG = KWSModelConfig()
PERFORMANCE = PerformanceMetrics()
TEST_DATASET = TestDatasetConfig()
