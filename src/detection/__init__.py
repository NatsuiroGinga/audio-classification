"""KWS Detection 模块初始化。

本模块提供关键词唤醒（Keyword Spotting）能力。
"""

from .model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KEYWORD_NIHAO_WENWEN,
    KEYWORD_XIAOAI_TONGXUE,
    KEYWORD_XIAOMI_XIAOMI,
    KeywordDetection,
    KeywordSpotterModel,
    KWSModelConfig,
    create_kws_model,
    G_SAMPLE_RATE,
)

__all__ = [
    "KeywordDetection",
    "KeywordSpotterModel",
    "KWSModelConfig",
    "create_kws_model",
    "G_SAMPLE_RATE",
    "KEYWORD_NIHAO_ZHENZHEN",
    "KEYWORD_NIHAO_WENWEN",
    "KEYWORD_XIAOAI_TONGXUE",
    "KEYWORD_XIAOMI_XIAOMI",
]
