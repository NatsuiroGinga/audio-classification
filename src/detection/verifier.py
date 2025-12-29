"""唤醒词二次验证模块。

解决谐音误报问题：KWS 检测后使用 ASR 进行文本确认。

使用示例:
    verifier = KeywordVerifier(
        asr_model="models/asr/model.onnx",
        keyword_text="你好真真",
    )

    # 在 KWS 检测到后进行验证
    if kws_detected:
        is_valid = verifier.verify(audio_segment)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 采样率
G_SAMPLE_RATE = 16000


@dataclass
class VerifierConfig:
    """验证器配置。"""

    keyword_text: str  # 目标唤醒词文本，如 "你好真真"
    keyword_pinyin: str = ""  # 唤醒词拼音，如 "ni hao zhen zhen"（可选，自动生成）
    asr_model: Optional[str] = None  # ASR 模型路径
    similarity_threshold: float = 0.8  # 文本相似度阈值
    use_fuzzy_match: bool = True  # 是否使用模糊匹配
    use_pinyin_match: bool = True  # 是否使用拼音匹配（解决同音字问题）


class KeywordVerifier:
    """唤醒词二次验证器。

    通过 ASR 识别音频内容，与目标唤醒词进行文本比对，
    过滤掉谐音误报。
    """

    def __init__(self, config: VerifierConfig) -> None:
        self.config = config
        self._asr_model = None
        self._init_asr()

    def _init_asr(self) -> None:
        """初始化 ASR 模型。"""
        if self.config.asr_model:
            try:
                import sherpa_onnx

                # 尝试加载 SenseVoice 或 Paraformer
                model_path = Path(self.config.asr_model)
                if model_path.is_dir():
                    # 目录模式 - 优先使用 int8 模型
                    int8_models = list(model_path.glob("*.int8.onnx"))
                    if int8_models:
                        model_file = int8_models[0]
                    else:
                        model_file = list(model_path.glob("model.onnx"))[0]
                    tokens_file = model_path / "tokens.txt"
                else:
                    model_file = model_path
                    tokens_file = model_path.parent / "tokens.txt"

                self._asr_model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_file),
                    tokens=str(tokens_file),
                    use_itn=True,
                    num_threads=2,
                )
            except Exception as e:
                print(f"ASR 模型加载失败: {e}")
                self._asr_model = None

    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度。

        使用编辑距离归一化后的相似度。
        """
        # 清理文本
        t1 = re.sub(r"[^\u4e00-\u9fff]", "", text1)
        t2 = re.sub(r"[^\u4e00-\u9fff]", "", text2)

        if not t1 or not t2:
            return 0.0

        # 编辑距离
        m, n = len(t1), len(t2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if t1[i - 1] == t2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        edit_distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (edit_distance / max_len)

    def _to_pinyin(self, text: str) -> str:
        """将中文转换为拼音（不带声调）。"""
        try:
            from pypinyin import lazy_pinyin

            clean = re.sub(r"[^\u4e00-\u9fff]", "", text)
            return " ".join(lazy_pinyin(clean))
        except ImportError:
            return text

    def _pinyin_similarity(self, text1: str, text2: str) -> float:
        """基于拼音的相似度计算，解决同音字问题。"""
        py1 = self._to_pinyin(text1)
        py2 = self._to_pinyin(text2)

        if not py1 or not py2:
            return 0.0

        # 拼音序列的编辑距离
        words1 = py1.split()
        words2 = py2.split()

        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

        edit_distance = dp[m][n]
        max_len = max(m, n)
        return 1.0 - (edit_distance / max_len)

    def _fuzzy_match(self, asr_text: str, keyword: str) -> bool:
        """模糊匹配：检查 ASR 结果是否包含目标唤醒词。

        允许一定的容错，如：
        - "你好真真" 匹配
        - "你好，真真" 匹配（标点）
        - "你好真真啊" 匹配（后缀）
        """
        # 移除标点
        clean_asr = re.sub(r"[^\u4e00-\u9fff]", "", asr_text)
        clean_keyword = re.sub(r"[^\u4e00-\u9fff]", "", keyword)

        # 完全包含
        if clean_keyword in clean_asr:
            return True

        # 前缀匹配（唤醒词在开头）
        if clean_asr.startswith(clean_keyword[:3]):
            return (
                self._text_similarity(clean_asr[: len(clean_keyword)], clean_keyword)
                >= 0.75
            )

        return False

    def verify(
        self,
        samples: np.ndarray,
        sample_rate: int = G_SAMPLE_RATE,
    ) -> Tuple[bool, str, float]:
        """验证音频是否包含目标唤醒词。

        Args:
            samples: 音频波形
            sample_rate: 采样率

        Returns:
            (is_valid, asr_text, similarity): 是否验证通过、ASR 识别文本、相似度
        """
        if self._asr_model is None:
            # 无 ASR 模型时跳过验证
            return True, "", 1.0

        # ASR 识别
        stream = self._asr_model.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._asr_model.decode_stream(stream)
        asr_text = stream.result.text.strip()

        # 计算相似度（优先使用拼音相似度，解决同音字问题）
        if self.config.use_pinyin_match:
            similarity = self._pinyin_similarity(asr_text, self.config.keyword_text)
        else:
            similarity = self._text_similarity(asr_text, self.config.keyword_text)

        # 判断
        is_valid = False
        if similarity >= self.config.similarity_threshold:
            is_valid = True
        elif self.config.use_fuzzy_match and self._fuzzy_match(
            asr_text, self.config.keyword_text
        ):
            is_valid = True

        return is_valid, asr_text, similarity


def create_verifier(
    keyword_text: str,
    asr_model: Optional[str] = None,
    similarity_threshold: float = 0.8,
    use_pinyin_match: bool = True,
) -> KeywordVerifier:
    """创建唤醒词验证器的便捷函数。"""
    config = VerifierConfig(
        keyword_text=keyword_text,
        asr_model=asr_model,
        similarity_threshold=similarity_threshold,
        use_pinyin_match=use_pinyin_match,
    )
    return KeywordVerifier(config)
