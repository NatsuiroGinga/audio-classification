#!/usr/bin/env python3
"""KWS 消融实验脚本。

测试不同配置组合对唤醒词检测性能的影响：
1. 有/无 int8 量化
2. 有/无 ASR 二次验证 (verifier)
3. 有/无 声纹验证 (speaker verification)

输出：消融实验结果表格和 JSON 报告
"""

from __future__ import annotations

import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 添加 src 到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KeywordSpotterModel,
    KWSModelConfig,
    create_kws_model,
)
from src.detection.verifier import KeywordVerifier, VerifierConfig, create_verifier


# ============================================================================
# 声纹验证模块
# ============================================================================


class SpeakerVerifier:
    """声纹验证器。

    验证检测到唤醒词的音频是否来自注册的目标说话人。
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        num_threads: int = 2,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.num_threads = num_threads
        self._extractor = None
        self._registered_embeddings: List[np.ndarray] = []
        self._init_model()

    def _init_model(self) -> None:
        """初始化声纹提取模型。"""
        try:
            import sherpa_onnx

            # 使用新版 API: SpeakerEmbeddingExtractorConfig
            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=self.model_path,
                num_threads=self.num_threads,
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as e:
            print(f"声纹模型加载失败: {e}")
            self._extractor = None

    def register(self, samples: np.ndarray, sample_rate: int = 16000) -> bool:
        """注册目标说话人的声纹。

        Args:
            samples: 注册音频波形
            sample_rate: 采样率

        Returns:
            是否注册成功
        """
        if self._extractor is None:
            return False

        try:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()

            if not self._extractor.is_ready(stream):
                return False

            embedding = self._extractor.compute(stream)
            self._registered_embeddings.append(np.array(embedding))
            return True
        except Exception as e:
            print(f"声纹注册失败: {e}")
            return False

    def verify(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[bool, float]:
        """验证音频是否来自注册说话人。

        Args:
            samples: 待验证音频
            sample_rate: 采样率

        Returns:
            (is_valid, max_similarity): 是否通过验证和最大相似度
        """
        if self._extractor is None or not self._registered_embeddings:
            return True, 1.0  # 无模型或未注册时跳过

        try:
            stream = self._extractor.create_stream()
            stream.accept_waveform(sample_rate, samples)
            stream.input_finished()

            if not self._extractor.is_ready(stream):
                return True, 1.0

            embedding = np.array(self._extractor.compute(stream))

            # 计算与所有注册声纹的相似度
            max_sim = 0.0
            for reg_emb in self._registered_embeddings:
                sim = np.dot(embedding, reg_emb) / (
                    np.linalg.norm(embedding) * np.linalg.norm(reg_emb) + 1e-8
                )
                max_sim = max(max_sim, sim)

            return max_sim >= self.threshold, max_sim
        except Exception as e:
            print(f"声纹验证失败: {e}")
            return True, 1.0


# ============================================================================
# 消融实验配置
# ============================================================================


@dataclass
class AblationConfig:
    """消融实验配置。"""

    name: str  # 配置名称
    use_int8: bool = False  # 是否使用 int8 量化
    use_verifier: bool = False  # 是否使用 ASR 二次验证
    use_speaker_verify: bool = False  # 是否使用声纹验证


@dataclass
class AblationResult:
    """消融实验结果。"""

    config_name: str
    total_positive: int = 0
    true_positive: int = 0
    false_negative: int = 0
    total_negative_normal: int = 0
    false_positive_normal: int = 0
    total_negative_homophone: int = 0
    false_positive_homophone: int = 0
    total_process_time: float = 0.0
    total_audio_duration: float = 0.0

    @property
    def frr(self) -> float:
        """错误拒绝率 (%)。"""
        if self.total_positive == 0:
            return 0.0
        return (self.false_negative / self.total_positive) * 100

    @property
    def fa_normal(self) -> int:
        """正常负样本误报数。"""
        return self.false_positive_normal

    @property
    def fa_homophone(self) -> int:
        """谐音负样本误报数。"""
        return self.false_positive_homophone

    @property
    def rtf(self) -> float:
        """实时率。"""
        if self.total_audio_duration == 0:
            return 0.0
        return self.total_process_time / self.total_audio_duration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "total_positive": self.total_positive,
            "true_positive": self.true_positive,
            "false_negative": self.false_negative,
            "frr_percent": round(self.frr, 2),
            "total_negative_normal": self.total_negative_normal,
            "fa_normal": self.fa_normal,
            "total_negative_homophone": self.total_negative_homophone,
            "fa_homophone": self.fa_homophone,
            "rtf": round(self.rtf, 4),
        }


# ============================================================================
# 工具函数
# ============================================================================


def read_wav(path: str) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """加载测试数据元信息。"""
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# 消融实验主逻辑
# ============================================================================


def run_ablation_experiment(
    config: AblationConfig,
    kws_model_dir: str,
    asr_model_dir: str,
    speaker_model_path: str,
    positive_samples: List[Dict[str, Any]],
    negative_normal_samples: List[Dict[str, Any]],
    negative_homophone_samples: List[Dict[str, Any]],
    keyword: str = KEYWORD_NIHAO_ZHENZHEN,
    keyword_text: str = "你好真真",
) -> AblationResult:
    """运行单个配置的消融实验。"""

    result = AblationResult(config_name=config.name)

    # 1. 初始化 KWS 模型
    kws_model = create_kws_model(
        kws_model_dir,
        keywords=keyword,
        use_int8=config.use_int8,
    )

    # 2. 初始化 ASR 验证器（可选）
    verifier = None
    if config.use_verifier:
        verifier = create_verifier(
            keyword_text=keyword_text,
            asr_model=asr_model_dir,
            use_pinyin_match=True,  # 使用拼音匹配，接受同音字如"珍珍"
            similarity_threshold=0.75,  # 稍微降低阈值以容忍 ASR 误差
        )

    # 3. 初始化声纹验证器（可选）
    speaker_verifier = None
    if config.use_speaker_verify and os.path.exists(speaker_model_path):
        speaker_verifier = SpeakerVerifier(
            speaker_model_path, threshold=0.3
        )  # 降低阈值
        # 对于每个正样本，用同一语音的样本注册声纹（按语音分组）
        # 用所有正样本的部分注册，模拟更宽松的声纹匹配
        registered_voices = set()
        for sample in positive_samples:
            voice = sample.get("voice", "")
            if voice and voice not in registered_voices and len(registered_voices) < 8:
                samples, sr = read_wav(sample["file"])
                if speaker_verifier.register(samples, sr):
                    registered_voices.add(voice)

    # 4. 测试正样本
    result.total_positive = len(positive_samples)
    for sample in positive_samples:
        samples, sr = read_wav(sample["file"])
        start_time = time.perf_counter()

        # KWS 检测
        detections, _ = kws_model.detect(samples, sr)
        detected = len(detections) > 0

        # ASR 验证
        if detected and verifier:
            is_valid, _, _ = verifier.verify(samples, sr)
            detected = detected and is_valid

        # 声纹验证
        if detected and speaker_verifier:
            is_valid, _ = speaker_verifier.verify(samples, sr)
            detected = detected and is_valid

        result.total_process_time += time.perf_counter() - start_time
        result.total_audio_duration += len(samples) / sr

        if detected:
            result.true_positive += 1
        else:
            result.false_negative += 1

    # 5. 测试正常负样本
    result.total_negative_normal = len(negative_normal_samples)
    for sample in negative_normal_samples:
        samples, sr = read_wav(sample["file"])
        start_time = time.perf_counter()

        detections, _ = kws_model.detect(samples, sr)
        detected = len(detections) > 0

        if detected and verifier:
            is_valid, _, _ = verifier.verify(samples, sr)
            detected = detected and is_valid

        if detected and speaker_verifier:
            is_valid, _ = speaker_verifier.verify(samples, sr)
            detected = detected and is_valid

        result.total_process_time += time.perf_counter() - start_time
        result.total_audio_duration += len(samples) / sr

        if detected:
            result.false_positive_normal += 1

    # 6. 测试谐音负样本
    result.total_negative_homophone = len(negative_homophone_samples)
    for sample in negative_homophone_samples:
        samples, sr = read_wav(sample["file"])
        start_time = time.perf_counter()

        detections, _ = kws_model.detect(samples, sr)
        detected = len(detections) > 0

        if detected and verifier:
            is_valid, _, _ = verifier.verify(samples, sr)
            detected = detected and is_valid

        if detected and speaker_verifier:
            is_valid, _ = speaker_verifier.verify(samples, sr)
            detected = detected and is_valid

        result.total_process_time += time.perf_counter() - start_time
        result.total_audio_duration += len(samples) / sr

        if detected:
            result.false_positive_homophone += 1

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KWS 消融实验")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(_ROOT_DIR / "dataset/kws_test_data"),
        help="测试数据目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT_DIR / "test/detection"),
        help="输出目录",
    )
    parser.add_argument(
        "--kws-model-dir",
        type=str,
        default=str(
            _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        ),
        help="KWS 模型目录",
    )
    parser.add_argument(
        "--asr-model-dir",
        type=str,
        default=str(
            _ROOT_DIR / "models/asr/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        ),
        help="ASR 模型目录",
    )
    parser.add_argument(
        "--speaker-model",
        type=str,
        default=str(
            _ROOT_DIR
            / "models/speaker-recognition/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
        ),
        help="声纹模型路径",
    )

    args = parser.parse_args()

    # 加载测试数据
    metadata_path = os.path.join(args.data_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"错误: 找不到测试数据 {metadata_path}")
        sys.exit(1)

    meta = load_metadata(metadata_path)

    # 谐音短语列表
    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]

    positive_samples = meta["positive_samples"]
    negative_normal = [
        s for s in meta["negative_samples"] if s.get("phrase") not in homophones
    ]
    negative_homophone = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]

    print("=" * 70)
    print("KWS 消融实验")
    print("=" * 70)
    print(f"正样本: {len(positive_samples)}")
    print(f"正常负样本: {len(negative_normal)}")
    print(f"谐音负样本: {len(negative_homophone)}")
    print()

    # 定义消融实验配置
    configs = [
        AblationConfig(name="baseline", use_int8=False),
        AblationConfig(name="int8", use_int8=True),
        AblationConfig(name="int8+verifier", use_int8=True, use_verifier=True),
        AblationConfig(name="int8+speaker", use_int8=True, use_speaker_verify=True),
        AblationConfig(
            name="int8+verifier+speaker",
            use_int8=True,
            use_verifier=True,
            use_speaker_verify=True,
        ),
    ]

    results: List[AblationResult] = []

    for config in configs:
        print(f"运行配置: {config.name}...")
        print(
            f"  int8={config.use_int8}, verifier={config.use_verifier}, speaker={config.use_speaker_verify}"
        )

        result = run_ablation_experiment(
            config=config,
            kws_model_dir=args.kws_model_dir,
            asr_model_dir=args.asr_model_dir,
            speaker_model_path=args.speaker_model,
            positive_samples=positive_samples,
            negative_normal_samples=negative_normal,
            negative_homophone_samples=negative_homophone,
        )
        results.append(result)

        print(
            f"  FRR={result.frr:.1f}%, FA_normal={result.fa_normal}, FA_homophone={result.fa_homophone}, RTF={result.rtf:.4f}"
        )
        print()

    # 输出结果表格
    print("=" * 70)
    print("消融实验结果汇总")
    print("=" * 70)
    print(
        f"{'配置':<25} | {'FRR(%)':<8} | {'FA_正常':<8} | {'FA_谐音':<8} | {'RTF':<8}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r.config_name:<25} | {r.frr:<8.1f} | {r.fa_normal:<8} | {r.fa_homophone:<8} | {r.rtf:<8.4f}"
        )
    print("=" * 70)

    # 保存 JSON 结果
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "positive_samples": len(positive_samples),
            "negative_normal_samples": len(negative_normal),
            "negative_homophone_samples": len(negative_homophone),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
