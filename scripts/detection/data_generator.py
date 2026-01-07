#!/usr/bin/env python3
"""合成数据生成器 - 使用 edge-tts 生成唤醒词测试样本。

功能:
1. 使用多种中文语音生成唤醒词样本
2. 支持不同语速、音调变化
3. 自动混合背景噪声（多种 SNR）
4. 生成正样本（含唤醒词）和负样本（不含唤醒词）

使用示例:
    # 生成默认的"你好真真"样本
    python data_generator.py --output-dir ./generated_data

    # 自定义唤醒词和数量
    python data_generator.py --keyword "你好真真" --num-samples 100 --output-dir ./data

    # 指定噪声目录
    python data_generator.py --noise-dir /path/to/noise --snr-range 5,10,15,20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import shutil
import sys
import tempfile
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# edge-tts 是异步的
try:
    import edge_tts
except ImportError:
    print("请安装 edge-tts: pip install edge-tts")
    sys.exit(1)


# ============================================================================
# 配置
# ============================================================================

# 中文语音配置
CHINESE_VOICES = [
    # 普通话女声
    {"name": "zh-CN-XiaoxiaoNeural", "gender": "female", "style": "warm"},
    {"name": "zh-CN-XiaoyiNeural", "gender": "female", "style": "lively"},
    # 普通话男声
    {"name": "zh-CN-YunjianNeural", "gender": "male", "style": "passion"},
    {"name": "zh-CN-YunxiNeural", "gender": "male", "style": "sunshine"},
    {"name": "zh-CN-YunxiaNeural", "gender": "male", "style": "cute"},
    {"name": "zh-CN-YunyangNeural", "gender": "male", "style": "professional"},
    # 方言
    {"name": "zh-CN-liaoning-XiaobeiNeural", "gender": "female", "style": "dialect"},
    {"name": "zh-CN-shaanxi-XiaoniNeural", "gender": "female", "style": "dialect"},
]

# 语速变化范围 (相对于正常速度的百分比)
RATE_VARIATIONS = ["-30%", "-20%", "-10%", "+0%", "+10%", "+20%", "+30%"]

# 音调变化范围 (Hz)
PITCH_VARIATIONS = ["-10Hz", "-5Hz", "+0Hz", "+5Hz", "+10Hz"]

# 默认 SNR 范围 (dB)
DEFAULT_SNR_RANGE = [5, 10, 15, 20, 30]

# 负样本短语分类
NEGATIVE_PHRASES_NORMAL = [
    "今天天气真好",
    "你好啊",
    "早上好",
    "晚安",
    "谢谢你",
    "没问题",
    "好的",
    "再见",
    "请问一下",
    "不客气",
    "对不起",
    "没关系",
    "真的吗",
    "是的",
    "不是",
    "我知道了",
    "请稍等",
    "好久不见",
    "最近怎么样",
    "在干嘛呢",
    "你好真好",  # 相似但不同
]

# 谐音负样本（应阻断）
# 注意：
# - 一声同音字（珍/甄/臻）与目标词音频完全相同，KWS 无法区分，无需测试
# - 同声调同韵母的字发音相同（如镇/阵/振/震都是zhèn），只保留一个代表词
NEGATIVE_PHRASES_HOMOPHONE = [
    # 声调变体 - zhen 系列（每个声调保留一个代表）
    "你好镇镇",  # zhèn (第四声) - 代表：镇/阵/振/震
    "你好诊诊",  # zhěn (第三声) - 代表：诊/枕
    # 声调变体 - zheng 系列
    "你好正正",  # zhèng (第四声)
    "你好争争",  # zhēng (第一声) - 代表：争/征
    "你好整整",  # zhěng (第三声)
    # 声母变体（r/c/z 与 zh 的区分）
    "你好认认",  # rèn (r vs zh)
    "你好曾曾",  # céng (c vs zh)
    "你好怎怎",  # zěn (z vs zh)
    # 声母变体（首字变化）
    "李浩真真",  # lǐ háo (l vs n)
    "泥豪真真",  # ní háo (n vs n)
]

# 合并（兼容旧代码）
NEGATIVE_PHRASES = NEGATIVE_PHRASES_NORMAL + NEGATIVE_PHRASES_HOMOPHONE


@dataclass
class GeneratorConfig:
    """生成器配置。"""

    keyword: str = "你好真真"
    output_dir: str = "./generated_data"
    num_positive: int = 50  # 正样本数量
    num_negative: int = 50  # 负样本数量
    noise_dir: Optional[str] = None  # 噪声目录
    snr_range: List[int] = field(default_factory=lambda: DEFAULT_SNR_RANGE.copy())
    sample_rate: int = 16000
    include_clean: bool = True  # 是否包含无噪声版本
    voices: List[str] = field(default_factory=list)  # 空表示使用所有语音
    seed: int = 42
    homophone_only: bool = False  # 是否只生成谐音词


# ============================================================================
# 音频处理工具
# ============================================================================


def read_wav(wav_path: str) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()
        data = wf.readframes(num_frames)

    if sample_width == 2:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if num_channels > 1:
        samples = samples[::num_channels]

    return samples, sample_rate


def write_wav(wav_path: str, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """写入 WAV 文件。"""
    # 归一化到 [-1, 1]
    max_val = np.max(np.abs(samples))
    if max_val > 1.0:
        samples = samples / max_val

    # 转换为 int16
    samples_int16 = (samples * 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())


def resample(samples: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """简单重采样。"""
    if src_sr == target_sr:
        return samples
    duration = len(samples) / src_sr
    num_samples = int(duration * target_sr)
    indices = np.linspace(0, len(samples) - 1, num_samples)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def mix_with_noise(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """以指定 SNR 混合信号和噪声。

    Args:
        signal: 干净信号
        noise: 噪声信号
        snr_db: 信噪比 (dB)

    Returns:
        混合后的信号
    """
    # 调整噪声长度
    if len(noise) < len(signal):
        # 循环填充噪声
        repeats = (len(signal) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[: len(signal)]

    # 计算功率
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return signal

    # 计算目标噪声功率
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)

    # 混合
    mixed = signal + noise * noise_scale

    return mixed


def generate_white_noise(length: int, amplitude: float = 0.1) -> np.ndarray:
    """生成白噪声。"""
    return np.random.randn(length).astype(np.float32) * amplitude


def generate_pink_noise(length: int, amplitude: float = 0.1) -> np.ndarray:
    """生成粉红噪声（1/f 噪声）。"""
    # 使用 Voss-McCartney 算法的简化版本
    white = np.random.randn(length).astype(np.float32)

    # 简单的低通滤波近似粉红噪声
    pink = np.zeros(length, dtype=np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    # 简单的 IIR 滤波
    for i in range(len(white)):
        pink[i] = white[i] * b[0]
        for j in range(1, min(i + 1, len(b))):
            pink[i] += white[i - j] * b[j]
        for j in range(1, min(i + 1, len(a))):
            pink[i] -= pink[i - j] * a[j]

    # 归一化
    pink = pink / (np.max(np.abs(pink)) + 1e-8) * amplitude

    return pink


# ============================================================================
# TTS 生成
# ============================================================================


async def generate_tts_async(
    text: str,
    voice: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bool:
    """异步生成 TTS 音频。

    Args:
        text: 要合成的文本
        voice: 语音名称
        output_path: 输出文件路径
        rate: 语速调整
        pitch: 音调调整

    Returns:
        是否成功
    """
    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await communicate.save(output_path)
        return True
    except Exception as e:
        print(f"TTS 生成失败: {e}")
        return False


def generate_tts(
    text: str,
    voice: str,
    output_path: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> bool:
    """同步包装的 TTS 生成。"""
    return asyncio.run(generate_tts_async(text, voice, output_path, rate, pitch))


# ============================================================================
# 数据生成器
# ============================================================================


class DataGenerator:
    """唤醒词数据生成器。"""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.positive_dir = self.output_dir / "positive"
        self.negative_dir = self.output_dir / "negative"
        self.noise_samples: List[np.ndarray] = []

        # 设置随机种子
        random.seed(config.seed)
        np.random.seed(config.seed)

        # 选择语音
        if config.voices:
            self.voices = [v for v in CHINESE_VOICES if v["name"] in config.voices]
        else:
            self.voices = CHINESE_VOICES

    def setup(self) -> None:
        """初始化输出目录。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.positive_dir.mkdir(exist_ok=True)
        self.negative_dir.mkdir(exist_ok=True)

        # 加载噪声
        self._load_noise()

    def _load_noise(self) -> None:
        """加载噪声样本。"""
        if self.config.noise_dir:
            noise_dir = Path(self.config.noise_dir)
            if noise_dir.exists():
                for wav_path in noise_dir.glob("*.wav"):
                    try:
                        samples, sr = read_wav(str(wav_path))
                        samples = resample(samples, sr, self.config.sample_rate)
                        self.noise_samples.append(samples)
                        print(f"加载噪声: {wav_path.name}")
                    except Exception as e:
                        print(f"加载噪声失败 {wav_path}: {e}")

        # 如果没有外部噪声，生成合成噪声
        if not self.noise_samples:
            print("未找到外部噪声，生成合成噪声...")
            duration_sec = 10
            num_samples = duration_sec * self.config.sample_rate

            # 白噪声
            self.noise_samples.append(generate_white_noise(num_samples))
            # 粉红噪声
            self.noise_samples.append(generate_pink_noise(num_samples))

    def _get_random_noise(self) -> np.ndarray:
        """获取随机噪声样本。"""
        return random.choice(self.noise_samples)

    def _generate_sample(
        self,
        text: str,
        voice: dict,
        rate: str,
        pitch: str,
        output_path: Path,
    ) -> Optional[np.ndarray]:
        """生成单个 TTS 样本。"""
        # 使用临时文件（edge-tts 输出 mp3）
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name

        try:
            # 生成 TTS
            success = generate_tts(text, voice["name"], tmp_mp3, rate, pitch)
            if not success:
                return None

            # 使用 miniaudio 解码 MP3（不需要 ffmpeg）
            samples = self._decode_mp3_to_samples(tmp_mp3)
            return samples

        finally:
            if os.path.exists(tmp_mp3):
                os.unlink(tmp_mp3)

    def _decode_mp3_to_samples(self, mp3_path: str) -> Optional[np.ndarray]:
        """使用 miniaudio 解码 MP3 文件。"""
        try:
            import miniaudio

            # 解码 MP3 到 PCM
            decoded = miniaudio.decode_file(
                mp3_path,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=self.config.sample_rate,
            )

            # 转换为 numpy float32
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0

            return samples
        except Exception as e:
            print(f"MP3 解码失败: {e}")
            return None

    def generate_positive_samples(self) -> List[Dict[str, Any]]:
        """生成正样本（含唤醒词）。"""
        print(f"\n生成正样本: {self.config.num_positive} 个")
        print(f"唤醒词: {self.config.keyword}")

        samples_info: List[Dict[str, Any]] = []
        sample_idx = 0

        # 计算每个语音需要生成的样本数
        samples_per_voice = max(1, self.config.num_positive // len(self.voices))

        for voice in self.voices:
            for _ in range(samples_per_voice):
                if sample_idx >= self.config.num_positive:
                    break

                # 随机选择语速和音调
                rate = random.choice(RATE_VARIATIONS)
                pitch = random.choice(PITCH_VARIATIONS)

                # 生成基础样本
                base_name = f"positive_{sample_idx:04d}_{voice['name']}_{rate}_{pitch}"
                base_name = (
                    base_name.replace("+", "p").replace("-", "m").replace("%", "pct")
                )

                samples = self._generate_sample(
                    self.config.keyword,
                    voice,
                    rate,
                    pitch,
                    self.positive_dir / f"{base_name}.wav",
                )

                if samples is None:
                    continue

                # 保存干净版本
                if self.config.include_clean:
                    clean_path = self.positive_dir / f"{base_name}_clean.wav"
                    write_wav(str(clean_path), samples, self.config.sample_rate)
                    samples_info.append(
                        {
                            "file": str(clean_path),
                            "type": "positive",
                            "keyword": self.config.keyword,
                            "voice": voice["name"],
                            "rate": rate,
                            "pitch": pitch,
                            "snr": "inf",
                            "noise_type": "none",
                        }
                    )

                # 添加噪声版本
                for snr in self.config.snr_range:
                    noise = self._get_random_noise()
                    noisy = mix_with_noise(samples, noise, snr)

                    noisy_path = self.positive_dir / f"{base_name}_snr{snr}dB.wav"
                    write_wav(str(noisy_path), noisy, self.config.sample_rate)

                    samples_info.append(
                        {
                            "file": str(noisy_path),
                            "type": "positive",
                            "keyword": self.config.keyword,
                            "voice": voice["name"],
                            "rate": rate,
                            "pitch": pitch,
                            "snr": snr,
                            "noise_type": "mixed",
                        }
                    )

                sample_idx += 1
                print(
                    f"  [{sample_idx}/{self.config.num_positive}] {voice['name']} rate={rate}"
                )

        return samples_info

    def generate_negative_samples(self) -> List[Dict[str, Any]]:
        """生成负样本（不含唤醒词）。"""
        print(f"\n生成负样本: {self.config.num_negative} 个")

        # 如果是 homophone_only 模式，使用固定谐音词列表并均匀分布
        if self.config.homophone_only:
            negative_phrases = NEGATIVE_PHRASES_HOMOPHONE
            print(f"谐音词模式：为 {len(negative_phrases)} 个谐音词各生成样本")
            # 均匀分布：每个谐音词生成相同数量的样本
            samples_per_phrase = self.config.num_negative // len(negative_phrases)
            phrase_list = negative_phrases * samples_per_phrase
            # 补齐不足的样本
            remaining = self.config.num_negative - len(phrase_list)
            phrase_list.extend(random.sample(negative_phrases, remaining))
            random.shuffle(phrase_list)
        else:
            negative_phrases = NEGATIVE_PHRASES
            phrase_list = None  # 随机模式

        samples_info: List[Dict[str, Any]] = []
        sample_idx = 0

        while sample_idx < self.config.num_negative:
            # 随机选择语音和短语
            voice = random.choice(self.voices)
            if phrase_list is not None:
                phrase = phrase_list[sample_idx]  # 均匀分布模式
            else:
                phrase = random.choice(negative_phrases)  # 随机模式
            rate = random.choice(RATE_VARIATIONS)
            pitch = random.choice(PITCH_VARIATIONS)

            base_name = f"negative_{sample_idx:04d}_{voice['name']}"
            base_name = (
                base_name.replace("+", "p").replace("-", "m").replace("%", "pct")
            )

            samples = self._generate_sample(
                phrase,
                voice,
                rate,
                pitch,
                self.negative_dir / f"{base_name}.wav",
            )

            if samples is None:
                continue

            # 保存干净版本
            if self.config.include_clean:
                clean_path = self.negative_dir / f"{base_name}_clean.wav"
                write_wav(str(clean_path), samples, self.config.sample_rate)
                samples_info.append(
                    {
                        "file": str(clean_path),
                        "type": "negative",
                        "phrase": phrase,
                        "voice": voice["name"],
                        "rate": rate,
                        "pitch": pitch,
                        "snr": "inf",
                        "noise_type": "none",
                    }
                )

            # 添加噪声版本
            for snr in self.config.snr_range:
                noise = self._get_random_noise()
                noisy = mix_with_noise(samples, noise, snr)

                noisy_path = self.negative_dir / f"{base_name}_snr{snr}dB.wav"
                write_wav(str(noisy_path), noisy, self.config.sample_rate)

                samples_info.append(
                    {
                        "file": str(noisy_path),
                        "type": "negative",
                        "phrase": phrase,
                        "voice": voice["name"],
                        "rate": rate,
                        "pitch": pitch,
                        "snr": snr,
                        "noise_type": "mixed",
                    }
                )

            sample_idx += 1
            print(f"  [{sample_idx}/{self.config.num_negative}] {phrase[:10]}...")

        return samples_info

    def generate(self) -> Dict[str, Any]:
        """生成所有样本。"""
        self.setup()

        positive_info = self.generate_positive_samples()
        negative_info = self.generate_negative_samples()

        # 汇总信息
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "keyword": self.config.keyword,
                "num_positive": self.config.num_positive,
                "num_negative": self.config.num_negative,
                "snr_range": self.config.snr_range,
                "sample_rate": self.config.sample_rate,
                "voices": [v["name"] for v in self.voices],
            },
            "statistics": {
                "total_positive_files": len(positive_info),
                "total_negative_files": len(negative_info),
                "voices_used": len(self.voices),
            },
            "positive_samples": positive_info,
            "negative_samples": negative_info,
        }

        # 保存元数据
        import json

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n生成完成!")
        print(f"  正样本: {len(positive_info)} 个文件")
        print(f"  负样本: {len(negative_info)} 个文件")
        print(f"  输出目录: {self.output_dir}")
        print(f"  元数据: {metadata_path}")

        return summary


# ============================================================================
# 命令行接口
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合成数据生成器 - 生成唤醒词测试样本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default="你好真真",
        help="唤醒词文本",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_data",
        help="输出目录",
    )
    parser.add_argument(
        "--num-positive",
        type=int,
        default=50,
        help="正样本数量（每个会生成多个噪声版本）",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=50,
        help="负样本数量",
    )
    parser.add_argument(
        "--noise-dir",
        type=str,
        default="",
        help="噪声 WAV 文件目录（可选，默认使用合成噪声）",
    )
    parser.add_argument(
        "--snr-range",
        type=str,
        default="5,10,15,20,30",
        help="SNR 范围，逗号分隔",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="输出采样率",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="不生成无噪声版本",
    )
    parser.add_argument(
        "--voices",
        type=str,
        default="",
        help="指定语音，逗号分隔（默认使用所有）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="列出可用语音",
    )
    parser.add_argument(
        "--homophone-only",
        action="store_true",
        help="只生成谐音词负样本（用于扩充测试数据）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_voices:
        print("可用的中文语音:")
        for v in CHINESE_VOICES:
            print(f"  {v['name']:<40} {v['gender']:<8} {v['style']}")
        return

    # 解析 SNR 范围
    snr_range = [int(x.strip()) for x in args.snr_range.split(",")]

    # 解析语音列表
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]

    # 如果只生成谐音词，覆盖负样本短语列表
    if args.homophone_only:
        print(f"\n只生成谐音词测试数据（共 {len(NEGATIVE_PHRASES_HOMOPHONE)} 个短语）")
        for i, phrase in enumerate(NEGATIVE_PHRASES_HOMOPHONE, 1):
            print(f"  {i}. {phrase}")
        # 使用固定的谐音词列表
        config = GeneratorConfig(
            keyword=args.keyword,
            output_dir=args.output_dir,
            num_positive=0,  # 不生成正样本
            num_negative=len(NEGATIVE_PHRASES_HOMOPHONE)
            * 12,  # 每个谐音词生成 12 个样本
            noise_dir=args.noise_dir or None,
            snr_range=snr_range,
            sample_rate=args.sample_rate,
            include_clean=not args.no_clean,
            voices=voices,
            seed=args.seed,
            homophone_only=True,
        )
    else:
        config = GeneratorConfig(
            keyword=args.keyword,
            output_dir=args.output_dir,
            num_positive=args.num_positive,
            num_negative=args.num_negative,
            noise_dir=args.noise_dir or None,
            snr_range=snr_range,
            sample_rate=args.sample_rate,
            include_clean=not args.no_clean,
            voices=voices,
            seed=args.seed,
        )

    generator = DataGenerator(config)
    generator.generate()


if __name__ == "__main__":
    main()
