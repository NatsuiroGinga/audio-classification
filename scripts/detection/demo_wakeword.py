#!/usr/bin/env python
"""
唤醒词检测演示脚本 - "你好真真"

使用最佳配置的 KWS 模型展示唤醒词检测效果。
配置: wenetspeech-3.3M int8 + 低漏报优先 (:1.5 #0.45, FRR=1.4%)

用法:
    python demo_wakeword.py <audio_file.wav>
    python demo_wakeword.py --config <config_name> <audio_file.wav>

配置选项:
    default      - 默认配置 :1.0 #0.25 (FRR=2.1%, FA_homophone=32)
    low-frr      - 低漏报优先 :1.5 #0.45 (FRR=1.4%, FA_homophone=29) [默认]
    balanced     - 平衡方案 :0.7 #0.60 (FRR=9.7%, FA_homophone=20)
    zero-fa      - 零误报优先 :0.3 #0.40 (FRR=13.9%, FA_homophone=0)
"""

import argparse
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

# 配置映射
CONFIGS = {
    "default": {
        "boost": 1.0,
        "threshold": 0.25,
        "desc": "默认配置",
        "frr": 2.1,
        "fa_homophone": 32,
    },
    "low-frr": {
        "boost": 1.5,
        "threshold": 0.45,
        "desc": "低漏报优先",
        "frr": 1.4,
        "fa_homophone": 29,
    },
    "balanced": {
        "boost": 0.7,
        "threshold": 0.60,
        "desc": "平衡方案",
        "frr": 9.7,
        "fa_homophone": 20,
    },
    "zero-fa": {
        "boost": 0.3,
        "threshold": 0.40,
        "desc": "零误报优先",
        "frr": 13.9,
        "fa_homophone": 0,
    },
}


def read_wav(wav_path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 音频文件。

    Args:
        wav_path: WAV 文件路径

    Returns:
        (samples, sample_rate): 音频样本和采样率
    """
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        num_frames = wf.getnframes()

        # 读取音频数据
        audio_data = wf.readframes(num_frames)

        # 转换为 numpy 数组
        if sample_width == 2:
            samples = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:
            samples = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # 转换为 float32 [-1, 1]
        samples = samples.astype(np.float32)
        if sample_width == 2:
            samples /= 32768.0
        elif sample_width == 4:
            samples /= 2147483648.0

        # 如果是多声道，取第一个声道
        if num_channels > 1:
            samples = samples[::num_channels]

        return samples, sample_rate


def create_kws_model(
    model_dir: str, boost: float, threshold: float
) -> sherpa_onnx.KeywordSpotter:
    """创建 KWS 模型。

    Args:
        model_dir: 模型目录路径
        boost: Boosting score（越高越容易触发）
        threshold: Trigger threshold（越高越难触发）

    Returns:
        KeywordSpotter 实例
    """
    import os
    import tempfile

    # 创建临时 keywords 文件
    keyword = f"n ǐ h ǎo zh ēn zh ēn :{boost} #{threshold} @你好真真"
    fd, kw_file = tempfile.mkstemp(suffix=".txt", prefix="kws_demo_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(keyword + "\n")

    # 创建 KWS 模型
    kws = sherpa_onnx.KeywordSpotter(
        tokens=f"{model_dir}/tokens.txt",
        encoder=f"{model_dir}/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        decoder=f"{model_dir}/decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        joiner=f"{model_dir}/joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
        keywords_file=kw_file,
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        max_active_paths=4,
        keywords_score=1.0,
        keywords_threshold=0.25,
        num_trailing_blanks=1,
        provider="cpu",
    )

    # 清理临时文件
    os.unlink(kw_file)

    return kws


def detect_keyword(
    kws: sherpa_onnx.KeywordSpotter, samples: np.ndarray, sample_rate: int
) -> dict | None:
    """检测唤醒词。

    Args:
        kws: KeywordSpotter 实例
        samples: 音频样本
        sample_rate: 采样率

    Returns:
        检测结果字典，如果未检测到则返回 None
    """
    # 获取底层对象
    low_kws = kws.keyword_spotter

    # 创建流
    stream = low_kws.create_stream()

    # 喂入音频数据
    stream.accept_waveform(sample_rate, samples)

    # 添加尾部填充（0.5 秒）
    tail_padding = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    stream.accept_waveform(sample_rate, tail_padding)

    # 解码
    decode_count = 0
    while low_kws.is_ready(stream):
        low_kws.decode_stream(stream)
        decode_count += 1

        # 获取结果（注意：只能调用一次，第二次调用结果会被清空）
        result = low_kws.get_result(stream)

        if result.keyword:
            return {
                "keyword": result.keyword,
                "tokens": list(result.tokens),
                "timestamps": list(result.timestamps),
                "decode_count": decode_count,
            }

    return None


def format_time(seconds: float) -> str:
    """格式化时间（秒）。

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串
    """
    return f"{seconds:.2f}s"


def print_result(
    audio_file: str,
    audio_duration: float,
    detection_time: float,
    result: dict | None,
    config: dict,
):
    """打印检测结果。

    Args:
        audio_file: 音频文件路径
        audio_duration: 音频时长（秒）
        detection_time: 检测耗时（秒）
        result: 检测结果字典
        config: 配置字典
    """
    print()
    print("=" * 100)
    print("唤醒词检测结果 - 你好真真")
    print("=" * 100)
    print()

    # 音频信息
    print(f"音频文件:  {audio_file}")
    print(f"音频时长:  {format_time(audio_duration)}")
    print()

    # 配置信息
    print(
        f"配置方案:  {config['desc']} (boost={config['boost']}, threshold={config['threshold']})"
    )
    print(f"性能指标:  FRR={config['frr']}%, 谐音误报={config['fa_homophone']}/36")
    print()

    # 检测结果
    if result:
        print("检测状态:  ✓ 检测到唤醒词")
        print(f"关键词:    {result['keyword']}")
        print()

        # Tokens 信息
        print("Token 序列:")
        tokens_str = "  "
        for i, token in enumerate(result["tokens"]):
            tokens_str += f"[{token}]"
            if (i + 1) % 8 == 0 and i + 1 < len(result["tokens"]):
                tokens_str += "\n  "
        print(tokens_str)
        print()

        # 时间戳信息
        print("时间戳 (秒):")
        timestamps_str = "  "
        for i, (token, ts) in enumerate(zip(result["tokens"], result["timestamps"])):
            timestamps_str += f"{token}:{ts:.2f}  "
            if (i + 1) % 4 == 0 and i + 1 < len(result["tokens"]):
                timestamps_str += "\n  "
        print(timestamps_str)
        print()

        # 检测信息
        print(f"解码次数:  {result['decode_count']}")
        print(f"检测耗时:  {format_time(detection_time)}")
        print(f"实时率:    {detection_time / audio_duration:.3f}x (RTF)")
    else:
        print("检测状态:  ✗ 未检测到唤醒词")
        print()
        print(f"解码耗时:  {format_time(detection_time)}")
        print(f"实时率:    {detection_time / audio_duration:.3f}x (RTF)")

    print()
    print("=" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="唤醒词检测演示 - 你好真真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置选项说明:
  default   - 默认配置 (FRR=2.1%, 谐音误报=32/36)
  low-frr   - 低漏报优先 (FRR=1.4%, 谐音误报=29/36) [推荐]
  balanced  - 平衡方案 (FRR=9.7%, 谐音误报=20/36)
  zero-fa   - 零误报优先 (FRR=13.9%, 谐音误报=0/36)

示例:
  # 使用默认配置（低漏报优先）
  python demo_wakeword.py test.wav

  # 使用零误报优先配置
  python demo_wakeword.py --config zero-fa test.wav

  # 从测试集中选择样本
  python demo_wakeword.py ../../dataset/kws_test_data/positive/positive_0000_*_clean.wav
        """,
    )
    parser.add_argument("audio_file", help="音频文件路径 (.wav)")
    parser.add_argument(
        "--config",
        "-c",
        choices=list(CONFIGS.keys()),
        default="low-frr",
        help="配置方案（默认: low-frr）",
    )
    parser.add_argument(
        "--model-dir",
        default="../../models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
        help="模型目录路径",
    )

    args = parser.parse_args()

    # 检查文件是否存在
    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        print(f"错误: 音频文件不存在: {audio_file}")
        sys.exit(1)

    # 获取配置
    config = CONFIGS[args.config]

    # 检查模型目录
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)

    print()
    print("正在加载模型...")
    try:
        kws = create_kws_model(str(model_dir), config["boost"], config["threshold"])
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        sys.exit(1)

    print("正在读取音频...")
    try:
        samples, sample_rate = read_wav(str(audio_file))
        audio_duration = len(samples) / sample_rate
    except Exception as e:
        print(f"错误: 读取音频失败: {e}")
        sys.exit(1)

    print("正在检测唤醒词...")
    start_time = time.perf_counter()
    result = detect_keyword(kws, samples, sample_rate)
    detection_time = time.perf_counter() - start_time

    # 打印结果
    print_result(str(audio_file), audio_duration, detection_time, result, config)


if __name__ == "__main__":
    main()
