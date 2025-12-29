#!/usr/bin/env python3
"""KWS 快速测试脚本 - 测试唤醒词 "你好真真"。

本脚本对比已下载的两个模型在 "你好真真" 唤醒词上的表现：
1. sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 (中文)
2. sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20 (中英文)

使用示例:
    python test_nihao_zhenzhen.py --wav /path/to/test.wav
    python test_nihao_zhenzhen.py --generate-test  # 生成测试音频（需要 TTS）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 添加项目根目录到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import (
    KEYWORD_NIHAO_ZHENZHEN,
    KeywordSpotterModel,
    KWSModelConfig,
    create_kws_model,
)

# 模型目录配置
MODELS_DIR = _ROOT_DIR / "models"

MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "wenetspeech-3.3M": {
        "dir": str(
            MODELS_DIR / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        ),
        "description": "WenetSpeech 3.3M 中文唤醒词模型",
    },
    "zh-en-3M": {
        "dir": str(MODELS_DIR / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"),
        "description": "中英文 3M 唤醒词模型",
    },
}


def read_wave_file(wav_path: str) -> Tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    import wave

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


def resample_if_needed(
    samples: np.ndarray, src_sr: int, target_sr: int = 16000
) -> np.ndarray:
    """重采样到目标采样率。"""
    if src_sr == target_sr:
        return samples
    duration = len(samples) / src_sr
    num_samples = int(duration * target_sr)
    indices = np.linspace(0, len(samples) - 1, num_samples)
    return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)


def test_single_model(
    model_name: str,
    samples: np.ndarray,
    sample_rate: int = 16000,
    keyword: str = KEYWORD_NIHAO_ZHENZHEN,
    keywords_threshold: float = 0.25,
    use_int8: bool = False,
    provider: str = "cpu",
) -> Dict[str, any]:
    """测试单个模型。

    Args:
        model_name: 模型名称（wenetspeech-3.3M 或 zh-en-3M）
        samples: 音频样本
        sample_rate: 采样率
        keyword: 唤醒词
        keywords_threshold: 阈值
        use_int8: 是否使用 INT8
        provider: 推理后端

    Returns:
        测试结果字典
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Unknown model: {model_name}")

    model_dir = config["dir"]
    if not Path(model_dir).exists():
        return {
            "model": model_name,
            "error": f"Model directory not found: {model_dir}",
        }

    print(f"\n{'='*50}")
    print(f"测试模型: {model_name}")
    print(f"描述: {config['description']}")
    print(f"唤醒词: {keyword}")
    print(f"阈值: {keywords_threshold}")
    print(f"{'='*50}")

    try:
        model = create_kws_model(
            model_dir=model_dir,
            keywords=keyword,
            provider=provider,
            num_threads=2,
            keywords_threshold=keywords_threshold,
            use_int8=use_int8,
        )

        audio_duration = len(samples) / sample_rate
        detections, process_time = model.detect(samples, sample_rate, keyword)

        rtf = process_time / audio_duration if audio_duration > 0 else 0

        result = {
            "model": model_name,
            "success": True,
            "audio_duration_sec": round(audio_duration, 4),
            "process_time_sec": round(process_time, 4),
            "rtf": round(rtf, 6),
            "num_detections": len(detections),
            "detections": [
                {
                    "keyword": d.keyword,
                    "start_time": round(d.start_time, 4),
                }
                for d in detections
            ],
        }

        if detections:
            print(f"✓ 检测到 {len(detections)} 次唤醒")
            for i, d in enumerate(detections):
                print(f"  [{i+1}] '{d.keyword}' @ {d.start_time:.2f}s")
                print(f"detection result: {d}")
        else:
            print("✗ 未检测到唤醒词")

        print(f"\n处理时间: {process_time:.4f}s")
        print(f"音频时长: {audio_duration:.4f}s")
        print(f"RTF: {rtf:.6f}")

        return result

    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return {
            "model": model_name,
            "error": str(e),
        }


def generate_test_audio(output_path: str, keyword_text: str = "你好真真") -> bool:
    """生成测试音频（使用 TTS）。

    注意: 需要安装 edge-tts 或其他 TTS 库。

    Args:
        output_path: 输出文件路径
        keyword_text: 唤醒词文本

    Returns:
        是否成功
    """
    try:
        import subprocess

        # 使用 edge-tts 生成
        cmd = [
            "edge-tts",
            "--voice",
            "zh-CN-XiaoxiaoNeural",
            "--text",
            keyword_text,
            "--write-media",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"已生成测试音频: {output_path}")
        return True
    except FileNotFoundError:
        print("需要安装 edge-tts: pip install edge-tts")
        return False
    except Exception as e:
        print(f"生成测试音频失败: {e}")
        return False


def main() -> None:
    """主入口。"""
    parser = argparse.ArgumentParser(
        description="测试 '你好真真' 唤醒词",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wav",
        type=str,
        default="",
        help="测试音频文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "wenetspeech-3.3M", "zh-en-3M"],
        help="测试的模型",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=KEYWORD_NIHAO_ZHENZHEN,
        help="唤醒词",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="唤醒阈值",
    )
    parser.add_argument(
        "--use-int8",
        action="store_true",
        help="使用 INT8 模型",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理后端",
    )
    parser.add_argument(
        "--generate-test",
        action="store_true",
        help="生成测试音频（需要 edge-tts）",
    )

    args = parser.parse_args()

    # 生成测试音频
    if args.generate_test:
        test_audio_path = str(_THIS_DIR / "test_nihao_zhenzhen.wav")
        if generate_test_audio(test_audio_path, "你好真真"):
            args.wav = test_audio_path
        else:
            return

    # 检查音频文件
    if not args.wav:
        print("请提供测试音频文件路径 (--wav) 或使用 --generate-test 生成")
        print("\n可用模型:")
        for name, config in MODEL_CONFIGS.items():
            exists = "✓" if Path(config["dir"]).exists() else "✗"
            print(f"  [{exists}] {name}: {config['description']}")
        return

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"音频文件不存在: {args.wav}")
        return

    # 读取音频
    print(f"读取音频: {args.wav}")
    samples, sr = read_wave_file(args.wav)
    samples = resample_if_needed(samples, sr, 16000)
    print(f"音频时长: {len(samples) / 16000:.2f}s")

    # 测试模型
    models_to_test = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    results: List[Dict[str, any]] = []
    for model_name in models_to_test:
        result = test_single_model(
            model_name=model_name,
            samples=samples,
            sample_rate=16000,
            keyword=args.keyword,
            keywords_threshold=args.threshold,
            use_int8=args.use_int8,
            provider=args.provider,
        )
        results.append(result)

    # 对比结果
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("模型对比")
        print("=" * 60)
        print(f"{'模型':<20} {'检测数':<8} {'RTF':<12} {'状态'}")
        print("-" * 60)
        for r in results:
            if "error" in r:
                print(f"{r['model']:<20} {'N/A':<8} {'N/A':<12} ✗ {r['error'][:30]}")
            else:
                status = "✓ 检测到" if r["num_detections"] > 0 else "✗ 未检测"
                print(
                    f"{r['model']:<20} {r['num_detections']:<8} {r['rtf']:<12.6f} {status}"
                )


if __name__ == "__main__":
    main()
