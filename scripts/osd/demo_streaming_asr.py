#!/usr/bin/env python3
"""
流式 ASR 演示脚本

功能：
- 实时显示中间结果（📝 Partial）
- 显示最终结果（✅ Final）
- 支持 VAD 语音边界检测
- 3 源分离 + 说话人验证

用法：
    python demo_streaming_asr.py --input-wav mix.wav --target-wav target.wav [options]
"""
import sys
import os
import time
import argparse

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(script_dir))

import numpy as np
import torchaudio

# ANSI 颜色
RESET = "\033[0m"
BOLD = "\033[1m"
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"


def parse_args():
    p = argparse.ArgumentParser(description="Streaming ASR Demo with Partial/Final")

    # 输入
    p.add_argument("--input-wav", required=True, help="Mixed audio file")
    p.add_argument("--target-wav", required=True, help="Target speaker reference")

    # 模型路径
    p.add_argument("--vad-model", required=True, help="VAD model path")
    p.add_argument("--spk-embed-model", required=True, help="Speaker embedding model")
    p.add_argument("--sense-voice", required=True, help="SenseVoice model dir")
    p.add_argument("--tokens", required=True, help="Tokens file for ASR")

    # 处理参数
    p.add_argument("--provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    p.add_argument(
        "--sv-threshold", type=float, default=0.5, help="Speaker verification threshold"
    )
    p.add_argument(
        "--chunk-duration", type=float, default=0.3, help="Audio chunk duration (s)"
    )
    p.add_argument(
        "--max-segment-duration",
        type=float,
        default=3.0,
        help="Max segment duration (s)",
    )
    p.add_argument(
        "--partial-interval",
        type=float,
        default=0.5,
        help="Partial result interval (s)",
    )
    p.add_argument(
        "--num-threads", type=int, default=2, help="Number of threads for inference"
    )

    # 分离器参数
    p.add_argument("--sep-backend", default="asteroid", help="Separation backend")
    p.add_argument("--sep-checkpoint", default=None, help="Separation checkpoint path")

    # 其他
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    p.add_argument(
        "--inline", action="store_true", help="Use inline update for partial results"
    )

    return p.parse_args()


def format_result(result, use_inline=False):
    """格式化结果显示

    Args:
        result: StreamingResult
        use_inline: 是否使用内联更新（partial 覆盖显示）
    """
    is_final = result.is_final
    result_type_str = "✅ Final" if is_final else "📝 Partial"
    color = GREEN if is_final else YELLOW

    time_str = f"{result.start_time:.2f}s-{result.end_time:.2f}s"
    sv_str = f"SV:{result.sv_score:.3f}" if result.sv_score else ""

    output = (
        f"{color}{result_type_str}{RESET} "
        f"[{CYAN}SEQ#{result.seq_id}{RESET}] "
        f"[{GRAY}{time_str}{RESET}] "
        f"[{BLUE}Stream{result.stream_id}{RESET}] "
        f"{sv_str} "
        f"{BOLD}{result.text}{RESET}"
    )

    if use_inline and not is_final:
        # 使用回车符覆盖当前行（用于实时更新效果）
        return f"\r{output}\033[K"  # \033[K 清除行尾

    return output


def print_result(result, use_inline=False):
    """打印结果"""
    formatted = format_result(result, use_inline)
    if use_inline and not result.is_final:
        print(formatted, end="", flush=True)
    else:
        if use_inline:
            print()  # 换行（因为 partial 可能在同一行）
        print(formatted)


def main():
    args = parse_args()

    print(f"\n{BOLD}=== Streaming ASR Demo (Partial/Final) ==={RESET}\n")

    # 导入 pipeline
    from scripts.osd.streaming_asr_pipeline import StreamingASRPipeline, G_SAMPLE_RATE

    # 加载输入音频
    print(f"Loading audio: {args.input_wav}")
    waveform, sr = torchaudio.load(args.input_wav)

    if sr != G_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, G_SAMPLE_RATE)
        sr = G_SAMPLE_RATE

    audio = waveform[0].numpy()  # 取第一个通道
    audio_duration = len(audio) / sr

    print(f"Audio duration: {audio_duration:.2f}s @ {sr}Hz")
    print(f"Chunk duration: {args.chunk_duration}s")
    print(f"Max segment: {args.max_segment_duration}s")
    print(f"Partial interval: {args.partial_interval}s")
    print()

    # 初始化 pipeline
    print("Initializing pipeline...")
    t0 = time.perf_counter()
    pipeline = StreamingASRPipeline(args, args.target_wav)
    init_time = time.perf_counter() - t0
    print(f"Pipeline initialized in {init_time:.2f}s\n")

    # 模拟流式处理
    print(f"{BOLD}--- Streaming Processing ---{RESET}\n")

    chunk_size = int(args.chunk_duration * sr)
    offset = 0
    total_results = []
    final_results = []
    use_inline = args.inline
    total_compute_time = 0.0  # 只统计核心处理时间

    while offset < len(audio):
        chunk = audio[offset : offset + chunk_size]
        offset += chunk_size

        # 处理音频块（只计时核心处理）
        t_chunk_start = time.perf_counter()
        results = pipeline.add_audio_data(chunk)
        t_chunk_end = time.perf_counter()
        total_compute_time += t_chunk_end - t_chunk_start

        # 显示结果（不计入 RTF）
        for r in results:
            print_result(r, use_inline)
            total_results.append(r)
            if r.is_final:
                final_results.append(r)

    # 刷新剩余数据
    if use_inline:
        print()  # 确保换行
    print(f"\n{GRAY}--- Flushing buffer ---{RESET}\n")

    # flush 也计入核心处理时间
    t_flush_start = time.perf_counter()
    flush_results = pipeline.flush()
    t_flush_end = time.perf_counter()
    total_compute_time += t_flush_end - t_flush_start

    # 显示 flush 结果（不计入 RTF）
    for r in flush_results:
        print_result(r, use_inline)
        total_results.append(r)
        if r.is_final:
            final_results.append(r)

    # 汇总
    print(f"\n{BOLD}=== Summary ==={RESET}\n")
    print(f"Audio duration:    {audio_duration:.2f}s")
    print(f"Compute time:      {total_compute_time:.2f}s (core processing only)")
    print(f"RTF:               {total_compute_time/audio_duration:.3f}x")
    print(f"Total results:     {len(total_results)} (Partial + Final)")
    print(f"Final segments:    {len(final_results)}")

    # 最终识别文本
    print(f"\n{BOLD}--- Final Transcription ---{RESET}\n")

    full_text = ""
    for r in final_results:
        full_text += r.text
        print(f"  [{r.start_time:.2f}-{r.end_time:.2f}s] {r.text}")

    print(f"\n{BOLD}Full text:{RESET} {full_text}")

    # 关闭
    pipeline.shutdown()
    print(f"\n{GREEN}Done!{RESET}\n")


if __name__ == "__main__":
    main()
