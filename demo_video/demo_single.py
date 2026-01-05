#!/usr/bin/env python3
"""
单样本演示脚本 - 用于视频录制
直接处理一个混合音频文件，识别目标说话人
"""
import sys
import time
from pathlib import Path

# 添加路径
SCRIPT_DIR = Path(__file__).parent.parent
OSD_DIR = SCRIPT_DIR / "scripts" / "osd"
SRC_DIR = SCRIPT_DIR / "src"
for p in [str(OSD_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import numpy as np
import torchaudio
import torch
import unicodedata


def get_display_width(s):
    """计算字符串的显示宽度（中文字符占2个宽度）"""
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ("F", "W", "A"):
            width += 2
        else:
            width += 1
    return width


def pad_to_width(s, target_width):
    """将字符串填充到指定显示宽度"""
    current_width = get_display_width(s)
    padding = target_width - current_width
    if padding > 0:
        return s + " " * padding
    return s


def print_box(title, content_lines, style="single"):
    """打印带边框的信息框"""
    width = 68
    if style == "double":
        top = "╔" + "═" * width + "╗"
        bottom = "╚" + "═" * width + "╝"
        side = "║"
    else:
        top = "┌" + "─" * width + "┐"
        bottom = "└" + "─" * width + "┘"
        side = "│"

    print(top)
    if title:
        print(f"{side} {pad_to_width(title, width - 2)} {side}")
        print("├" + "─" * width + "┤")
    for line in content_lines:
        print(f"{side} {pad_to_width(line, width - 2)} {side}")
    print(bottom)


def print_result(kind, sv_score, text, start=None, end=None):
    """打印单条识别结果"""
    # 根据类型选择图标和颜色
    if kind == "clean":
        icon = "✓"
        kind_str = "无重叠"
    elif kind == "overlap":
        icon = "⚡"
        kind_str = "OSD检测"
    elif kind == "full_separation":
        icon = "🔄"
        kind_str = "全分离"
    else:
        icon = "•"
        kind_str = kind

    # SV分数颜色指示
    if sv_score >= 0.8:
        sv_indicator = "🟢"
    elif sv_score >= 0.6:
        sv_indicator = "🟡"
    else:
        sv_indicator = "🔴"

    time_str = f"[{start:.2f}s-{end:.2f}s]" if start is not None else ""
    text_display = text[:45] + "..." if len(text) > 45 else text

    print(f"  {icon} [{kind_str:^8}] {sv_indicator} SV={sv_score:.3f} {time_str}")
    print(f"     📝 {text_display}")
    print()


def main():
    parser = argparse.ArgumentParser(description="声纹识别演示")
    parser.add_argument("--mix-wav", required=True, help="混合音频文件")
    parser.add_argument("--target-wav", required=True, help="目标说话人音频")
    parser.add_argument("--spk-embed-model", required=True, help="说话人嵌入模型")
    parser.add_argument("--sense-voice", required=True, help="ASR模型")
    parser.add_argument("--tokens", required=True, help="tokens文件")
    parser.add_argument("--provider", default="cuda", help="推理设备")
    parser.add_argument("--sv-threshold", type=float, default=0.4, help="SV阈值")
    args = parser.parse_args()

    # 创建参数对象
    class Args:
        pass

    pipeline_args = Args()
    pipeline_args.spk_embed_model = args.spk_embed_model
    pipeline_args.sense_voice = args.sense_voice
    pipeline_args.tokens = args.tokens
    pipeline_args.provider = args.provider
    pipeline_args.sv_threshold = args.sv_threshold
    pipeline_args.paraformer = ""
    pipeline_args.encoder = ""
    pipeline_args.decoder = ""
    pipeline_args.joiner = ""
    pipeline_args.decoding_method = "greedy_search"
    pipeline_args.feature_dim = 80
    pipeline_args.language = "auto"
    pipeline_args.num_threads = 2
    pipeline_args.osd_backend = "pyannote"
    pipeline_args.sep_backend = "asteroid"
    pipeline_args.sep_checkpoint = ""
    pipeline_args.osd_thr = 0.5
    pipeline_args.osd_win = 0.5
    pipeline_args.osd_hop = 0.25
    pipeline_args.min_overlap_dur = 0.2
    pipeline_args.exclusive_segments = True

    print()
    print("╔" + "═" * 68 + "╗")
    print("║          🎤 多说话人重叠语音分离与识别系统 演示                  ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 获取音频信息
    mix_path = Path(args.mix_wav)
    target_path = Path(args.target_wav)

    info = torchaudio.info(str(mix_path))
    duration = info.num_frames / info.sample_rate

    print_box(
        "📊 演示信息",
        [
            f"混合音频: {mix_path.name}",
            f"音频时长: {duration:.2f} 秒",
            f"目标说话人: {target_path.name}",
            f"采样率: {info.sample_rate} Hz",
        ],
    )
    print()

    print_box(
        "🔧 技术栈",
        [
            "• OSD 检测: pyannote.audio",
            "• 语音分离: Asteroid Conv-TasNet (3源)",
            "• 说话人验证: 3DSpeaker (ONNX)",
            "• 语音识别: SenseVoice (sherpa-onnx)",
            f"• 硬件加速: {args.provider.upper()}",
        ],
    )
    print()

    print("⏳ 正在加载模型...")
    start_time = time.time()

    # 导入并初始化pipeline
    from overlap3_core import Overlap3Pipeline, G_SAMPLE_RATE

    # 静默加载模型的警告
    import warnings

    warnings.filterwarnings("ignore")

    # 创建pipeline（使用文件模式）
    pipeline_args.input_wavs = [str(mix_path)]
    pipeline_args.target_wav = str(target_path)
    pipeline_args.librimix_root = ""
    pipeline_args.subset = ""
    pipeline_args.sample_rate = 16000
    pipeline_args.task = "sep_clean"
    pipeline_args.mode = "min"
    pipeline_args.max_files = 0
    pipeline_args.seed = -1
    pipeline_args.refs_csv = ""
    pipeline_args.ref_wavs = None
    pipeline_args.eval_separation = False
    pipeline_args.out_dir = ""
    pipeline_args.enable_metrics = False
    pipeline_args.monitor_interval = 0.5
    pipeline_args.metrics_out = ""
    pipeline_args.save_sep_details = False
    pipeline_args.sep_details_out = ""

    pipeline = Overlap3Pipeline(pipeline_args)
    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 ({load_time:.2f}s)")
    print()

    # 获取目标说话人原文
    target_wav, target_sr = torchaudio.load(str(target_path))
    if target_sr != G_SAMPLE_RATE:
        target_wav = torchaudio.functional.resample(
            target_wav, target_sr, G_SAMPLE_RATE
        )
    target_np = target_wav.squeeze().numpy()
    if target_np.ndim > 1:
        target_np = target_np[0]

    st = pipeline.asr.create_stream()
    st.accept_waveform(G_SAMPLE_RATE, target_np)
    pipeline.asr.decode_stream(st)
    target_text = st.result.text or ""

    print_box("🎯 目标说话人原文", [target_text[:65]], style="single")
    print()

    print("⏳ 正在处理音频...")
    process_start = time.time()

    # 运行处理
    result = pipeline.run()

    process_time = time.time() - process_start
    print(f"✓ 处理完成 ({process_time:.2f}s)")
    print()

    # 显示结果
    segments = result.segments
    matched_segments = [
        s for s in segments if s.get("sv_score", 0) >= args.sv_threshold
    ]

    print("╔" + "═" * 68 + "╗")
    print("║                       ✨ 识别结果                               ║")
    print("╚" + "═" * 68 + "╝")
    print()

    if matched_segments:
        for seg in matched_segments:
            print_result(
                seg.get("kind", "unknown"),
                seg.get("sv_score", 0),
                seg.get("text", ""),
                seg.get("start"),
                seg.get("end"),
            )
    else:
        print("  ⚠️  未检测到匹配的目标说话人语音片段")
        print()

    # 统计信息
    rtf = process_time / duration if duration > 0 else 0
    sv_scores = [s.get("sv_score", 0) for s in matched_segments if s.get("sv_score")]
    avg_sv = np.mean(sv_scores) if sv_scores else 0
    max_sv = max(sv_scores) if sv_scores else 0

    kind_counts = {}
    for s in matched_segments:
        k = s.get("kind", "unknown")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    print("╔" + "═" * 68 + "╗")
    print("║                       📈 性能统计                               ║")
    print("╚" + "═" * 68 + "╝")
    print()

    stats_lines = [
        f"识别分段数: {len(matched_segments)}",
        f"处理耗时: {process_time:.2f} 秒",
        f"音频时长: {duration:.2f} 秒",
        f"RTF: {rtf:.3f}x ({1/rtf:.1f}x 实时速度)" if rtf > 0 else "RTF: N/A",
        "",
        f"平均 SV 分数: {avg_sv:.3f}",
        f"最高 SV 分数: {max_sv:.3f}",
    ]

    for k, v in kind_counts.items():
        if k == "clean":
            stats_lines.append(f"无重叠分段: {v}")
        elif k == "overlap":
            stats_lines.append(f"OSD检测分段: {v}")
        elif k == "full_separation":
            stats_lines.append(f"全分离分段: {v}")

    for line in stats_lines:
        print(f"  {line}")
    print()

    print("╔" + "═" * 68 + "╗")
    print("║                      ✅ 演示完成！                              ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
