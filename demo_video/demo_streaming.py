#!/usr/bin/env python3
"""
流式演示脚本 - 模拟实时麦克风输入
从 WAV 文件读取音频，按流式方式逐块处理，展示实时识别效果
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
import threading
import unicodedata
from queue import Queue, Empty


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


def print_realtime_result(result, simulated_time):
    """打印实时识别结果"""
    kind = result.get("kind", "unknown")
    sv_score = result.get("sv_score", 0)
    text = result.get("text", "")
    stream_id = result.get("stream", "?")
    duration = result.get("duration", 0)

    # 图标和类型标识
    if kind == "vad_separated":
        icon = "🎤"
        kind_str = f"VAD#{stream_id}"
        color = "\033[96m"  # 青色
    elif kind == "separated":
        icon = "🔊"
        kind_str = f"分离#{stream_id}"
        color = "\033[94m"  # 蓝色
    elif kind == "clean":
        icon = "✓"
        kind_str = "无重叠"
        color = "\033[92m"  # 绿色
    elif kind == "overlap":
        icon = "⚡"
        kind_str = "OSD检测"
        color = "\033[93m"  # 黄色
    elif kind == "full_separation":
        icon = "🔄"
        kind_str = "全分离"
        color = "\033[94m"  # 蓝色
    else:
        icon = "•"
        kind_str = kind
        color = "\033[0m"

    # SV分数指示
    if sv_score >= 0.8:
        sv_indicator = "🟢"
    elif sv_score >= 0.6:
        sv_indicator = "🟡"
    else:
        sv_indicator = "🔴"

    reset = "\033[0m"
    text_display = text[:50] + "..." if len(text) > 50 else text

    print(
        f"  [{simulated_time:>6.2f}s] {icon} {color}[{kind_str:^8}]{reset} {sv_indicator} SV={sv_score:.3f}"
    )
    print(f"            📝 {text_display}")


def create_progress_bar(current, total, width=40):
    """创建进度条"""
    progress = current / total
    filled = int(width * progress)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {progress*100:>5.1f}%"


class StreamingSimulator:
    """流式处理模拟器 - 从WAV文件模拟实时流式输入"""

    def __init__(
        self, pipeline, audio_data, sample_rate, chunk_size=1024, process_seconds=2.0
    ):
        self.pipeline = pipeline
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.process_seconds = process_seconds

        # 计算参数
        self.total_samples = len(audio_data)
        self.total_duration = self.total_samples / sample_rate
        self.frames_per_process = int(sample_rate * process_seconds / chunk_size)

        # 状态
        self.current_position = 0
        self.simulated_time = 0.0
        self.results = []
        self.running = False

    def run(self, realtime_display=True):
        """运行流式模拟处理"""
        self.running = True
        audio_buffer = []
        chunk_count = 0

        print()
        print("╔" + "═" * 68 + "╗")
        print("║                    🎙️ 流式处理中...                              ║")
        print("╚" + "═" * 68 + "╝")
        print()

        process_start = time.time()
        last_display_count = 0

        while self.current_position < self.total_samples and self.running:
            # 读取一个 chunk
            end_pos = min(self.current_position + self.chunk_size, self.total_samples)
            audio_chunk = self.audio_data[self.current_position : end_pos]
            self.current_position = end_pos

            audio_buffer.append(audio_chunk)
            chunk_count += 1

            # 更新模拟时间
            self.simulated_time = self.current_position / self.sample_rate

            # 累积够 process_seconds 后处理
            if len(audio_buffer) >= self.frames_per_process:
                combined_audio = np.concatenate(audio_buffer)

                # 显示进度
                progress_bar = create_progress_bar(
                    self.current_position, self.total_samples
                )
                print(f"  ⏳ 处理分块 [{self.simulated_time:.1f}s] {progress_bar}")

                # 发送到 pipeline
                self.pipeline.add_audio_data(combined_audio)
                audio_buffer = []

                # 等待处理完成
                time.sleep(0.3)

                # 获取并显示结果
                if realtime_display:
                    new_count = self._display_new_results()
                    if new_count > 0:
                        print()  # 结果后空行

        # 处理剩余数据
        if audio_buffer:
            combined_audio = np.concatenate(audio_buffer)
            print(f"  ⏳ 处理最后分块 [{self.simulated_time:.1f}s]")
            self.pipeline.add_audio_data(combined_audio)

        print()
        print("  ✅ 音频处理完成！等待识别结果...")

        # 等待所有异步处理完成
        time.sleep(1.0)

        # 收集并显示所有剩余结果
        final_count = (
            self._collect_and_display_results()
            if realtime_display
            else self._collect_all_results()
        )
        if final_count > 0:
            print(f"  📥 共收到 {final_count} 个延迟结果")

        process_time = time.time() - process_start
        return process_time

    def _display_new_results(self):
        """显示新的识别结果，返回新结果数量"""
        count = 0
        while not self.pipeline.results_queue.empty():
            try:
                result = self.pipeline.results_queue.get_nowait()
                self.results.append(result)
                print_realtime_result(result, self.simulated_time)
                count += 1
            except Empty:
                break
        return count

    def _collect_all_results(self):
        """收集所有剩余结果，返回收集数量"""
        count = 0
        while not self.pipeline.results_queue.empty():
            try:
                result = self.pipeline.results_queue.get_nowait()
                self.results.append(result)
                count += 1
            except Empty:
                break
        return count

    def _collect_and_display_results(self):
        """收集并显示所有剩余结果，返回收集数量"""
        count = 0
        while not self.pipeline.results_queue.empty():
            try:
                result = self.pipeline.results_queue.get_nowait()
                self.results.append(result)
                print_realtime_result(result, self.simulated_time)
                count += 1
            except Empty:
                break
        return count


def main():
    parser = argparse.ArgumentParser(description="流式声纹识别演示")
    parser.add_argument("--mix-wav", required=True, help="混合音频文件")
    parser.add_argument("--target-wav", required=True, help="目标说话人音频")
    parser.add_argument("--spk-embed-model", required=True, help="说话人嵌入模型")
    parser.add_argument("--sense-voice", required=True, help="ASR模型")
    parser.add_argument("--tokens", required=True, help="tokens文件")
    parser.add_argument("--provider", default="cuda", help="推理设备")
    parser.add_argument("--sv-threshold", type=float, default=0.4, help="SV阈值")
    parser.add_argument("--chunk-size", type=int, default=1024, help="音频块大小")
    parser.add_argument(
        "--process-seconds", type=float, default=3.0, help="每次处理的秒数"
    )
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式，显示被过滤的结果"
    )
    parser.add_argument(
        "--vad-model", default="", help="VAD 模型路径（启用 VAD 分段模式）"
    )
    parser.add_argument(
        "--vad-min-silence", type=float, default=0.25, help="VAD 最小静音时长"
    )
    parser.add_argument(
        "--vad-min-speech", type=float, default=0.25, help="VAD 最小语音时长"
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=3.0,
        help="VAD 模式下最大分段时长（秒）",
    )
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
    pipeline_args.sample_rate = 16000
    pipeline_args.chunk_size = args.chunk_size
    pipeline_args.process_seconds = args.process_seconds
    # VAD 相关参数
    pipeline_args.vad_model = args.vad_model
    pipeline_args.vad_min_silence = args.vad_min_silence
    pipeline_args.vad_min_speech = args.vad_min_speech
    pipeline_args.max_segment_duration = args.max_segment_duration

    # 判断使用哪种模式
    use_vad_mode = bool(args.vad_model)

    print()
    print("╔" + "═" * 68 + "╗")
    print("║       🎤 多说话人重叠语音分离与识别系统 - 流式演示              ║")
    print("╚" + "═" * 68 + "╝")
    print()

    # 获取音频信息
    mix_path = Path(args.mix_wav)
    target_path = Path(args.target_wav)

    info = torchaudio.info(str(mix_path))
    duration = info.num_frames / info.sample_rate

    mode_str = "VAD 分段模式" if use_vad_mode else "固定间隔模式"
    print_box(
        "📊 演示信息",
        [
            f"混合音频: {mix_path.name}",
            f"音频时长: {duration:.2f} 秒",
            f"目标说话人: {target_path.name}",
            f"处理模式: {mode_str}",
            f"流式块大小: {args.chunk_size} samples",
        ]
        + ([f"处理间隔: {args.process_seconds} 秒"] if not use_vad_mode else []),
    )
    print()

    if use_vad_mode:
        print_box(
            "🔧 流式处理流程 (VAD 版)",
            [
                "1. 音频分块输入 (模拟麦克风)",
                "2. VAD 语音检测 (Silero VAD)",
                "3. 3源分离 (Conv-TasNet)",
                "4. 说话人验证 (3DSpeaker)",
                "5. 语音识别 (SenseVoice)",
                "※ 基于语音边界分段，保持语义完整性",
            ],
        )
    else:
        print_box(
            "🔧 流式处理流程 (优化版)",
            [
                "1. 音频分块输入 (模拟麦克风)",
                "2. 3源直接分离 (Conv-TasNet)",
                "3. 说话人验证 (3DSpeaker)",
                "4. 语音识别 (SenseVoice)",
                "5. 实时输出匹配结果",
                "※ 无OSD依赖，避免级联失败",
            ],
        )
    print()

    print("⏳ 正在加载模型...")
    start_time = time.time()

    import warnings

    warnings.filterwarnings("ignore")

    # 根据模式选择 pipeline
    if use_vad_mode:
        from vad_streaming_overlap3_core import VADStreamingOverlap3Pipeline

        pipeline = VADStreamingOverlap3Pipeline(pipeline_args, str(target_path))
    else:
        from optimized_streaming_overlap3_core import OptimizedStreamingOverlap3Pipeline

        pipeline = OptimizedStreamingOverlap3Pipeline(pipeline_args, str(target_path))

    load_time = time.time() - start_time
    print(f"✓ 模型加载完成 ({load_time:.2f}s)")
    print()

    # 显示目标说话人信息
    print_box(
        "🎯 目标说话人",
        [
            f"文件: {target_path.name}",
            f"原文: {pipeline.target_src_text[:60]}{'...' if len(pipeline.target_src_text) > 60 else ''}",
        ],
    )
    print()

    # 加载混合音频
    audio_data, file_sr = torchaudio.load(str(mix_path))
    if audio_data.shape[0] > 1:
        audio_data = audio_data.mean(dim=0)
    else:
        audio_data = audio_data.squeeze(0)

    if file_sr != 16000:
        audio_data = torchaudio.functional.resample(audio_data, file_sr, 16000)

    audio_np = audio_data.numpy()

    # 创建流式模拟器并运行
    simulator = StreamingSimulator(
        pipeline=pipeline,
        audio_data=audio_np,
        sample_rate=16000,
        chunk_size=args.chunk_size,
        process_seconds=args.process_seconds,
    )

    process_time = simulator.run(realtime_display=True)

    # 过滤匹配的结果
    matched_results = [
        r for r in simulator.results if r.get("sv_score", 0) >= args.sv_threshold
    ]

    # 调试模式：显示被过滤的结果
    if args.debug:
        filtered_results = [
            r for r in simulator.results if r.get("sv_score", 0) < args.sv_threshold
        ]
        if filtered_results:
            print("╔" + "═" * 68 + "╗")
            print(
                "║              🔍 调试信息 - 被过滤的结果（SV分数过低）              ║"
            )
            print("╚" + "═" * 68 + "╝")
            print()
            for r in filtered_results:
                sv_score = r.get("sv_score", 0)
                text = r.get("text", "")
                stream_id = r.get("stream", "?")
                print(
                    f"  [分离#{stream_id}] SV={sv_score:.3f} (阈值: {args.sv_threshold}) "
                    f"→ 📝 {text[:50]}{'...' if len(text) > 50 else ''}"
                )
            print()

    # 显示最终统计
    print("╔" + "═" * 68 + "╗")
    print("║                       📈 处理统计                               ║")
    print("╚" + "═" * 68 + "╝")
    print()

    rtf = process_time / duration if duration > 0 else 0
    sv_scores = [r.get("sv_score", 0) for r in matched_results if r.get("sv_score")]
    avg_sv = np.mean(sv_scores) if sv_scores else 0
    max_sv = max(sv_scores) if sv_scores else 0

    # 按类型统计
    kind_counts = {}
    for r in matched_results:
        k = r.get("kind", "unknown")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    stats_lines = [
        f"总识别分段: {len(matched_results)}",
        f"处理耗时: {process_time:.2f} 秒",
        f"音频时长: {duration:.2f} 秒",
        f"RTF: {rtf:.3f}x ({1/rtf:.1f}x 实时速度)" if rtf > 0 else "RTF: N/A",
        "",
        f"平均 SV 分数: {avg_sv:.3f}",
        f"最高 SV 分数: {max_sv:.3f}",
    ]

    # 调试模式：显示过滤统计
    total_results = len(simulator.results)
    filtered_count = total_results - len(matched_results)
    if args.debug and filtered_count > 0:
        stats_lines.insert(
            5,
            f"被过滤分段: {filtered_count} (SV分数 < {args.sv_threshold})",
        )

    for k, v in kind_counts.items():
        if k == "vad_separated":
            stats_lines.append(f"VAD分离分段: {v}")
        elif k == "separated":
            stats_lines.append(f"分离匹配分段: {v}")
        elif k == "clean":
            stats_lines.append(f"无重叠分段: {v}")
        elif k == "overlap":
            stats_lines.append(f"OSD检测分段: {v}")
        elif k == "full_separation":
            stats_lines.append(f"全分离分段: {v}")

    for line in stats_lines:
        print(f"  {line}")
    print()

    # 显示完整识别文本拼接
    if matched_results:
        # 按序列号排序（如果有的话），然后按时间排序
        sorted_results = sorted(
            matched_results, key=lambda r: (r.get("seq_id", 0), r.get("start", 0))
        )

        # 直接拼接文本（不使用后处理器）
        full_text = "".join([r.get("text", "") for r in sorted_results])

        print("╔" + "═" * 68 + "╗")
        print("║                    📄 完整识别文本                             ║")
        print("╚" + "═" * 68 + "╝")
        print()

        # 处理长文本的换行显示
        max_width = 66
        if len(full_text) <= max_width:
            print(f"  {full_text}")
        else:
            # 按字符分行显示
            for i in range(0, len(full_text), max_width):
                chunk = full_text[i : i + max_width]
                print(f"  {chunk}")
        print()

        # 如果启用调试模式，显示目标文本对比
        if args.debug:
            print("╔" + "═" * 68 + "╗")
            print("║                  📊 文本对比分析                             ║")
            print("╚" + "═" * 68 + "╝")
            print()
            print(f"  🎯 目标: {pipeline.target_src_text}")
            print(f"  🔍 识别: {full_text}")
            print()
    else:
        print("╔" + "═" * 68 + "╗")
        print("║                    📄 完整识别文本                             ║")
        print("╚" + "═" * 68 + "╝")
        print()
        print("  （无匹配结果）")
        print()

    print("╔" + "═" * 68 + "╗")
    print("║                      ✅ 流式演示完成！                          ║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
