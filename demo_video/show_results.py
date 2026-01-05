#!/usr/bin/env python3
"""
演示结果展示工具 - 美化输出用于视频录制
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich import box
import time

console = Console()


def display_header():
    """显示标题"""
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🎤 多说话人重叠语音分离与识别系统[/bold cyan]\n"
            "[dim]Overlapped Speech Separation & Speaker Recognition[/dim]",
            border_style="cyan",
        )
    )
    console.print()


def display_audio_info(sample_name: str, target_file: str, duration: float):
    """显示音频信息"""
    table = Table(title="📊 音频信息", box=box.ROUNDED, show_header=False)
    table.add_column("项目", style="cyan", width=20)
    table.add_column("值", style="yellow")

    table.add_row("样本名称", sample_name)
    table.add_row("目标说话人", target_file)
    table.add_row("音频时长", f"{duration:.2f} 秒")
    table.add_row("混合人数", "3人")

    console.print(table)
    console.print()


def display_target_text(text: str):
    """显示目标说话人文本"""
    console.print(
        Panel(
            f"[bold green]{text}[/bold green]",
            title="🎯 目标说话人原文",
            border_style="green",
        )
    )
    console.print()


def display_processing_results(results: List[Dict]):
    """显示处理结果"""
    if not results:
        console.print("[yellow]⚠️  未检测到目标说话人语音[/yellow]")
        return

    table = Table(title="✨ 识别结果", box=box.DOUBLE_EDGE, show_lines=True)
    table.add_column("序号", justify="center", style="cyan", width=6)
    table.add_column("类型", justify="center", style="magenta", width=18)
    table.add_column("SV分数", justify="center", style="green", width=10)
    table.add_column("时间段", justify="center", style="blue", width=15)
    table.add_column("识别文本", style="yellow")

    for i, result in enumerate(results, 1):
        kind = result.get("kind", "unknown")
        sv_score = result.get("sv_score", 0)
        start = result.get("start", 0)
        end = result.get("end", 0)
        text = result.get("text", "")

        # 根据类型设置图标和颜色
        if kind == "clean":
            kind_display = "✓ 无重叠"
            kind_style = "green"
        elif kind == "overlap":
            kind_display = "⚡ OSD检测"
            kind_style = "yellow"
        elif kind == "full_separation":
            kind_display = "🔄 全分离"
            kind_style = "blue"
        else:
            kind_display = kind
            kind_style = "white"

        # SV分数颜色
        if sv_score >= 0.8:
            sv_style = "bold green"
        elif sv_score >= 0.6:
            sv_style = "bold yellow"
        else:
            sv_style = "bold red"

        table.add_row(
            str(i),
            f"[{kind_style}]{kind_display}[/{kind_style}]",
            f"[{sv_style}]{sv_score:.3f}[/{sv_style}]",
            f"{start:.2f}~{end:.2f}s",
            text[:50] + ("..." if len(text) > 50 else ""),
        )

    console.print(table)
    console.print()


def display_statistics(results: List[Dict], elapsed: float, duration: float):
    """显示统计信息"""
    if not results:
        return

    # 计算统计
    sv_scores = [r.get("sv_score", 0) for r in results]
    avg_sv = sum(sv_scores) / len(sv_scores) if sv_scores else 0
    max_sv = max(sv_scores) if sv_scores else 0
    min_sv = min(sv_scores) if sv_scores else 0

    rtf = elapsed / duration if duration > 0 else 0

    # 按类型统计
    kind_counts = {}
    for r in results:
        kind = r.get("kind", "unknown")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

    # 创建统计表
    table = Table(title="📈 性能统计", box=box.ROUNDED, show_header=False)
    table.add_column("指标", style="cyan", width=25)
    table.add_column("值", style="yellow")

    table.add_row("总分段数", str(len(results)))
    table.add_row("处理耗时", f"{elapsed:.2f} 秒")
    table.add_row("音频时长", f"{duration:.2f} 秒")
    table.add_row("RTF (Real-Time Factor)", f"[bold green]{rtf:.3f}x[/bold green]")
    table.add_row(
        "处理速度", f"[bold cyan]{1/rtf:.1f}x 实时[/bold cyan]" if rtf > 0 else "N/A"
    )
    table.add_row("", "")
    table.add_row("平均 SV 分数", f"{avg_sv:.3f}")
    table.add_row("最高 SV 分数", f"[bold green]{max_sv:.3f}[/bold green]")
    table.add_row("最低 SV 分数", f"{min_sv:.3f}")
    table.add_row("", "")

    for kind, count in kind_counts.items():
        if kind == "clean":
            display_name = "无重叠分段"
        elif kind == "overlap":
            display_name = "OSD检测分段"
        elif kind == "full_separation":
            display_name = "全分离分段"
        else:
            display_name = kind
        table.add_row(display_name, str(count))

    console.print(table)
    console.print()


def load_results(output_dir: Path):
    """加载测试结果"""
    # 查找最新的输出目录
    result_dirs = sorted(output_dir.glob("*"), reverse=True)
    if not result_dirs:
        console.print("[red]错误: 未找到结果目录[/red]")
        return None

    latest_dir = result_dirs[0]

    # 加载结果
    results_file = latest_dir / "batch_results.json"
    segments_file = latest_dir / "all_segments.jsonl"

    if not results_file.exists():
        console.print(f"[red]错误: 未找到结果文件 {results_file}[/red]")
        return None

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 加载所有分段
    segments = []
    if segments_file.exists():
        with open(segments_file, "r", encoding="utf-8") as f:
            for line in f:
                segments.append(json.loads(line))

    return {"data": data, "segments": segments, "dir": latest_dir}


def main():
    parser = argparse.ArgumentParser(description="演示结果展示工具")
    parser.add_argument(
        "--output-dir", default="demo_video/demo_output", help="输出目录路径"
    )
    parser.add_argument("--delay", type=float, default=0.5, help="显示延迟（秒）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # 加载结果
    result = load_results(output_dir)
    if not result:
        return

    data = result["data"]
    segments = result["segments"]

    # 显示标题
    display_header()
    time.sleep(args.delay)

    # 获取测试信息
    test_results = data.get("test_results", [])
    if not test_results:
        console.print("[red]错误: 测试结果为空[/red]")
        return

    test = test_results[0]
    sample = test.get("sample", "unknown")
    target = test.get("target", "unknown")
    elapsed = test.get("elapsed", 0)
    duration = test.get("total_duration", 0)

    # 显示音频信息
    display_audio_info(sample, target, duration)
    time.sleep(args.delay)

    # 显示目标文本（如果有）
    if segments:
        # 从第一个结果推断目标文本（实际应该从别处获取）
        console.print(
            Panel(
                "[dim]系统正在处理音频...[/dim]",
                title="⏳ 处理中",
                border_style="yellow",
            )
        )
        time.sleep(args.delay * 2)

    # 显示处理结果
    display_processing_results(segments)
    time.sleep(args.delay)

    # 显示统计信息
    display_statistics(segments, elapsed, duration)

    # 显示成功消息
    console.print(
        Panel.fit(
            f"[bold green]✓ 演示完成！[/bold green]\n"
            f"[dim]结果已保存至: {result['dir']}[/dim]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
