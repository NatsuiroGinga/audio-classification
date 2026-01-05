#!/usr/bin/env python3
"""
音频波形可视化工具 - 用于演示视频
"""
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 无GUI后端
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import argparse


def plot_waveform(audio_path, output_path, title="Audio Waveform", figsize=(14, 5)):
    """绘制音频波形图"""
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制波形
    times = np.linspace(0, duration, len(y))
    ax.plot(times, y, linewidth=0.5, alpha=0.8, color="#2196F3")
    ax.fill_between(times, y, alpha=0.3, color="#2196F3")

    # 设置标签和标题
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("时间 (秒)", fontsize=12)
    ax.set_ylabel("振幅", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, duration)

    # 添加时长信息
    ax.text(
        0.98,
        0.95,
        f"时长: {duration:.2f}s\n采样率: {sr}Hz",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_comparison(mix_path, target_path, output_path):
    """绘制混合音频与目标说话人音频对比图"""
    print(f"Creating comparison plot...")

    # 加载音频
    y_mix, sr = librosa.load(mix_path, sr=16000)
    y_target, _ = librosa.load(target_path, sr=16000)

    duration_mix = len(y_mix) / sr
    duration_target = len(y_target) / sr

    # 创建子图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 混合音频
    times_mix = np.linspace(0, duration_mix, len(y_mix))
    axes[0].plot(times_mix, y_mix, linewidth=0.5, alpha=0.8, color="#FF5722")
    axes[0].fill_between(times_mix, y_mix, alpha=0.3, color="#FF5722")
    axes[0].set_title("混合音频 (3人同时说话)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("振幅", fontsize=11)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].set_xlim(0, duration_mix)
    axes[0].text(
        0.98,
        0.95,
        f"时长: {duration_mix:.2f}s",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="#FFE0B2", alpha=0.8),
    )

    # 目标说话人音频
    times_target = np.linspace(0, duration_target, len(y_target))
    axes[1].plot(times_target, y_target, linewidth=0.5, alpha=0.8, color="#4CAF50")
    axes[1].fill_between(times_target, y_target, alpha=0.3, color="#4CAF50")
    axes[1].set_title("目标说话人音频", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("时间 (秒)", fontsize=11)
    axes[1].set_ylabel("振幅", fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xlim(0, duration_target)
    axes[1].text(
        0.98,
        0.95,
        f"时长: {duration_target:.2f}s",
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="#C8E6C9", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_spectrogram(audio_path, output_path, title="Spectrogram"):
    """绘制音频频谱图"""
    print(f"Creating spectrogram: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000)

    # 计算梅尔频谱
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(14, 5))
    img = librosa.display.specshow(
        D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="viridis"
    )
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("时间 (秒)", fontsize=12)
    ax.set_ylabel("频率 (Hz)", fontsize=12)

    # 添加颜色条
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.set_label("幅度 (dB)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="音频可视化工具")
    parser.add_argument("--sample", default="s1", help="样本名称")
    parser.add_argument("--data-dir", default="dataset/cn", help="数据目录")
    parser.add_argument("--output-dir", default="demo_video/visuals", help="输出目录")
    parser.add_argument(
        "--type",
        default="all",
        choices=["all", "waveform", "comparison", "spectrogram"],
        help="生成类型",
    )
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = Path(args.data_dir) / args.sample
    mix_path = sample_dir / "mix.wav"
    target_files = sorted([f for f in sample_dir.glob("*.wav") if f.name != "mix.wav"])

    if not target_files:
        print(f"Error: No target audio found in {sample_dir}")
        return

    target_path = target_files[0]  # 使用第一个目标说话人

    print(f"\n{'='*60}")
    print(f"音频可视化 - {args.sample}")
    print(f"{'='*60}")
    print(f"混合音频: {mix_path}")
    print(f"目标音频: {target_path}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")

    if args.type in ["all", "waveform"]:
        # 生成波形图
        plot_waveform(
            mix_path,
            output_dir / f"{args.sample}_mix_waveform.png",
            title=f"{args.sample} - 混合音频波形",
        )
        plot_waveform(
            target_path,
            output_dir / f"{args.sample}_target_waveform.png",
            title=f"{args.sample} - 目标说话人波形",
        )

    if args.type in ["all", "comparison"]:
        # 生成对比图
        plot_comparison(
            mix_path, target_path, output_dir / f"{args.sample}_comparison.png"
        )

    if args.type in ["all", "spectrogram"]:
        # 生成频谱图
        plot_spectrogram(
            mix_path,
            output_dir / f"{args.sample}_mix_spectrogram.png",
            title=f"{args.sample} - 混合音频频谱图",
        )
        plot_spectrogram(
            target_path,
            output_dir / f"{args.sample}_target_spectrogram.png",
            title=f"{args.sample} - 目标说话人频谱图",
        )

    print(f"\n{'='*60}")
    print("可视化完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
