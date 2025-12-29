#!/usr/bin/env python3
"""KWS 参数优化脚本。

网格搜索 boosting score 和 trigger threshold 的最优组合，
并绘制热力图可视化结果。
"""

from __future__ import annotations

import json
import sys
import wave
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "DejaVu Sans",
    "Arial Unicode MS",
    "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

# 添加 src 到路径
_THIS_DIR = Path(__file__).parent
_ROOT_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_ROOT_DIR))

from src.detection.model import create_kws_model


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV 文件。"""
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr


def test_config(
    model_dir: str,
    boost: float,
    thresh: float,
    positive_samples: list,
    homophone_samples: list,
) -> tuple[int, int, float]:
    """测试单个配置。"""
    kw = f"n ǐ h ǎo zh ēn zh ēn :{boost} #{thresh} @你好真真"
    kws = create_kws_model(
        model_dir,
        keywords=kw,
        use_int8=True,
        epoch=12,
        avg=2,
        keywords_threshold=0.25,
    )

    # FA 测试
    fa = 0
    for s in homophone_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            fa += 1

    # TP 测试
    tp = 0
    for s in positive_samples:
        samples, sr = read_wav(s["file"])
        detections, _ = kws.detect(samples, sr)
        if detections:
            tp += 1

    frr = (len(positive_samples) - tp) / len(positive_samples) * 100
    return fa, tp, frr


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KWS 参数优化")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT_DIR / "test/detection"),
        help="输出目录",
    )
    args = parser.parse_args()

    model_dir = str(
        _ROOT_DIR / "models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    )

    # 加载测试数据
    with open(_ROOT_DIR / "dataset/kws_test_data/metadata.json", "r") as f:
        meta = json.load(f)

    homophones = ["你好镇镇", "李浩真真", "泥豪真真", "你好珍珍"]
    homophone_samples = [
        s for s in meta["negative_samples"] if s.get("phrase") in homophones
    ]
    positive_samples = meta["positive_samples"]

    print(f"正样本: {len(positive_samples)}, 谐音负样本: {len(homophone_samples)}")
    print()

    # 定义搜索空间
    boost_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
    thresh_values = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    # 存储结果
    fa_matrix = np.zeros((len(boost_values), len(thresh_values)))
    frr_matrix = np.zeros((len(boost_values), len(thresh_values)))
    results = []

    total = len(boost_values) * len(thresh_values)
    count = 0

    print("开始网格搜索...")
    print(f"{'Boost':<8} | {'Thresh':<8} | {'FA_谐音':<8} | {'TP':<8} | {'FRR(%)':<8}")
    print("-" * 50)

    for i, boost in enumerate(boost_values):
        for j, thresh in enumerate(thresh_values):
            count += 1
            fa, tp, frr = test_config(
                model_dir, boost, thresh, positive_samples, homophone_samples
            )
            fa_matrix[i, j] = fa
            frr_matrix[i, j] = frr
            results.append(
                {
                    "boost": boost,
                    "threshold": thresh,
                    "fa_homophone": fa,
                    "tp": tp,
                    "frr": frr,
                }
            )
            print(f"{boost:<8.2f} | {thresh:<8.2f} | {fa:<8} | {tp:<8} | {frr:<8.1f}")

    print()

    # 找最优解（目标：最小化 FA_谐音，约束 FRR < 10%）
    print("=" * 60)
    print("最优解搜索（约束: FRR < 10%）")
    print("=" * 60)

    valid_results = [r for r in results if r["frr"] < 10.0]
    if valid_results:
        best = min(valid_results, key=lambda x: (x["fa_homophone"], x["frr"]))
        print(f"最优配置: boost={best['boost']}, threshold={best['threshold']}")
        print(f"  FA_谐音 = {best['fa_homophone']}/{len(homophone_samples)}")
        print(f"  FRR = {best['frr']:.1f}%")
    else:
        print("无法找到 FRR < 10% 的配置")

    # 放宽约束
    print()
    print("=" * 60)
    print("最优解搜索（约束: FRR < 15%）")
    print("=" * 60)

    valid_results = [r for r in results if r["frr"] < 15.0]
    if valid_results:
        best = min(valid_results, key=lambda x: (x["fa_homophone"], x["frr"]))
        print(f"最优配置: boost={best['boost']}, threshold={best['threshold']}")
        print(f"  FA_谐音 = {best['fa_homophone']}/{len(homophone_samples)}")
        print(f"  FRR = {best['frr']:.1f}%")

    # 绘制热力图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. FA 热力图
    ax1 = axes[0]
    im1 = ax1.imshow(fa_matrix, cmap="Reds", aspect="auto")
    ax1.set_xticks(range(len(thresh_values)))
    ax1.set_xticklabels([f"{t:.2f}" for t in thresh_values])
    ax1.set_yticks(range(len(boost_values)))
    ax1.set_yticklabels([f"{b:.2f}" for b in boost_values])
    ax1.set_xlabel("Trigger Threshold (#)")
    ax1.set_ylabel("Boosting Score (:)")
    ax1.set_title("FA_homophone (lower is better)")
    plt.colorbar(im1, ax=ax1)

    # 在每个格子显示数值
    for i in range(len(boost_values)):
        for j in range(len(thresh_values)):
            ax1.text(
                j,
                i,
                f"{int(fa_matrix[i, j])}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    # 2. FRR 热力图
    ax2 = axes[1]
    im2 = ax2.imshow(frr_matrix, cmap="Blues", aspect="auto")
    ax2.set_xticks(range(len(thresh_values)))
    ax2.set_xticklabels([f"{t:.2f}" for t in thresh_values])
    ax2.set_yticks(range(len(boost_values)))
    ax2.set_yticklabels([f"{b:.2f}" for b in boost_values])
    ax2.set_xlabel("Trigger Threshold (#)")
    ax2.set_ylabel("Boosting Score (:)")
    ax2.set_title("FRR% (lower is better)")
    plt.colorbar(im2, ax=ax2)

    for i in range(len(boost_values)):
        for j in range(len(thresh_values)):
            ax2.text(
                j,
                i,
                f"{frr_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    # 3. 综合得分（FA + FRR 加权）
    # 归一化后加权：FA 权重 0.6，FRR 权重 0.4
    fa_norm = fa_matrix / fa_matrix.max() if fa_matrix.max() > 0 else fa_matrix
    frr_norm = frr_matrix / frr_matrix.max() if frr_matrix.max() > 0 else frr_matrix
    score_matrix = 0.6 * fa_norm + 0.4 * frr_norm

    ax3 = axes[2]
    im3 = ax3.imshow(score_matrix, cmap="RdYlGn_r", aspect="auto")
    ax3.set_xticks(range(len(thresh_values)))
    ax3.set_xticklabels([f"{t:.2f}" for t in thresh_values])
    ax3.set_yticks(range(len(boost_values)))
    ax3.set_yticklabels([f"{b:.2f}" for b in boost_values])
    ax3.set_xlabel("Trigger Threshold (#)")
    ax3.set_ylabel("Boosting Score (:)")
    ax3.set_title("Combined Score (0.6*FA + 0.4*FRR, lower is better)")
    plt.colorbar(im3, ax=ax3)

    for i in range(len(boost_values)):
        for j in range(len(thresh_values)):
            ax3.text(
                j,
                i,
                f"{score_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    plt.tight_layout()

    # 保存图表
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = output_dir / f"param_optimization_{timestamp}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存到: {fig_path}")

    # 保存 JSON 结果
    json_path = output_dir / f"param_optimization_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "boost_values": boost_values,
                "thresh_values": thresh_values,
                "results": results,
                "best_frr10": next(
                    (
                        r
                        for r in sorted(
                            results, key=lambda x: (x["fa_homophone"], x["frr"])
                        )
                        if r["frr"] < 10
                    ),
                    None,
                ),
                "best_frr15": next(
                    (
                        r
                        for r in sorted(
                            results, key=lambda x: (x["fa_homophone"], x["frr"])
                        )
                        if r["frr"] < 15
                    ),
                    None,
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"结果已保存到: {json_path}")

    plt.show()


if __name__ == "__main__":
    main()
