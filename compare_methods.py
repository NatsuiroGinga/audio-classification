#!/usr/bin/env python3
"""
对比验证脚本：OSD-based vs Direct Separation

用于在现有测试数据上验证两种方法的效果差异。
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_comparison_data(analysis_file: str) -> List[Dict]:
    """从现有对比分析文件中提取数据"""
    data = []

    # 从 comparison_analysis.md 中手动提取的数据（可选自动解析）
    # 为了演示，这里使用预定义的数据
    samples = [
        {
            "id": "s1",
            "reference": "王英汉被枪毙后，部分遗孽深藏起来，几次围捕均未抓获。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.5051,
        },
        {
            "id": "s2",
            "reference": "第二，要把经济手段和恰当的行政手段结合起来加以运用，特别要注意运用好经济手段。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.7169,
        },
        {
            "id": "s3",
            "reference": "散文是颇具魅力的一种文体，因其在语言文采与构思和意境上的讲求，散文又有美文之称。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.7276,
        },
        {
            "id": "s4",
            "reference": "和雪一样，有无数无声的激情，与世界浑然一体，曾经历史养活，又孕育美丽的明天。",
            "osd_stitching": 0.86,
            "osd_audio_cat": 0.86,
            "direct_separation": 0.8605,
        },
        {
            "id": "s5",
            "reference": "他走到屋角脸盆架旁，把脸盆咣当一声扔在一摞脸盆上。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.5718,
        },
        {
            "id": "s6",
            "reference": "此外，与汽车相关的非运输从业人员也达2万多人，年可增收5000万元以上。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.6103,
        },
        {
            "id": "s7",
            "reference": "力生经营的体育用品包罗万象小到游泳耳塞、大致大型体育器械，可以说应有尽有。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.6946,
        },
        {
            "id": "s8",
            "reference": "你要我有了爱情才结婚，可是我就没有啊，也找不到啊，叫我怎么办？",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.6587,
        },
        {
            "id": "s9",
            "reference": "越往森林里走，光色越来越暗，远远传来熊猫等人的相互呼唤，他也呵呵地大声作答。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.5014,
        },
        {
            "id": "s10",
            "reference": "王一汉被枪毙后，部分鱼孽深藏起来，几次围捕，均未抓获。",
            "osd_stitching": 0.0,
            "osd_audio_cat": 0.0,
            "direct_separation": 0.3470,
        },
    ]

    return samples


def compute_metrics(samples: List[Dict]) -> Dict:
    """计算各种指标"""

    metrics = {
        "samples_count": len(samples),
        "osd_method": {
            "mean_score": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "zero_score_count": 0,
            "nonzero_count": 0,
        },
        "direct_method": {
            "mean_score": 0.0,
            "median_score": 0.0,
            "max_score": 0.0,
            "min_score": 0.0,
            "std_score": 0.0,
            "zero_score_count": 0,
            "nonzero_count": 0,
        },
        "comparison": {
            "direct_wins": 0,
            "osd_wins": 0,
            "ties": 0,
            "direct_better_rate": 0.0,
            "average_improvement": 0.0,
            "max_improvement": 0.0,
            "min_improvement": 0.0,
        },
    }

    # 收集分数
    osd_scores = []
    direct_scores = []
    improvements = []

    for sample in samples:
        # 使用平均OSD分数（两种方法都是0）
        osd_score = (sample["osd_stitching"] + sample["osd_audio_cat"]) / 2
        direct_score = sample["direct_separation"]

        osd_scores.append(osd_score)
        direct_scores.append(direct_score)
        improvements.append(direct_score - osd_score)

        # 计数
        if osd_score == 0:
            metrics["osd_method"]["zero_score_count"] += 1
        else:
            metrics["osd_method"]["nonzero_count"] += 1

        if direct_score == 0:
            metrics["direct_method"]["zero_score_count"] += 1
        else:
            metrics["direct_method"]["nonzero_count"] += 1

        # 对比
        if direct_score > osd_score:
            metrics["comparison"]["direct_wins"] += 1
        elif direct_score < osd_score:
            metrics["comparison"]["osd_wins"] += 1
        else:
            metrics["comparison"]["ties"] += 1

    # 计算统计量
    osd_array = np.array(osd_scores)
    direct_array = np.array(direct_scores)
    improvements_array = np.array(improvements)

    metrics["osd_method"]["mean_score"] = round(float(np.mean(osd_array)), 4)
    metrics["osd_method"]["median_score"] = round(float(np.median(osd_array)), 4)
    metrics["osd_method"]["max_score"] = round(float(np.max(osd_array)), 4)
    metrics["osd_method"]["min_score"] = round(float(np.min(osd_array)), 4)
    metrics["osd_method"]["std_score"] = round(float(np.std(osd_array)), 4)

    metrics["direct_method"]["mean_score"] = round(float(np.mean(direct_array)), 4)
    metrics["direct_method"]["median_score"] = round(float(np.median(direct_array)), 4)
    metrics["direct_method"]["max_score"] = round(float(np.max(direct_array)), 4)
    metrics["direct_method"]["min_score"] = round(float(np.min(direct_array)), 4)
    metrics["direct_method"]["std_score"] = round(float(np.std(direct_array)), 4)

    metrics["comparison"]["direct_better_rate"] = round(
        metrics["comparison"]["direct_wins"] / len(samples), 4
    )
    metrics["comparison"]["average_improvement"] = round(
        float(np.mean(improvements_array)), 4
    )
    metrics["comparison"]["max_improvement"] = round(
        float(np.max(improvements_array)), 4
    )
    metrics["comparison"]["min_improvement"] = round(
        float(np.min(improvements_array)), 4
    )

    return metrics


def generate_report(samples: List[Dict], metrics: Dict, output_file: str = None):
    """生成对比报告"""

    print("\n" + "=" * 80)
    print("OSD-based vs Direct Separation 对比分析报告".center(80))
    print("=" * 80 + "\n")

    # 样本级对比表
    print("【样本级对比】")
    print("-" * 120)
    print(
        f"{'ID':<5} {'OSD (avg)':<12} {'Direct':<12} {'Improvement':<15} {'Winner':<10}"
    )
    print("-" * 120)

    for sample in samples:
        osd_avg = (sample["osd_stitching"] + sample["osd_audio_cat"]) / 2
        direct = sample["direct_separation"]
        improvement = direct - osd_avg

        if direct > osd_avg:
            winner = "✅ Direct"
        elif direct < osd_avg:
            winner = "❌ OSD"
        else:
            winner = "➡️  Tie"

        print(
            f"{sample['id']:<5} {osd_avg:<12.4f} {direct:<12.4f} {improvement:+<14.4f} {winner:<10}"
        )

    print("-" * 120 + "\n")

    # 聚合指标
    print("【聚合指标】")
    print("-" * 80)
    print(f"{'指标':<30} {'OSD方法':<20} {'Direct方法':<20}")
    print("-" * 80)

    osd = metrics["osd_method"]
    direct = metrics["direct_method"]

    print(
        f"{'平均分数 (Mean)':<30} {osd['mean_score']:<20.4f} {direct['mean_score']:<20.4f}"
    )
    print(
        f"{'中位数 (Median)':<30} {osd['median_score']:<20.4f} {direct['median_score']:<20.4f}"
    )
    print(
        f"{'最高分 (Max)':<30} {osd['max_score']:<20.4f} {direct['max_score']:<20.4f}"
    )
    print(
        f"{'最低分 (Min)':<30} {osd['min_score']:<20.4f} {direct['min_score']:<20.4f}"
    )
    print(
        f"{'标准差 (Std)':<30} {osd['std_score']:<20.4f} {direct['std_score']:<20.4f}"
    )
    print(
        f"{'零分样本数':<30} {osd['zero_score_count']:<20} {direct['zero_score_count']:<20}"
    )
    print(
        f"{'非零分样本数':<30} {osd['nonzero_count']:<20} {direct['nonzero_count']:<20}"
    )

    print("-" * 80 + "\n")

    # 对比结果
    print("【对比结果】")
    print("-" * 80)
    cmp = metrics["comparison"]
    total = cmp["direct_wins"] + cmp["osd_wins"] + cmp["ties"]

    print(f"{'总样本数':<30} {total}")
    print(
        f"{'Direct方法胜利':<30} {cmp['direct_wins']} / {total} ({cmp['direct_better_rate']*100:.1f}%)"
    )
    print(
        f"{'OSD方法胜利':<30} {cmp['osd_wins']} / {total} ({cmp['osd_wins']/total*100:.1f}%)"
    )
    print(f"{'平局':<30} {cmp['ties']} / {total} ({cmp['ties']/total*100:.1f}%)")
    print(f"{'平均改进值':<30} {cmp['average_improvement']:+.4f}")
    print(f"{'最大改进':<30} {cmp['max_improvement']:+.4f}")
    print(f"{'最小改进':<30} {cmp['min_improvement']:+.4f}")

    print("-" * 80 + "\n")

    # 结论
    print("【结论】")
    print("-" * 80)
    if cmp["direct_better_rate"] > 0.6:
        print("✅ Direct Separation方法在本数据集上表现明显优于OSD-based方法")
        print(f"   - 胜率：{cmp['direct_better_rate']*100:.1f}%")
        print(f"   - 平均改进：{cmp['average_improvement']:+.4f} 分")
    else:
        print("⚠️  两种方法表现接近，需要更多数据验证")

    print("\n📊 建议：")
    if cmp["direct_better_rate"] > 0.6:
        print("   1. 采用 Direct Separation 作为主处理路径")
        print("   2. 保留 OSD 作为可选监控工具（不控制处理流程）")
        print("   3. 简化 streaming_overlap3_core.py，移除冗余的 OSD-based 分支")

    print("\n" + "=" * 80 + "\n")

    # 保存结果
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "samples": samples,
                    "metrics": metrics,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"📁 详细结果已保存到：{output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OSD vs Direct Separation 对比分析")
    parser.add_argument(
        "--analysis-file",
        default="test_overlap/comparison_analysis.md",
        help="输入的对比分析文件",
    )
    parser.add_argument(
        "--output",
        default="comparison_metrics.json",
        help="输出结果文件（JSON格式）",
    )

    args = parser.parse_args()

    # 加载数据
    samples = load_comparison_data(args.analysis_file)

    # 计算指标
    metrics = compute_metrics(samples)

    # 生成报告
    generate_report(samples, metrics, args.output)


if __name__ == "__main__":
    main()
