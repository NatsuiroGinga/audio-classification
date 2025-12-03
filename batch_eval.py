#!/usr/bin/env python3
"""
批量分析多个测试结果的性能和速度指标
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse


def analyze_batch_results(result_dirs: List[Path]) -> Dict[str, Any]:
    """分析批量结果"""

    # 初始化统计字典
    stats = {
        # 速度指标
        "rtf_total": [],
        "rtf_asr": [],
        "time_total": [],
        "time_osd": [],
        "time_sep": [],
        "time_asr": [],
        # 准确率指标
        "target_hit_rate": [],
        "segments_matched": [],
        "segments_total": [],
        # 分离质量指标
        "sisdr": [],
        "sisdri": [],
        "sep_eval_segments": [],
        # 资源使用
        "cpu_avg": [],
        "cpu_peak": [],
        "memory_avg": [],
        "memory_peak": [],
        # 音频统计
        "audio_duration": [],
        "overlap_ratio": [],
    }

    valid_results = 0

    for result_dir in result_dirs:
        metrics_file = result_dir / "metrics.json"
        summary_file = result_dir / "summary.json"

        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            # 提取指标
            # 速度指标
            if metrics.get("rtf_total"):
                stats["rtf_total"].append(metrics["rtf_total"])
            if metrics.get("rtf_asr"):
                stats["rtf_asr"].append(metrics["rtf_asr"])
            if metrics.get("time_compute_total_sec"):
                stats["time_total"].append(metrics["time_compute_total_sec"])
            if metrics.get("time_osd_sec"):
                stats["time_osd"].append(metrics["time_osd_sec"])
            if metrics.get("time_sep_sec"):
                stats["time_sep"].append(metrics["time_sep_sec"])
            if metrics.get("time_asr_sec"):
                stats["time_asr"].append(metrics["time_asr_sec"])

            # 准确率指标
            if metrics.get("target_hit_rate_segments"):
                stats["target_hit_rate"].append(metrics["target_hit_rate_segments"])
            if metrics.get("segments_matched"):
                stats["segments_matched"].append(metrics["segments_matched"])
            if metrics.get("segments_total"):
                stats["segments_total"].append(metrics["segments_total"])

            # 分离质量指标
            if metrics.get("sep_sisdr_mean"):
                stats["sisdr"].append(metrics["sep_sisdr_mean"])
            if metrics.get("sep_sisdri_mean"):
                stats["sisdri"].append(metrics["sep_sisdri_mean"])
            if metrics.get("sep_eval_segments"):
                stats["sep_eval_segments"].append(metrics["sep_eval_segments"])

            # 资源使用
            if metrics.get("cpu_avg"):
                stats["cpu_avg"].append(metrics["cpu_avg"])
            if metrics.get("cpu_peak"):
                stats["cpu_peak"].append(metrics["cpu_peak"])
            if metrics.get("rss_avg_mb"):
                stats["memory_avg"].append(metrics["rss_avg_mb"])
            if metrics.get("rss_peak_mb"):
                stats["memory_peak"].append(metrics["rss_peak_mb"])

            # 音频统计
            if metrics.get("total_audio_sec"):
                stats["audio_duration"].append(metrics["total_audio_sec"])
                if metrics.get("audio_overlap_sec"):
                    overlap_ratio = (
                        metrics["audio_overlap_sec"] / metrics["total_audio_sec"]
                    )
                    stats["overlap_ratio"].append(overlap_ratio)

            valid_results += 1

        except Exception as e:
            print(f"Error processing {result_dir}: {e}")
            continue

    # 计算统计量
    summary = {"total_results": valid_results, "metrics": {}}

    for key, values in stats.items():
        if values:
            arr = np.array(values)
            summary["metrics"][key] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": len(values),
            }
        else:
            summary["metrics"][key] = {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "count": 0,
            }

    return summary


def generate_report(summary: Dict[str, Any]) -> str:
    """生成可读的报告"""

    report = []
    report.append("=" * 60)
    report.append("语音分离识别系统批量测试报告")
    report.append("=" * 60)
    report.append(f"总测试结果数: {summary['total_results']}")
    report.append("")

    metrics = summary["metrics"]

    # 速度指标
    report.append("处理速度指标:")
    if metrics["rtf_total"]["count"] > 0:
        rtf = metrics["rtf_total"]
        report.append(f"  • 实时因子 (RTF): {rtf['mean']:.4f} ± {rtf['std']:.4f}")
        report.append(f"    (范围: {rtf['min']:.4f} - {rtf['max']:.4f})")

        # RTF性能评级
        rtf_mean = rtf["mean"]
        report.append(f"    处理1秒音频需 {rtf_mean:.2f} 秒")

    # 准确率指标
    report.append("")
    report.append("准确率指标(?):")
    if metrics["target_hit_rate"]["count"] > 0:
        hit_rate = metrics["target_hit_rate"]
        report.append(
            f"  • target_hit_rate_segments: {hit_rate['mean']:.1%} ± {hit_rate['std']:.1%}"
        )

    # 分离质量指标
    report.append("")
    report.append("分离质量指标:")
    if metrics["sisdr"]["count"] > 0:
        sisdr = metrics["sisdr"]
        sisdri = metrics["sisdri"]
        report.append(f"  • SI-SDR: {sisdr['mean']:.2f} ± {sisdr['std']:.2f} dB")
        report.append(f"  • SI-SDRi: {sisdri['mean']:.2f} ± {sisdri['std']:.2f} dB")

    # 资源使用
    report.append("")
    report.append("资源使用:")
    if metrics["cpu_avg"]["count"] > 0:
        cpu = metrics["cpu_avg"]
        mem = metrics["memory_avg"]
        report.append(f"  • CPU使用率: {cpu['mean']:.1f}% ± {cpu['std']:.1f}%")
        report.append(f"  • 内存使用: {mem['mean']:.1f} MB ± {mem['std']:.1f} MB")

    # 时间分解
    report.append("")
    report.append("时间分解 (秒):")
    if metrics["time_total"]["count"] > 0:
        total = metrics["time_total"]["mean"]
        osd = metrics["time_osd"]["mean"] or 0
        sep = metrics["time_sep"]["mean"] or 0
        asr = metrics["time_asr"]["mean"] or 0

        report.append(f"  • 重叠检测: {osd:.3f}s ({osd/total*100:.1f}%)")
        report.append(f"  • 语音分离: {sep:.3f}s ({sep/total*100:.1f}%)")
        report.append(f"  • 语音识别: {asr:.3f}s ({asr/total*100:.1f}%)")
        report.append(
            f"  • 其他: {total-osd-sep-asr:.3f}s ({(total-osd-sep-asr)/total*100:.1f}%)"
        )

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="批量分析语音分离识别结果")
    parser.add_argument("--results-dir", required=True, help="包含多个测试结果的目录")
    parser.add_argument("--output", default="batch_analysis.json", help="输出文件路径")
    args = parser.parse_args()

    base_dir = Path(args.results_dir)

    # 查找所有结果目录
    result_dirs = []
    for item in base_dir.iterdir():
        if item.is_dir():
            # 检查是否包含metrics.json
            if (item / "metrics.json").exists():
                result_dirs.append(item)

    if not result_dirs:
        print(f"在 {base_dir} 中未找到有效的结果目录")
        return

    print(f"找到 {len(result_dirs)} 个结果目录")

    # 分析批量结果
    summary = analyze_batch_results(result_dirs)

    # 保存详细统计
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 生成并打印报告
    report = generate_report(summary)
    print(report)

    # 保存报告文本
    report_file = Path(args.output).with_suffix(".txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n详细统计已保存到: {args.output}")
    print(f"报告文本已保存到: {report_file}")


if __name__ == "__main__":
    main()
