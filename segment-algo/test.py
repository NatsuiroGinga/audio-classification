import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm
from main import SpeakerVerificationWithSegmentation

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetTester:
    def __init__(
        self,
        dataset_path: str = "./dataset/test",
        output_dir: str = "./test_results",
        device: str = "cuda:0",  # 添加设备参数
        **segmenter_kwargs,
    ):
        """
        数据集测试器

        Args:
            dataset_path: 测试数据集路径
            output_dir: 测试结果输出目录
            device: 推理设备，'cuda:0' 使用GPU，'cpu' 使用CPU
            **segmenter_kwargs: FunASR分段器参数
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化声纹识别系统 - 传递设备参数
        # 保留单模式实例（向后兼容旧用法）
        self.sv_system = SpeakerVerificationWithSegmentation(
            device=device, **segmenter_kwargs
        )
        self.device = device
        self.base_segmenter_kwargs = segmenter_kwargs

        # 测试结果存储
        self.test_results = []
        self.summary_stats = {}

    def scan_dataset(self) -> Dict[str, List[str]]:
        """
        扫描数据集，获取每个说话人的音频文件列表

        Returns:
            字典，键为说话人ID，值为该说话人的音频文件路径列表
        """
        speaker_files = {}

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.dataset_path}")

        # 遍历所有子目录
        for speaker_dir in self.dataset_path.iterdir():
            if speaker_dir.is_dir():
                speaker_id = speaker_dir.name
                audio_files = []

                # 获取该说话人的所有wav文件
                for audio_file in speaker_dir.glob("*.wav"):
                    audio_files.append(str(audio_file))

                if audio_files:
                    speaker_files[speaker_id] = sorted(audio_files)
                    logger.info(
                        f"发现说话人 {speaker_id}: {len(audio_files)} 个音频文件"
                    )

        logger.info(f"总共发现 {len(speaker_files)} 个说话人")
        return speaker_files

    def test_same_speaker_verification(
        self, speaker_files: Dict[str, List[str]], max_pairs_per_speaker: int = 10
    ) -> List[Dict]:
        """
        测试同一说话人的声纹识别准确率

        Args:
            speaker_files: 说话人文件字典
            max_pairs_per_speaker: 每个说话人最大测试对数

        Returns:
            测试结果列表
        """
        same_speaker_results = []

        logger.info("开始测试同一说话人声纹识别...")

        for speaker_id, files in tqdm(speaker_files.items(), desc="测试同一说话人"):
            if len(files) < 2:
                logger.warning(f"说话人 {speaker_id} 只有 {len(files)} 个文件，跳过")
                continue

            # 限制测试对数以避免过多计算
            pairs_tested = 0
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    if pairs_tested >= max_pairs_per_speaker:
                        break

                    try:
                        result = self.sv_system.segment_and_verify(
                            files[i], files[j], thr=0.5
                        )
                        analysis = result.get("analysis", {})
                        test_result = {
                            "test_type": "same_speaker",
                            "speaker_id": speaker_id,
                            "file1": os.path.basename(files[i]),
                            "file2": os.path.basename(files[j]),
                            "file1_path": files[i],
                            "file2_path": files[j],
                            "conclusion": analysis.get("conclusion", "unknown"),
                            "confidence": analysis.get("confidence", 0.0),
                            "avg_score": analysis.get("avg_score", 0.0),
                            "same_speaker_ratio": analysis.get(
                                "same_speaker_ratio", None
                            ),
                            "num_segments1": result.get("num_segments1", 0),
                            "num_segments2": result.get("num_segments2", 0),
                            "total_comparisons": analysis.get("total_comparisons", 0),
                            "expected_result": "same_speaker",
                        }
                        test_result["correct"] = (
                            test_result["conclusion"] == "same_speaker"
                        )
                        same_speaker_results.append(test_result)
                        pairs_tested += 1
                    except Exception as e:
                        logger.error(
                            f"测试失败 {speaker_id}: {files[i]} vs {files[j]}, 错误: {e}"
                        )

                if pairs_tested >= max_pairs_per_speaker:
                    break

        return same_speaker_results

    def test_different_speaker_verification(
        self, speaker_files: Dict[str, List[str]], max_pairs: int = 50
    ) -> List[Dict]:
        """
        测试不同说话人的声纹识别准确率

        Args:
            speaker_files: 说话人文件字典
            max_pairs: 最大测试对数

        Returns:
            测试结果列表
        """
        different_speaker_results = []
        speaker_ids = list(speaker_files.keys())

        logger.info("开始测试不同说话人声纹识别...")

        pairs_tested = 0
        for i in tqdm(range(len(speaker_ids)), desc="测试不同说话人"):
            for j in range(i + 1, len(speaker_ids)):
                if pairs_tested >= max_pairs:
                    break

                speaker1_id = speaker_ids[i]
                speaker2_id = speaker_ids[j]

                # 随机选择每个说话人的一个文件
                file1 = np.random.choice(speaker_files[speaker1_id])
                file2 = np.random.choice(speaker_files[speaker2_id])

                try:
                    result = self.sv_system.segment_and_verify(file1, file2, thr=0.5)
                    analysis = result.get("analysis", {})
                    test_result = {
                        "test_type": "different_speaker",
                        "speaker1_id": speaker1_id,
                        "speaker2_id": speaker2_id,
                        "file1": os.path.basename(file1),
                        "file2": os.path.basename(file2),
                        "file1_path": file1,
                        "file2_path": file2,
                        "conclusion": analysis.get("conclusion", "unknown"),
                        "confidence": analysis.get("confidence", 0.0),
                        "avg_score": analysis.get("avg_score", 0.0),
                        "same_speaker_ratio": analysis.get("same_speaker_ratio", None),
                        "num_segments1": result.get("num_segments1", 0),
                        "num_segments2": result.get("num_segments2", 0),
                        "total_comparisons": analysis.get("total_comparisons", 0),
                        "expected_result": "different_speaker",
                    }
                    test_result["correct"] = (
                        test_result["conclusion"] == "different_speaker"
                    )
                    different_speaker_results.append(test_result)
                    pairs_tested += 1

                except Exception as e:
                    logger.error(
                        f"测试失败 {speaker1_id} vs {speaker2_id}: {file1} vs {file2}, 错误: {str(e)}"
                    )

            if pairs_tested >= max_pairs:
                break

        return different_speaker_results

    def calculate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """
        计算测试指标

        Args:
            results: 测试结果列表

        Returns:
            指标字典
        """
        if not results:
            return {}

        # 分组
        same_speaker_results = [
            r for r in results if r.get("test_type") == "same_speaker"
        ]
        different_speaker_results = [
            r for r in results if r.get("test_type") == "different_speaker"
        ]

        metrics: Dict[str, Any] = {}

        # 同一说话人
        if same_speaker_results:
            same_correct = sum(1 for r in same_speaker_results if r.get("correct"))
            same_total = len(same_speaker_results)
            same_accuracy = same_correct / same_total if same_total else 0.0
            same_conf_list = [r.get("confidence", 0.0) for r in same_speaker_results]
            same_score_list = [r.get("avg_score", 0.0) for r in same_speaker_results]
            metrics["same_speaker"] = {
                # 保留小数点后三位
                "accuracy": round(same_accuracy, 3),
                "correct_count": same_correct,
                "total_count": same_total,
                "avg_confidence": (
                    float(np.mean(same_conf_list)) if same_conf_list else 0.0
                ),
                "avg_score": (
                    float(np.mean(same_score_list)) if same_score_list else 0.0
                ),
            }

        # 不同说话人
        if different_speaker_results:
            diff_correct = sum(1 for r in different_speaker_results if r.get("correct"))
            diff_total = len(different_speaker_results)
            diff_accuracy = diff_correct / diff_total if diff_total else 0.0
            diff_conf_list = [
                r.get("confidence", 0.0) for r in different_speaker_results
            ]
            diff_score_list = [
                r.get("avg_score", 0.0) for r in different_speaker_results
            ]
            metrics["different_speaker"] = {
                "accuracy": round(diff_accuracy, 3),
                "correct_count": diff_correct,
                "total_count": diff_total,
                "avg_confidence": (
                    float(np.mean(diff_conf_list)) if diff_conf_list else 0.0
                ),
                "avg_score": (
                    float(np.mean(diff_score_list)) if diff_score_list else 0.0
                ),
            }

        # 总体
        all_correct = sum(1 for r in results if r.get("correct"))
        all_total = len(results)
        overall_accuracy = all_correct / all_total if all_total else 0.0
        metrics["overall"] = {
            "accuracy": round(overall_accuracy, 3),
            "correct_count": all_correct,
            "total_count": all_total,
            "same_speaker_tests": len(same_speaker_results),
            "different_speaker_tests": len(different_speaker_results),
        }

        return metrics

    def save_results(self, results: List[Dict], metrics: Dict[str, Any]):
        """
        保存测试结果到文件

        Args:
            results: 测试结果列表
            metrics: 测试指标
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存详细结果到JSON
        json_file = self.output_dir / f"test_results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": timestamp, "results": results, "metrics": metrics},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 保存结果到CSV
        csv_file = self.output_dir / f"test_results_{timestamp}.csv"
        if results:
            # 统一所有字段（并保证稳定顺序：固定优先字段 + 其它排序）
            preferred = [
                "test_type",
                "speaker_id",
                "speaker1_id",
                "speaker2_id",
                "file1",
                "file2",
                "file1_path",
                "file2_path",
                "conclusion",
                "expected_result",
                "correct",
                "confidence",
                "avg_score",
                "same_speaker_ratio",
                "num_segments1",
                "num_segments2",
                "total_comparisons",
            ]
            all_keys = set()
            for r in results:
                all_keys.update(r.keys())
            ordered = [k for k in preferred if k in all_keys] + sorted(
                k for k in all_keys if k not in preferred
            )
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
                writer.writeheader()
                for r in results:
                    row = {k: r.get(k, "") for k in ordered}
                    writer.writerow(row)

        # 保存指标摘要
        summary_file = self.output_dir / f"test_summary_{timestamp}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"声纹识别测试报告\n")
            f.write(f"测试时间: {timestamp}\n")
            f.write(f"数据集路径: {self.dataset_path}\n\n")

            if "overall" in metrics:
                f.write(f"总体结果:\n")
                f.write(f"  总准确率: {metrics['overall']['accuracy']:.3f}\n")
                f.write(
                    f"  正确预测: {metrics['overall']['correct_count']}/{metrics['overall']['total_count']}\n"
                )
                f.write(
                    f"  同一说话人测试: {metrics['overall']['same_speaker_tests']} 对\n"
                )
                f.write(
                    f"  不同说话人测试: {metrics['overall']['different_speaker_tests']} 对\n\n"
                )

            if "same_speaker" in metrics:
                f.write(f"同一说话人测试:\n")
                f.write(f"  准确率: {metrics['same_speaker']['accuracy']:.3f}\n")
                f.write(
                    f"  平均置信度: {metrics['same_speaker']['avg_confidence']:.3f}\n"
                )
                f.write(f"  平均得分: {metrics['same_speaker']['avg_score']:.3f}\n\n")

            if "different_speaker" in metrics:
                f.write(f"不同说话人测试:\n")
                f.write(f"  准确率: {metrics['different_speaker']['accuracy']:.3f}\n")
                f.write(
                    f"  平均置信度: {metrics['different_speaker']['avg_confidence']:.3f}\n"
                )
                f.write(
                    f"  平均得分: {metrics['different_speaker']['avg_score']:.3f}\n"
                )

        logger.info(f"测试结果已保存到:")
        logger.info(f"  详细结果: {json_file}")
        logger.info(f"  CSV格式: {csv_file}")
        logger.info(f"  摘要报告: {summary_file}")

    def run_full_test(
        self, max_same_pairs_per_speaker: int = 5, max_different_pairs: int = 30
    ):
        """
        运行完整测试

        Args:
            max_same_pairs_per_speaker: 每个说话人最大同一说话人测试对数
            max_different_pairs: 最大不同说话人测试对数
        """
        logger.info("开始运行完整测试...")

        # 扫描数据集
        speaker_files = self.scan_dataset()

        if not speaker_files:
            logger.error("未找到任何音频文件，测试终止")
            return

        # 测试同一说话人
        same_results = self.test_same_speaker_verification(
            speaker_files, max_same_pairs_per_speaker
        )

        # 测试不同说话人
        different_results = self.test_different_speaker_verification(
            speaker_files, max_different_pairs
        )

        # 合并结果
        all_results = same_results + different_results

        # 计算指标
        metrics = self.calculate_metrics(all_results)

        # 保存结果
        self.save_results(all_results, metrics)

        # 打印摘要
        if "overall" in metrics:
            logger.info(f"测试完成! 总准确率: {metrics['overall']['accuracy']:.3f}")
            logger.info(
                f"同一说话人准确率: {metrics.get('same_speaker', {}).get('accuracy', 0):.3f}"
            )
            logger.info(
                f"不同说话人准确率: {metrics.get('different_speaker', {}).get('accuracy', 0):.3f}"
            )

    # ------------------------------------------------------------------
    # 新增：多分段模式对比（funasr vs streaming）
    # ------------------------------------------------------------------
    def run_compare_modes(
        self,
        segmenter_types=("funasr", "streaming"),
        streaming_cfg: Dict[str, Any] | None = None,
        max_same_pairs_per_speaker: int = 5,
        max_different_pairs: int = 30,
    ):
        """对多个分段模式分别评测并输出对比结果。

        会为每个模式生成独立的结果文件，并输出一个总汇总 JSON。
        """
        speaker_files = self.scan_dataset()
        if not speaker_files:
            logger.error("未找到任何音频文件，终止对比测试")
            return

        summary = {}
        all_mode_metrics = {}
        compare_index = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for mode in segmenter_types:
            logger.info(f"=== 开始模式评估: {mode} ===")
            if mode == "funasr":
                sv_system = SpeakerVerificationWithSegmentation(
                    device=self.device,
                    segmenter_type="funasr",
                    **self.base_segmenter_kwargs,
                )
            elif mode == "streaming":
                sv_system = SpeakerVerificationWithSegmentation(
                    device=self.device,
                    segmenter_type="streaming",
                    streaming_cfg=streaming_cfg or {},
                )
            else:
                logger.warning(f"未知模式 {mode}，跳过")
                continue

            # 执行同/不同说话人测试
            original_system = self.sv_system
            self.sv_system = sv_system  # 复用已有测试函数
            same_results = self.test_same_speaker_verification(
                speaker_files, max_same_pairs_per_speaker
            )
            different_results = self.test_different_speaker_verification(
                speaker_files, max_different_pairs
            )
            self.sv_system = original_system  # 还原

            mode_results = same_results + different_results
            # 增加模式标记
            for r in mode_results:
                r["segmenter_type"] = mode

            metrics = self.calculate_metrics(mode_results)
            all_mode_metrics[mode] = metrics

            # 保存单模式结果
            mode_tag = f"{mode}_results_{timestamp}.json"
            mode_path = self.output_dir / mode_tag
            with open(mode_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": mode,
                        "timestamp": timestamp,
                        "results": mode_results,
                        "metrics": metrics,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(f"模式 {mode} 结果已保存: {mode_path}")

            if "overall" in metrics:
                compare_index.append(
                    {
                        "mode": mode,
                        "overall_accuracy": metrics["overall"].get("accuracy", 0),
                        "same_accuracy": metrics.get("same_speaker", {}).get(
                            "accuracy", 0
                        ),
                        "different_accuracy": metrics.get("different_speaker", {}).get(
                            "accuracy", 0
                        ),
                        "avg_same_confidence": metrics.get("same_speaker", {}).get(
                            "avg_confidence", 0
                        ),
                        "avg_diff_confidence": metrics.get("different_speaker", {}).get(
                            "avg_confidence", 0
                        ),
                    }
                )

        # 写入函数参数
        summary["params"] = {
            "segmenter_types": segmenter_types,
            "streaming_cfg": streaming_cfg,
            "max_same_pairs_per_speaker": max_same_pairs_per_speaker,
            "max_different_pairs": max_different_pairs,
        }
        summary["modes"] = compare_index
        summary["detailed_metrics"] = all_mode_metrics
        summary_path = self.output_dir / f"mode_comparison_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"多模式对比汇总已保存: {summary_path}，摘要:")
        for row in compare_index:
            logger.info(
                f"  {row['mode']} -> overall={row['overall_accuracy']:.3f}, same={row['same_accuracy']:.3f}, diff={row['different_accuracy']:.3f}"
            )


def main():
    """主函数"""
    dataset_root = "/data/workspace/llm/audio-classification/dataset/test"
    out_dir = "/data/workspace/llm/audio-classification/test_results"

    tester = DatasetTester(
        dataset_path=dataset_root,
        output_dir=out_dir,
        device="cuda:0",
        vad_model="fsmn-vad",
        max_segment_length=8.0,
        min_segment_length=1.5,
        merge_threshold=0.3,
    )

    # TODO: 提取命令行参数，自定义测试配置(数据集路径、输出目录、分段器参数等)

    # 单模式（原流程）
    # tester.run_full_test(max_same_pairs_per_speaker=3, max_different_pairs=3)

    # 多模式对比：funasr vs streaming
    tester.run_compare_modes(
        segmenter_types=("funasr", "streaming"),
        streaming_cfg={
            "target_sentence_sec": 3.0,
            "hard_max_sentence_sec": 5.0,
            "min_speech_sec": 0.25,
            "min_silence_sec": 0.18,
            "use_webrtcvad": True,
            "combine_with_vad": True,
        },
        max_same_pairs_per_speaker=100,
        max_different_pairs=100,
    )


if __name__ == "__main__":
    main()
