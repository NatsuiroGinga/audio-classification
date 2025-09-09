import numpy as np
import soundfile as sf
from modelscope.pipelines import pipeline
from typing import List, Tuple, Dict, Any
import tempfile
import os
import logging
from segmenter import (
    FunASRSegmenter,
    StreamingSentenceSegmenter,
    StreamingSegmenterConfig,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerVerificationWithSegmentation:
    def __init__(
        self,
        sv_model: str = "iic/speech_eres2net_base_sv_zh-cn_3dspeaker_16k",
        device: str = "cuda:0",  # 添加设备参数
        segmenter_type: str = "funasr",  # 'funasr' | 'streaming'
        streaming_cfg: dict | None = None,
        **segmenter_kwargs,
    ):
        """
        带音频分段的声纹识别系统

        Args:
            sv_model: 声纹识别模型
            device: 推理设备，'cuda:0' 使用GPU，'cpu' 使用CPU
            **segmenter_kwargs: FunASR分段器参数
        """
        # 初始化声纹识别管道 - 指定GPU设备
        self.sv_pipeline = pipeline(
            task="speaker-verification",
            model=sv_model,
            model_revision="v1.0.1",
            device=device,  # 添加设备参数
        )

        # 初始化分段器
        if segmenter_type == "funasr":
            self.segmenter = FunASRSegmenter(device=device, **segmenter_kwargs)
            self.segmenter_mode = "funasr"
        elif segmenter_type == "streaming":
            cfg = StreamingSegmenterConfig(**(streaming_cfg or {}))
            self.segmenter = StreamingSentenceSegmenter(cfg)
            self.segmenter_mode = "streaming"
        else:
            raise ValueError("segmenter_type 必须是 'funasr' 或 'streaming'")
        logger.info(f"使用分段器类型: {self.segmenter_mode}")

    def segment_and_verify(
        self, audio_path1: str, audio_path2: str, thr: float = 0.5
    ) -> Dict[str, Any]:
        """
        对音频进行分段并进行声纹识别

        Args:
            audio_path1: 第一个音频文件路径
            audio_path2: 第二个音频文件路径
            thr: 识别阈值

        Returns:
            包含分段结果和识别结果的字典
        """
        try:
            # 分段处理
            segments1 = self._get_audio_segments(audio_path1)
            segments2 = self._get_audio_segments(audio_path2)

            logger.info(f"音频1分段数: {len(segments1)}, 音频2分段数: {len(segments2)}")

            # 检查分段结果
            if not segments1 or not segments2:
                logger.warning(
                    f"分段结果为空: segments1={len(segments1)}, segments2={len(segments2)}"
                )
                return {
                    "segment_results": [],
                    "analysis": {
                        "conclusion": "no_segments",
                        "confidence": 0.0,
                        "avg_score": 0.0,
                    },
                    "num_segments1": len(segments1),
                    "num_segments2": len(segments2),
                }

            # 对每个分段进行声纹识别
            results = []

            for i, seg1 in enumerate(segments1):
                for j, seg2 in enumerate(segments2):
                    # 保存临时音频文件
                    temp_path1 = self._save_temp_audio(seg1, f"temp1_{i}.wav")
                    temp_path2 = self._save_temp_audio(seg2, f"temp2_{j}.wav")

                    try:
                        # 进行声纹识别
                        result = self.sv_pipeline([temp_path1, temp_path2], thr=thr)

                        # 调试输出：打印原始结果
                        logger.debug(f"原始声纹识别结果: {result}")

                        # 检查结果格式
                        if not isinstance(result, dict):
                            logger.error(
                                f"声纹识别结果格式错误，期望dict，得到: {type(result)}"
                            )
                            continue

                        if "score" not in result:
                            logger.error(f"声纹识别结果缺少'score'字段: {result}")
                            continue

                        if "text" not in result:
                            logger.error(f"声纹识别结果缺少'text'字段: {result}")
                            continue

                        # 根据实际输出格式处理结果
                        results.append(
                            {
                                "segment1_idx": i,
                                "segment2_idx": j,
                                "score": float(result["score"]),  # 确保是float类型
                                "text": result["text"],  # "yes" 或 "no"
                                "label": (
                                    "same" if result["text"] == "yes" else "different"
                                ),  # 转换为统一格式
                            }
                        )

                    except Exception as e:
                        logger.error(f"声纹识别失败 segment {i}-{j}: {str(e)}")
                        continue
                    finally:
                        # 清理临时文件
                        try:
                            if os.path.exists(temp_path1):
                                os.unlink(temp_path1)
                            if os.path.exists(temp_path2):
                                os.unlink(temp_path2)
                        except Exception as e:
                            logger.warning(f"清理临时文件失败: {str(e)}")

            # 分析结果
            analysis = self._analyze_results(results)

            return {
                "segment_results": results,
                "analysis": analysis,
                "num_segments1": len(segments1),
                "num_segments2": len(segments2),
            }

        except Exception as e:
            logger.error(f"segment_and_verify执行失败: {str(e)}")
            return {
                "segment_results": [],
                "analysis": {
                    "conclusion": "error",
                    "confidence": 0.0,
                    "avg_score": 0.0,
                    "error": str(e),
                },
                "num_segments1": 0,
                "num_segments2": 0,
            }

    def _get_audio_segments(self, audio_path: str) -> List[np.ndarray]:
        """获取音频分段"""
        try:
            segments, _ = self.segmenter.extract_audio_segments(audio_path)
            return segments
        except Exception as e:
            logger.error(f"音频分段失败 {audio_path}: {str(e)}")
            return []

    def _save_temp_audio(self, audio_segment: np.ndarray, filename: str) -> str:
        """保存临时音频文件"""
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)

        sf.write(temp_path, audio_segment, 16000)
        return temp_path

    def _analyze_results(self, results: List[Dict]) -> Dict[str, Any]:
        """分析识别结果"""
        if not results:
            logger.warning("没有有效的识别结果")
            return {
                "conclusion": "no_segments",
                "confidence": 0.0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "same_speaker_ratio": 0.0,
                "total_comparisons": 0,
            }

        try:
            # 计算平均得分 - 增加错误检查
            scores = []
            for r in results:
                if "score" in r and isinstance(r["score"], (int, float)):
                    scores.append(float(r["score"]))
                else:
                    logger.warning(f"结果中缺少有效的score字段: {r}")

            if not scores:
                logger.error("没有有效的分数数据")
                return {
                    "conclusion": "no_valid_scores",
                    "confidence": 0.0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "min_score": 0.0,
                    "same_speaker_ratio": 0.0,
                    "total_comparisons": len(results),
                }

            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)

            # 统计标签 - 使用转换后的label字段
            labels = []
            for r in results:
                if "label" in r:
                    labels.append(r["label"])
                else:
                    logger.warning(f"结果中缺少label字段: {r}")

            if not labels:
                logger.error("没有有效的标签数据")
                return {
                    "conclusion": "no_valid_labels",
                    "confidence": 0.0,
                    "avg_score": avg_score,
                    "max_score": max_score,
                    "min_score": min_score,
                    "same_speaker_ratio": 0.0,
                    "total_comparisons": len(results),
                }

            same_speaker_count = sum(1 for label in labels if label == "same")
            total_comparisons = len(labels)
            same_speaker_ratio = same_speaker_count / total_comparisons

            # 得出结论
            if same_speaker_ratio > 0.6 and avg_score > 0.5:
                conclusion = "same_speaker"
                confidence = avg_score
            elif same_speaker_ratio < 0.4 and avg_score < 0.5:
                conclusion = "different_speaker"
                confidence = 1 - avg_score
            else:
                conclusion = "uncertain"
                confidence = 0.5

            return {
                "conclusion": conclusion,
                "confidence": float(confidence),
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "min_score": float(min_score),
                "same_speaker_ratio": float(same_speaker_ratio),
                "total_comparisons": total_comparisons,
            }

        except Exception as e:
            logger.error(f"分析结果时出错: {str(e)}")
            return {
                "conclusion": "analysis_error",
                "confidence": 0.0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "same_speaker_ratio": 0.0,
                "total_comparisons": len(results),
                "error": str(e),
            }


# 使用示例
if __name__ == "__main__":
    # 使用GPU进行推理
    sv_system = SpeakerVerificationWithSegmentation(
        device="cuda:0",  # 使用第一块GPU
        vad_model="fsmn-vad",
        max_segment_length=8.0,
        min_segment_length=1.5,
        merge_threshold=0.3,
    )

    # 测试音频路径
    speaker1_a_wav = "/data/workspace/llm/audio-classification/dataset/test/3D_SPK_06154/3D_SPK_06154_001_Device03_Distance00_Dialect00.wav"
    speaker1_b_wav = "/data/workspace/llm/audio-classification/dataset/test/3D_SPK_06154/3D_SPK_06154_001_Device06_Distance00_Dialect00.wav"

    # 进行分段声纹识别
    result = sv_system.segment_and_verify(speaker1_a_wav, speaker1_b_wav)

    print("分段声纹识别结果:")
    print(f"结论: {result['analysis']['conclusion']}")
    print(f"置信度: {result['analysis']['confidence']:.3f}")
    print(f"平均得分: {result['analysis']['avg_score']:.3f}")
    print(f"相同说话人比例: {result['analysis']['same_speaker_ratio']:.3f}")
    print(f"总比较次数: {result['analysis']['total_comparisons']}")
