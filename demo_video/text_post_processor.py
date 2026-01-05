#!/usr/bin/env python3
"""
文本后处理器 - 清理和改进识别结果
"""
import re
from typing import List, Dict
from difflib import SequenceMatcher


class TextPostProcessor:
    """后处理识别文本的工具类"""

    def __init__(self, target_text: str = ""):
        """
        初始化处理器

        Args:
            target_text: 目标参考文本，用于对齐和纠正
        """
        self.target_text = target_text
        self.target_text_no_punct = self._remove_punctuation(target_text)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """移除中文标点符号"""
        chinese_puncts = r"[。，、；：？！（）《》" "''【】…—～·]"
        return re.sub(chinese_puncts, "", text)

    def merge_segments(self, results: List[Dict]) -> str:
        """
        合并识别段落，去重和修复

        Args:
            results: 识别结果列表

        Returns:
            合并后的文本
        """
        if not results:
            return ""

        # 提取文本并排序
        sorted_results = sorted(
            results, key=lambda r: (r.get("seq_id", 0), r.get("start", 0))
        )
        texts = [r.get("text", "") for r in sorted_results]

        # 合并前清理
        texts = [self._remove_punctuation(t) for t in texts]

        # 去除空白文本
        texts = [t for t in texts if t.strip()]

        # 去重相邻重复
        merged = []
        for text in texts:
            if merged and merged[-1] == text:
                continue  # 跳过完全重复的相邻段
            merged.append(text)

        # 尝试与目标文本对齐，纠正明显错误
        merged = self._align_with_target(merged)

        return "".join(merged)

    def _align_with_target(self, texts: List[str]) -> List[str]:
        """
        尝试与目标文本对齐，纠正拼写错误

        Args:
            texts: 分段文本列表

        Returns:
            修正后的文本列表
        """
        if not self.target_text_no_punct:
            return texts

        # 简单启发式对齐：看target中是否包含该文本的部分
        result = []
        current_pos = 0

        for text in texts:
            # 查找最接近的匹配
            best_match = None
            best_score = 0.5  # 最小相似度阈值

            for start in range(
                max(0, current_pos - 5),
                min(len(self.target_text_no_punct), current_pos + 20),
            ):
                for end in range(start + len(text) - 2, start + len(text) + 3):
                    if start >= 0 and end <= len(self.target_text_no_punct):
                        target_substr = self.target_text_no_punct[start:end]
                        ratio = SequenceMatcher(None, text, target_substr).ratio()
                        if ratio > best_score:
                            best_score = ratio
                            best_match = target_substr
                            current_pos = end

            # 如果找到相似的目标文本片段，使用它；否则保持原文
            if best_match and best_score > 0.6:
                result.append(best_match)
            else:
                result.append(text)

        return result

    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        获取识别统计信息

        Args:
            results: 识别结果列表

        Returns:
            统计信息字典
        """
        if not results:
            return {}

        sv_scores = [r.get("sv_score", 0) for r in results if r.get("sv_score")]
        kinds = {}
        for r in results:
            k = r.get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1

        return {
            "total_segments": len(results),
            "avg_sv_score": sum(sv_scores) / len(sv_scores) if sv_scores else 0,
            "max_sv_score": max(sv_scores) if sv_scores else 0,
            "min_sv_score": min(sv_scores) if sv_scores else 0,
            "kind_distribution": kinds,
        }

    def print_comparison(self, merged_text: str) -> None:
        """
        打印目标文本和识别文本的对比

        Args:
            merged_text: 识别合并后的文本
        """
        print("╔" + "═" * 68 + "╗")
        print("║                  📊 文本对比分析                             ║")
        print("╚" + "═" * 68 + "╝")
        print()

        print("🎯 目标文本:")
        print(f"  {self.target_text}")
        print()

        print("🔍 识别文本:")
        print(f"  {merged_text}")
        print()

        # 计算相似度
        target_clean = self._remove_punctuation(self.target_text)
        ratio = SequenceMatcher(None, target_clean, merged_text).ratio()
        print(f"📈 相似度: {ratio*100:.1f}%")
        print()


if __name__ == "__main__":
    # 测试示例
    target = "苏北军的一些爱国将士马占山、李杜、唐巨武、苏炳爱、邓铁梅等也奋起抗战。"
    processor = TextPostProcessor(target)

    # 模拟识别结果
    results = [
        {"seq_id": 1, "start": 3.97, "text": "苏北军的一些爱国。"},
        {"seq_id": 2, "start": 5.95, "text": "将士马占山。"},
        {"seq_id": 3, "start": 5.95, "text": "杜唐巨武。"},
        {"seq_id": 4, "start": 7.94, "text": "朱炳爱、邓铁梅等。"},
        {"seq_id": 5, "start": 9.62, "text": "并且抗战。"},
        {"seq_id": 6, "start": 9.62, "text": "也奋起。"},
    ]

    merged = processor.merge_segments(results)
    stats = processor.get_statistics(results)

    print("合并结果:", merged)
    print("统计信息:", stats)
    processor.print_comparison(merged)
