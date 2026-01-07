"""诱导词拦截器模块。

提供诱导词过滤功能，配合 KWS 模型使用。

注意：KWS 是基于声学特征识别，不是文字识别。
- 一声同音字（珍/甄/臻）与目标词"真"音频波形完全相同，KWS 无法区分
- 只需阻断非一声变体（镇/诊/振/阵/震/枕）和声母变体（李浩/泥豪）
"""

from typing import List, Set

# 目标词
TARGET_KEYWORD_NIHAO_ZHENZHEN = "你好真真"

# 诱导词集合 - "你好真真"的防御矩阵（应阻断）
# 注意：
# - 一声同音字（珍/甄/臻）与目标词发音完全相同，KWS 无法区分，无需处理
# - 同声调同韵母的字发音相同（如镇/阵/振/震都是zhèn），只需保留一个代表词
# - 声母变体（李浩/泥豪）KWS 模型无法有效区分，已移除
DECOY_KEYWORDS_NIHAO_ZHENZHEN = {
    # Level 1: 声调层（非一声，每个声调保留一个代表词）
    "你好镇镇",  # zhèn (第四声) - 代表：镇/阵/振/震
    "你好诊诊",  # zhěn (第三声) - 代表：诊/枕
    # Level 2: 鼻音层（韵尾不同，每个声调保留一个代表词）
    "你好正正",  # zhèng (第四声)
    "你好争争",  # zhēng (第一声) - 代表：争/征
    "你好整整",  # zhěng (第三声)
    # Level 3: 声母层（声母不同，需分别保留）
    "你好认认",  # rèn (r vs zh)
    "你好曾曾",  # céng (c vs zh)
    "你好怎怎",  # zěn (z vs zh)
}


class DecoyFilter:
    """诱导词过滤器。

    用于在 KWS 检测后阻断非目标谐音词。

    注意：KWS 是声学特征识别，一声同音字（珍/甄/臻）与目标词音频相同，
    KWS 会直接返回目标词，无需白名单映射。

    使用示例:
        filter = DecoyFilter(DECOY_KEYWORDS_NIHAO_ZHENZHEN)
        detections = kws_model.detect(samples, sr)
        filtered = filter.filter(detections)
    """

    def __init__(
        self,
        decoy_keywords: Set[str],
        log_intercepted: bool = False,
    ):
        """初始化过滤器。

        Args:
            decoy_keywords: 阻断词集合
            log_intercepted: 是否记录被拦截的词（用于调试）
        """
        self.decoy_keywords = decoy_keywords
        self.log_intercepted = log_intercepted
        self.intercepted_count = 0
        self.intercepted_keywords: List[str] = []

    def is_decoy(self, keyword: str) -> bool:
        """判断是否为诱导词（应阻断）。"""
        return keyword in self.decoy_keywords

    def filter(self, detections: List) -> List:
        """过滤检测结果，阻断诱导词。

        Args:
            detections: KWS 检测结果列表

        Returns:
            过滤后的检测结果
        """
        filtered = []
        for detection in detections:
            keyword = detection.keyword

            if self.is_decoy(keyword):
                # 阻断
                self.intercepted_count += 1
                if self.log_intercepted:
                    self.intercepted_keywords.append(keyword)
                    print(f"[Decoy Intercepted] {keyword}")
            else:
                # 直接通过
                filtered.append(detection)

        return filtered

    def get_stats(self) -> dict:
        """获取统计信息。

        Returns:
            统计字典，包含拦截详情
        """
        return {
            "total_intercepted": self.intercepted_count,
            "intercepted_keywords": self.intercepted_keywords,
        }

    def reset_stats(self):
        """重置统计信息。"""
        self.intercepted_count = 0
        self.intercepted_keywords = []


def create_nihao_zhenzhen_filter(log_intercepted: bool = False) -> DecoyFilter:
    """创建"你好真真"诱导词过滤器。

    Args:
        log_intercepted: 是否记录被拦截的词

    Returns:
        DecoyFilter 实例
    """
    return DecoyFilter(
        decoy_keywords=DECOY_KEYWORDS_NIHAO_ZHENZHEN,
        log_intercepted=log_intercepted,
    )
