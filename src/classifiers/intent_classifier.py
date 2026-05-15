"""
intent_classifier.py - 意图分类器
"""

from typing import Tuple
from src.logger import logger


class IntentClassifier:
    """医疗意图分类器"""

    def __init__(self):
        # 紧急情况关键词
        self.emergency_keywords = [
            "呼吸困难", "胸痛", "昏迷", "大出血", "抽搐",
            "无法说话", "意识不清", "剧烈疼痛", "吐血",
            "休克", "窒息", "心脏骤停"
        ]

        # 用药咨询关键词
        self.medication_keywords = [
            "吃药", "剂量", "服用", "副作用", "禁忌",
            "怎么吃", "吃多少", "用药", "药品"
        ]

    def classify(self, text: str) -> Tuple[str, float]:
        """
        分类意图

        返回: (intent_type, confidence)
        intent_type: "emergency", "medication", "general"
        """
        text_lower = text.lower()

        # 检查紧急情况
        for kw in self.emergency_keywords:
            if kw in text_lower:
                logger.info(f"检测到紧急情况关键词: {kw}")
                return ("emergency", 0.95)

        # 检查用药咨询
        for kw in self.medication_keywords:
            if kw in text_lower:
                return ("medication", 0.80)

        # 普通咨询
        return ("general", 0.60)

    def get_emergency_response(self) -> str:
        """获取紧急情况的回复"""
        return (
            "🏥 紧急提醒！\n"
            "根据您的描述，这可能属于紧急医疗情况。\n"
            "请立即：\n"
            "1. 拨打120急救电话\n"
            "2. 或立即前往最近的医院急诊\n"
            "3. 在等待救援时保持患者平静\n\n"
            "⚠️ 本系统无法处理紧急情况，请立即就医！"
        )
