"""
ml_intent_classifier.py - 基于机器学习的意图分类器
使用 TF-IDF + Logistic Regression 从 QA 数据自动训练
"""

import os
import re
import jieba
import joblib
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.logger import logger
from config import config

# ── 停用词（中文无实义词 + 医疗问诊常见语气词） ──
STOP_WORDS = frozenset({
    "的", "了", "是", "在", "有", "不", "就", "都", "也", "很",
    "会", "要", "还", "这", "那", "我", "你", "他", "它", "她",
    "吗", "呢", "啊", "吧", "哦", "嘛", "嗯", "哈", "呀", "哦",
    "请问", "想", "咨询", "一下", "了解", "知道", "告诉",
    "可以", "应该", "需要", "能够", "可能", "怎么", "什么",
    "如何", "为啥", "为什么", "哪个", "哪些", "哪种", "这个",
    "那个", "还是", "或者", "没有", "不是", "就是", "但是", "而且",
    "比较", "已经", "一直", "经常", "有时", "每次", "现在",
})

# ── 意图类别 ──
INTENT_NAMES = {
    "emergency":   "紧急情况",
    "symptom":     "症状咨询",
    "treatment":   "治疗方案",
    "medication":  "用药咨询",
    "prevention":  "预防保健",
    "examination": "检查诊断",
    "general":     "普通咨询",
}

# ── 自动标注关键词规则 ──
# 优先级从高到低，emergency 触发即归入 emergency
LABEL_RULES: List[Tuple[str, List[str]]] = [
    ("emergency",   ["呼吸困难", "胸痛", "昏迷", "大出血", "抽搐", "无法说话",
                     "意识不清", "剧烈疼痛", "吐血", "休克", "窒息", "心脏骤停",
                     "120", "急救", "猝死"]),
    ("medication",  ["吃药", "用药", "剂量", "服用", "药品", "怎么吃", "吃多少",
                     "副作用", "禁忌", "口服", "外敷", "用药咨询", "什么药"]),
    ("treatment",   ["怎么治", "治疗方法", "如何治疗", "治疗方案", "怎么治疗",
                     "治疗", "治愈", "治得好", "能治", "疗效"]),
    ("symptom",     ["症状", "表现", "征兆", "怎么办", "怎么回事", "是什么原因",
                     "什么原因", "怎么回事", "疼", "痛", "发热", "发烧",
                     "咳嗽", "流鼻涕", "头晕", "头痛", "呕吐", "腹泻",
                     "浑身", "全身", "酸疼", "乏力"]),
    ("prevention",  ["预防", "饮食", "注意", "保养", "注意事项", "如何预防",
                     "预防措施", "锻炼", "生活习惯", "忌口", "食疗", "养生"]),
    ("examination", ["检查", "化验", "诊断", "检测", "体检", "CT", "X光",
                     "血常规", "B超", "核磁", "心电图", "胸透", "胃镜"]),
]


def _tokenize(text: str) -> List[str]:
    """jieba 分词 + 去停用词（模块级函数，保证可 pickle）"""
    return [w for w in jieba.cut(text) if w.strip() and w not in STOP_WORDS]


class MLIntentClassifier:
    """基于 ML 的意图分类器（TF-IDF + Logistic Regression）"""

    def __init__(self, documents: Optional[List[Dict]] = None):
        self.pipeline: Optional[Pipeline] = None
        self._load_or_train(documents)

    # ── 自动标注 ──

    @staticmethod
    def _rule_label(text: str) -> str:
        """基于关键词规则标注意图（也用作模型 fallback）"""
        for intent, keywords in LABEL_RULES:
            for kw in keywords:
                if kw in text:
                    return intent
        return "general"

    def _auto_label(self, documents: List[Dict]) -> Tuple[List[str], List[str]]:
        """从 QA 数据中提取问题并自动标注意图"""
        texts, labels = [], []
        for doc in documents:
            content = doc.get("content", "")
            # QA 格式："问 | 答"，取问题部分
            question = content.split(" | ")[0].strip()
            if not question:
                continue
            texts.append(question)
            labels.append(self._rule_label(question))
        return texts, labels

    # ── 训练 / 加载 ──

    def _load_or_train(self, documents: Optional[List[Dict]]):
        """从缓存加载模型或重新训练"""
        cache_path = os.path.join(config.CACHE_FOLDER, "intent_classifier.joblib")

        # 尝试从缓存加载
        if os.path.exists(cache_path):
            try:
                self.pipeline = joblib.load(cache_path)
                logger.info("意图分类器从缓存加载")
                return
            except Exception as e:
                logger.warning(f"意图分类器缓存加载失败: {e}")

        # 需要训练
        if not documents or len(documents) < 50:
            logger.info("训练数据不足，使用规则分类（无 ML 模型）")
            return

        texts, labels = self._auto_label(documents)
        unique_labels = set(labels)
        logger.info(f"自动标注完成：{len(texts)} 条，类别分布："
                    f"{ {l: labels.count(l) for l in sorted(unique_labels)} }")

        if len(unique_labels) < 2:
            logger.warning("标注类别不足，跳过 ML 训练")
            return

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                tokenizer=_tokenize,
                max_features=3000,
                min_df=3,
                ngram_range=(1, 2),
            )),
            ("clf", LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
            )),
        ])

        # 训练—分词已在 tokenizer 中完成
        X = texts
        y = labels
        self.pipeline.fit(X, y)
        logger.info("意图分类器 ML 模型训练完成")

        # 缓存模型
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            joblib.dump(self.pipeline, cache_path)
            logger.info(f"意图分类器已缓存到 {cache_path}")
        except Exception as e:
            logger.warning(f"意图分类器缓存失败: {e}")

    # ── 分类接口 ──

    def classify(self, text: str) -> Tuple[str, float]:
        """
        分类意图

        Returns:
            (intent_type, confidence)
            intent_type: "emergency", "symptom", "treatment",
                         "medication", "prevention", "examination", "general"
        """
        if self.pipeline is not None:
            try:
                probs = self.pipeline.predict_proba([text])[0]
                pred = self.pipeline.predict([text])[0]
                confidence = float(max(probs))

                # 紧急情况：阈值放低以避免漏报
                if pred == "emergency":
                    return (pred, max(confidence, 0.6))

                if confidence >= 0.45:
                    return (pred, round(confidence, 3))
            except Exception as e:
                logger.warning(f"ML 分类失败，回退规则: {e}")

        # 规则 fallback
        intent = self._rule_label(text)
        confidence = 0.8 if intent == "emergency" else 0.65
        return (intent, confidence)

    def get_emergency_response(self) -> str:
        """紧急情况回复"""
        return (
            "⚠️ 紧急提醒！\n"
            "根据您的描述，这可能属于紧急医疗情况。\n"
            "请立即：\n"
            "1. 拨打120急救电话\n"
            "2. 或立即前往最近的医院急诊\n"
            "3. 在等待救援时保持患者平静\n\n"
            "⚠️ 本系统无法处理紧急情况，请立即就医！"
        )
