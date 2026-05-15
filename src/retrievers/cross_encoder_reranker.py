"""
cross_encoder_reranker.py - Cross-Encoder精排模块
使用交叉编码器对检索结果进行精排，提升排序准确性
"""

from typing import List, Tuple, Dict, Optional
from sentence_transformers import CrossEncoder
import torch
from src.logger import logger
from config import config


class CrossEncoderReranker:
    """Cross-Encoder精排器"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._init_model()

    def _init_model(self):
        """初始化Cross-Encoder模型"""
        try:
            logger.info(f"加载Cross-Encoder模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Cross-Encoder模型加载完成，使用设备: {self.device}")
        except Exception as e:
            logger.warning(f"加载Cross-Encoder失败，将跳过精排: {e}")
            self.model = None

    def rerank(self, query: str, candidates: List[Tuple[str, Dict, float]], 
               top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        对候选文档进行精排

        Args:
            query: 用户查询
            candidates: 粗排后的候选文档列表 [(content, meta, score), ...]
            top_k: 返回前k个结果

        Returns:
            精排后的结果列表
        """
        if self.model is None or not candidates:
            return candidates[:top_k]

        try:
            # 准备输入对
            pairs = [[query, doc[0]] for doc in candidates]
            
            # 预测分数
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # 按分数排序（Cross-Encoder分数越高越好）
            scored_results = list(zip(candidates, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 返回精排后的结果
            final_results = []
            for candidate, score in scored_results[:top_k]:
                content, meta, _ = candidate
                final_results.append((content, meta, float(score)))
            
            logger.info(f"Cross-Encoder精排完成，输入{len(candidates)}个候选，输出{len(final_results)}个结果")
            return final_results

        except Exception as e:
            logger.error(f"Cross-Encoder精排失败，使用粗排结果: {e}")
            return candidates[:top_k]

    def batch_rerank(self, queries: List[str], candidates_list: List[List[Tuple[str, Dict, float]]],
                     top_k: int = 5) -> List[List[Tuple[str, Dict, float]]]:
        """
        批量精排

        Args:
            queries: 查询列表
            candidates_list: 每个查询对应的候选列表
            top_k: 返回前k个结果

        Returns:
            精排后的结果列表
        """
        results = []
        for query, candidates in zip(queries, candidates_list):
            results.append(self.rerank(query, candidates, top_k))
        return results
