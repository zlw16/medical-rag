"""
retrieval_evaluator.py - 检索评估模块
用于评估检索质量，支持多种指标
"""

from typing import List, Dict, Tuple
from collections import defaultdict
import json
import os
from datetime import datetime
from src.logger import logger


class RetrievalEvaluator:
    """检索评估器"""

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or "./logs"
        self.metrics_history = defaultdict(list)
        os.makedirs(self.log_dir, exist_ok=True)

    def _calculate_mrr(self, predictions: List[List[str]], references: List[List[str]]) -> float:
        """
        计算 Mean Reciprocal Rank (MRR)

        Args:
            predictions: 预测结果列表，每个查询的检索结果ID列表
            references: 参考结果列表，每个查询的正确答案ID列表

        Returns:
            MRR值
        """
        total_mrr = 0.0
        count = 0

        for pred_list, ref_list in zip(predictions, references):
            if not ref_list:
                continue

            ref_set = set(ref_list)
            for rank, doc_id in enumerate(pred_list, 1):
                if doc_id in ref_set:
                    total_mrr += 1.0 / rank
                    break
            count += 1

        return total_mrr / count if count > 0 else 0.0

    def _calculate_recall(self, predictions: List[List[str]], references: List[List[str]], 
                          k: int = 5) -> float:
        """
        计算 Recall@k

        Args:
            predictions: 预测结果列表
            references: 参考结果列表
            k: 召回位置

        Returns:
            Recall@k值
        """
        total_recall = 0.0
        count = 0

        for pred_list, ref_list in zip(predictions, references):
            if not ref_list:
                continue

            ref_set = set(ref_list)
            pred_set = set(pred_list[:k])
            overlap = len(ref_set & pred_set)
            total_recall += overlap / len(ref_set)
            count += 1

        return total_recall / count if count > 0 else 0.0

    def _calculate_precision(self, predictions: List[List[str]], references: List[List[str]],
                             k: int = 5) -> float:
        """
        计算 Precision@k

        Args:
            predictions: 预测结果列表
            references: 参考结果列表
            k: 位置

        Returns:
            Precision@k值
        """
        total_precision = 0.0
        count = 0

        for pred_list, ref_list in zip(predictions, references):
            if not pred_list:
                continue

            ref_set = set(ref_list)
            pred_set = set(pred_list[:k])
            overlap = len(ref_set & pred_set)
            total_precision += overlap / min(k, len(pred_list))
            count += 1

        return total_precision / count if count > 0 else 0.0

    def evaluate(self, predictions: List[List[str]], references: List[List[str]],
                 k_list: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        评估检索结果

        Args:
            predictions: 预测结果列表
            references: 参考结果列表
            k_list: 评估位置列表

        Returns:
            评估指标字典
        """
        results = {}

        # 计算 MRR
        mrr = self._calculate_mrr(predictions, references)
        results['MRR'] = mrr
        self.metrics_history['MRR'].append(mrr)
        logger.info(f"MRR: {mrr:.4f}")

        # 计算 Recall@k
        for k in k_list:
            recall = self._calculate_recall(predictions, references, k)
            key = f"Recall@{k}"
            results[key] = recall
            self.metrics_history[key].append(recall)
            logger.info(f"{key}: {recall:.4f}")

        # 计算 Precision@k
        for k in k_list:
            precision = self._calculate_precision(predictions, references, k)
            key = f"Precision@{k}"
            results[key] = precision
            self.metrics_history[key].append(precision)
            logger.info(f"{key}: {precision:.4f}")

        # 保存历史记录
        self._save_history(results)

        return results

    def evaluate_single(self, query: str, predictions: List[Tuple[str, Dict, float]],
                        references: List[str]) -> Dict[str, float]:
        """
        评估单个查询

        Args:
            query: 查询文本
            predictions: 检索结果
            references: 参考文档ID列表

        Returns:
            评估指标字典
        """
        pred_ids = [f"{p[1]['source']}-{p[1]['id']}" for p in predictions]
        return self.evaluate([pred_ids], [references])

    def _save_history(self, results: Dict[str, float]):
        """保存评估历史"""
        history_file = os.path.join(self.log_dir, "evaluation_history.json")
        
        try:
            # 读取历史
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = []

            # 添加新记录
            record = {
                'timestamp': datetime.now().isoformat(),
                **results
            }
            history.append(record)

            # 保存
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存评估历史失败: {e}")

    def get_history(self) -> List[Dict]:
        """获取评估历史"""
        history_file = os.path.join(self.log_dir, "evaluation_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def report(self) -> str:
        """生成评估报告"""
        history = self.get_history()
        if not history:
            return "暂无评估记录"

        report = ["检索评估报告"]
        report.append("=" * 40)

        # 最新一次评估
        latest = history[-1]
        report.append(f"评估时间: {latest['timestamp']}")
        report.append("")

        # 指标汇总
        metrics = ['MRR', 'Recall@1', 'Recall@3', 'Recall@5', 'Precision@1', 'Precision@3', 'Precision@5']
        for metric in metrics:
            if metric in latest:
                report.append(f"{metric}: {latest[metric]:.4f}")

        # 趋势分析
        report.append("")
        report.append("趋势分析:")
        for metric in metrics:
            values = [h.get(metric) for h in history if metric in h]
            if len(values) >= 2:
                trend = "↑" if values[-1] > values[-2] else "↓" if values[-1] < values[-2] else "→"
                report.append(f"  {metric}: {values[-1]:.4f} ({trend})")

        return "\n".join(report)
