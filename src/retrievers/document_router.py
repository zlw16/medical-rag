"""
document_router.py - 文档路由模块
根据查询意图将查询路由到不同的文档子集
"""

from typing import List, Dict, Optional
from src.classifiers.intent_classifier import IntentClassifier
from src.logger import logger


class DocumentRouter:
    """文档路由器"""

    def __init__(self):
        self.classifier = IntentClassifier()
        
        # 意图到文档类型的映射
        self.intent_routing = {
            'symptom': ['症状', '症状描述', '临床表现'],
            'disease': ['疾病', '病症', '病'],
            'treatment': ['治疗', '疗法', '手术', '药物治疗'],
            'medication': ['药物', '药品', '用药', '服药'],
            'prevention': ['预防', '保健', '护理'],
            'diagnosis': ['诊断', '检查', '检测'],
            'general': []  # 通用，使用全部文档
        }

    def route(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        根据查询意图路由到文档子集

        Args:
            query: 用户查询
            documents: 全部文档列表

        Returns:
            路由后的文档子集
        """
        # 识别意图
        intent, confidence = self.classifier.classify(query)
        logger.info(f"查询意图: {intent} (置信度: {confidence:.2f})")

        # 根据意图选择文档
        if intent in self.intent_routing:
            keywords = self.intent_routing[intent]
            
            if not keywords:  # general意图
                return documents
            
            # 过滤相关文档
            routed_docs = []
            for doc in documents:
                source = doc.get('source', '').lower()
                content = doc.get('content', '').lower()
                
                # 检查是否包含相关关键词
                matched = False
                for keyword in keywords:
                    if keyword.lower() in source or keyword.lower() in content:
                        matched = True
                        break
                
                if matched:
                    routed_docs.append(doc)
            
            # 如果过滤后文档太少，返回全部文档
            if len(routed_docs) < 10:
                logger.info(f"路由后文档太少({len(routed_docs)}个)，使用全部文档")
                return documents
            
            logger.info(f"路由完成，从{len(documents)}个文档筛选出{len(routed_docs)}个相关文档")
            return routed_docs
        
        return documents

    def get_intent_statistics(self, documents: List[Dict]) -> Dict[str, int]:
        """
        获取各意图类别的文档统计

        Args:
            documents: 全部文档列表

        Returns:
            统计字典
        """
        stats = {intent: 0 for intent in self.intent_routing}
        stats['other'] = 0

        for doc in documents:
            source = doc.get('source', '').lower()
            content = doc.get('content', '').lower()
            matched = False

            for intent, keywords in self.intent_routing.items():
                if intent == 'general':
                    continue
                
                for keyword in keywords:
                    if keyword.lower() in source or keyword.lower() in content:
                        stats[intent] += 1
                        matched = True
                        break
            
            if not matched:
                stats['other'] += 1

        return stats
