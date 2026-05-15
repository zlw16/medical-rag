"""
test_retriever.py - 检索器单元测试
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrievers.enhanced_hybrid_retriever import EnhancedHybridRetriever
from src.retrievers.cross_encoder_reranker import CrossEncoderReranker
from src.retrievers.document_router import DocumentRouter
from src.evaluator.retrieval_evaluator import RetrievalEvaluator


# 测试数据
SAMPLE_DOCUMENTS = [
    {'content': '感冒是一种常见的上呼吸道感染疾病，主要症状包括流鼻涕、打喷嚏、喉咙痛等', 'source': 'symptom.txt', 'id': '1'},
    {'content': '治疗感冒的常用药物有对乙酰氨基酚、布洛芬等', 'source': 'medication.txt', 'id': '2'},
    {'content': '发烧是身体的一种防御反应，体温超过37.5℃称为发热', 'source': 'symptom.txt', 'id': '3'},
    {'content': '高血压患者需要长期规律服药，控制血压在正常范围', 'source': 'disease.txt', 'id': '4'},
    {'content': '预防感冒的方法包括勤洗手、保持室内通风、接种流感疫苗等', 'source': 'prevention.txt', 'id': '5'},
]


class TestEnhancedHybridRetriever:
    """增强型混合检索器测试"""

    @pytest.fixture
    def retriever(self):
        """创建测试用检索器"""
        return EnhancedHybridRetriever(
            documents=SAMPLE_DOCUMENTS,
            use_reranker=False,  # 禁用精排加速测试
            use_router=False,    # 禁用路由简化测试
            cache_dir=None       # 禁用缓存
        )

    def test_search_basic(self, retriever):
        """测试基本检索功能"""
        results = retriever.search("感冒", top_k=3)
        assert len(results) > 0
        assert all(len(res) == 3 for res in results)  # (content, meta, score)

    def test_search_empty_query(self, retriever):
        """测试空查询"""
        results = retriever.search("", top_k=3)
        assert len(results) == 0

    def test_search_top_k(self, retriever):
        """测试返回数量限制"""
        results = retriever.search("感冒", top_k=2)
        assert len(results) <= 2

    def test_search_no_results(self, retriever):
        """测试无结果查询"""
        results = retriever.search("宇宙飞船", top_k=3)
        assert len(results) == 0

    def test_add_documents(self, retriever):
        """测试增量添加文档"""
        initial_count = len(retriever.documents)
        new_docs = [
            {'content': '新型冠状病毒肺炎的主要症状包括发热、干咳、乏力等', 'source': 'disease.txt', 'id': '6'}
        ]
        retriever.add_documents(new_docs)
        assert len(retriever.documents) == initial_count + 1

    def test_remove_documents(self, retriever):
        """测试删除文档"""
        initial_count = len(retriever.documents)
        retriever.remove_documents(["symptom.txt-1"])
        assert len(retriever.documents) == initial_count - 1


class TestCrossEncoderReranker:
    """Cross-Encoder精排器测试"""

    def test_reranker_initialization(self):
        """测试精排器初始化"""
        reranker = CrossEncoderReranker()
        # 模型加载可能失败（无网络），所以只检查对象创建成功
        assert reranker is not None

    def test_rerank_empty_candidates(self):
        """测试空候选列表"""
        reranker = CrossEncoderReranker()
        results = reranker.rerank("test", [], top_k=5)
        assert len(results) == 0


class TestDocumentRouter:
    """文档路由器测试"""

    def test_router_initialization(self):
        """测试路由器初始化"""
        router = DocumentRouter()
        assert router is not None

    def test_route_basic(self):
        """测试基本路由功能"""
        router = DocumentRouter()
        result = router.route("感冒症状", SAMPLE_DOCUMENTS)
        assert len(result) > 0

    def test_get_statistics(self):
        """测试统计功能"""
        router = DocumentRouter()
        stats = router.get_intent_statistics(SAMPLE_DOCUMENTS)
        assert isinstance(stats, dict)


class TestRetrievalEvaluator:
    """检索评估器测试"""

    def test_mrr_calculation(self):
        """测试MRR计算"""
        evaluator = RetrievalEvaluator()
        predictions = [["doc1", "doc2", "doc3"], ["doc4"]]
        references = [["doc2"], ["doc4"]]
        results = evaluator.evaluate(predictions, references)
        assert 'MRR' in results
        assert 0 <= results['MRR'] <= 1

    def test_recall_calculation(self):
        """测试Recall计算"""
        evaluator = RetrievalEvaluator()
        predictions = [["doc1", "doc2", "doc3"]]
        references = [["doc2", "doc4"]]
        results = evaluator.evaluate(predictions, references)
        assert 'Recall@1' in results
        assert 'Recall@3' in results

    def test_precision_calculation(self):
        """测试Precision计算"""
        evaluator = RetrievalEvaluator()
        predictions = [["doc1", "doc2", "doc3"]]
        references = [["doc2"]]
        results = evaluator.evaluate(predictions, references)
        assert 'Precision@1' in results
        assert 'Precision@3' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
