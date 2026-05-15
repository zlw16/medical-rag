"""
test_basic.py - 基础测试用例
"""

import sys
import os
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 意图分类器测试
# ============================================================
class TestIntentClassifier:
    def setup_method(self):
        from src.classifiers.intent_classifier import IntentClassifier
        self.classifier = IntentClassifier()

    def test_emergency(self):
        intent, confidence = self.classifier.classify("我呼吸困难")
        assert intent == "emergency"
        assert confidence > 0.9

    def test_medication(self):
        intent, confidence = self.classifier.classify("这个药怎么吃")
        assert intent == "medication"

    def test_general(self):
        intent, confidence = self.classifier.classify("感冒怎么办")
        assert intent == "general"

    def test_empty_input(self):
        intent, confidence = self.classifier.classify("")
        assert intent == "general"

    def test_emergency_response_not_empty(self):
        response = self.classifier.get_emergency_response()
        assert "120" in response


# ============================================================
# 文档加载器测试
# ============================================================
class TestDocumentLoader:
    def setup_method(self):
        from src.loaders.document_loader import DocumentLoader
        self.loader = DocumentLoader()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def _create_doc(self, filename, content):
        path = os.path.join(self.temp_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_load_single_file(self):
        self._create_doc("感冒.txt", "【疾病】普通感冒\n【症状】鼻塞、流涕")
        docs = self.loader.load_from_folder(self.temp_dir)
        assert len(docs) > 0
        assert docs[0]['source'] == "感冒.txt"

    def test_load_multiple_files(self):
        self._create_doc("感冒.txt", "【疾病】普通感冒")
        self._create_doc("高血压.txt", "【疾病】高血压")
        docs = self.loader.load_from_folder(self.temp_dir)
        assert len(docs) == 2

    def test_ignore_non_txt(self):
        self._create_doc("感冒.txt", "内容")
        path = os.path.join(self.temp_dir, "notes.md")
        with open(path, 'w') as f:
            f.write("# Markdown")
        docs = self.loader.load_from_folder(self.temp_dir)
        assert all(doc['source'].endswith('.txt') for doc in docs)

    def test_clean_text_removes_citations(self):
        text = "症状包括头痛[1]。"
        cleaned = self.loader._clean_text(text)
        assert "头痛" in cleaned
        assert "[1]" not in cleaned

    def test_empty_folder_returns_empty(self):
        docs = self.loader.load_from_folder(self.temp_dir)
        assert docs == []

    def test_nonexistent_folder_returns_empty(self):
        docs = self.loader.load_from_folder("/nonexistent/path")
        assert docs == []


# ============================================================
# 混合检索器测试
# ============================================================
class TestHybridRetriever:
    def setup_method(self):
        from src.retrievers.hybrid_retriever import HybridRetriever

        self.documents = [
            {"id": 0, "content": "感冒需要多休息多喝水", "source": "感冒.txt", "chunk_id": 0},
            {"id": 1, "content": "高血压患者应该少吃盐", "source": "高血压.txt", "chunk_id": 0},
            {"id": 2, "content": "发烧时可以服用布洛芬", "source": "感冒.txt", "chunk_id": 1},
        ]

        self.temp_dir = tempfile.mkdtemp()
        self.retriever = HybridRetriever(
            self.documents,
            tfidf_weight=0.5,
            bm25_weight=0.5,
            cache_dir=self.temp_dir
        )

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_search_returns_results(self):
        results = self.retriever.search("感冒怎么办", top_k=3)
        assert len(results) > 0

    def test_search_top_k(self):
        results = self.retriever.search("感冒", top_k=2)
        assert len(results) <= 2

    def test_relevant_results_ranked_first(self):
        results = self.retriever.search("感冒", top_k=3)
        top_content = results[0][0] if results else ""
        assert "感冒" in top_content

    def test_empty_corpus_returns_empty(self):
        from src.retrievers.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever([], cache_dir=self.temp_dir)
        results = retriever.search("感冒")
        assert results == []


# ============================================================
# RAG 引擎测试
# ============================================================
class TestMedicalRAG:
    def setup_method(self):
        from src.rag_engine import MedicalRAG

        # 用临时知识库目录避免污染真实数据
        self.temp_dir = tempfile.mkdtemp()
        self.old_doc_folder = os.environ.get("DOC_FOLDER")
        os.environ["DOC_FOLDER"] = self.temp_dir
        os.environ["USE_SYNONYMS"] = "false"
        os.environ["USE_QUERY_REWRITE"] = "false"
        os.environ["SEMANTIC_WEIGHT"] = "0"  # 关闭语义检索加速测试

        # 创建一个测试文档
        doc_path = os.path.join(self.temp_dir, "测试.txt")
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("【疾病】测试疾病\n【症状】测试症状\n【治疗】测试治疗")

        self.rag = MedicalRAG(doc_folder=self.temp_dir)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)
        if self.old_doc_folder:
            os.environ["DOC_FOLDER"] = self.old_doc_folder
        else:
            del os.environ["DOC_FOLDER"]
        del os.environ["USE_SYNONYMS"]
        del os.environ["USE_QUERY_REWRITE"]
        del os.environ["SEMANTIC_WEIGHT"]

    def test_knowledge_base_loaded(self):
        assert len(self.rag.documents) > 0

    def test_query_returns_response(self):
        answer = self.rag.query("测试疾病")
        assert answer is not None
        assert len(answer) > 0

    def test_empty_query_returns_prompt(self):
        answer = self.rag.query("")
        assert "请输入" in answer or "有效" in answer

    def test_emergency_query_returns_emergency(self):
        answer = self.rag.query("呼吸困难")
        assert "120" in answer or "急救" in answer


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
