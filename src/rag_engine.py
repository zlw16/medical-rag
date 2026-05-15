"""
rag_engine.py - RAG主引擎
"""

from typing import List, Dict, Optional
from src.loaders.document_loader import DocumentLoader
from src.retrievers.enhanced_medical_retriever import EnhancedMedicalRetriever
from src.classifiers.ml_intent_classifier import MLIntentClassifier
from src.llm_generator import LLMAnswerGenerator
from src.logger import logger
from config import config


class MedicalRAG:
    """医疗RAG系统主类"""

    def __init__(self, doc_folder: str = None):
        self.doc_folder = doc_folder or config.DOC_FOLDER
        self.documents: List[Dict] = []
        self.retriever: Optional[EnhancedMedicalRetriever] = None
        self.loader = DocumentLoader()
        self.llm_generator = LLMAnswerGenerator()
        self.classifier = MLIntentClassifier()
        self._classifier_trained = False

        # 加载知识库
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """加载知识库"""
        logger.info("正在加载知识库...")
        try:
            self.documents = self.loader.load_from_folder(self.doc_folder)

            if self.documents:
                # 使用优化的医学检索器
                self.retriever = EnhancedMedicalRetriever(
                    self.documents,
                    tfidf_weight=config.TFIDF_WEIGHT,
                    bm25_weight=config.BM25_WEIGHT,
                    semantic_weight=config.SEMANTIC_WEIGHT,
                    use_synonyms=config.USE_SYNONYMS,
                    use_query_rewrite=config.USE_QUERY_REWRITE,
                    use_reranker=config.USE_RERANKER  # 从配置读取重排序器设置
                )

                # 用加载的文档训练 ML 意图分类器
                self.classifier = MLIntentClassifier(self.documents)
                self._classifier_trained = True
                logger.info("知识库加载完成（增强型检索器）")
            else:
                logger.warning("未找到知识文档")

        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            self.documents = []

    def query(self, question: str) -> str:
        """处理用户问题"""
        if not question or not question.strip():
            return "请输入有效的问题。"

        logger.info(f"收到查询: {question}")

        try:
            # 1. 意图分类
            intent, confidence = self.classifier.classify(question)
            logger.info(f"识别意图: {intent} (置信度: {confidence:.2f})")

            # 2. 紧急情况处理
            if intent == "emergency" and confidence > 0.7:
                return self.classifier.get_emergency_response()

            # 3. 检索相关文档
            if not self.retriever or not self.documents:
                return "知识库未加载，请先添加医疗文档。"

            results = self.retriever.search(question, top_k=config.TOP_K)

            if not results:
                return "根据现有知识库，我无法找到与您问题相关的医学资料。建议咨询专业医生。"

            # 4. 构建答案（含检索分数，供 LLM 评估检索质量）
            contexts = [
                {'content': content, 'meta': meta, 'score': score}
                for content, meta, score in results
            ]
            answer = self.llm_generator.generate(question, contexts)

            return answer

        except Exception as e:
            logger.error(f"处理查询失败: {e}")
            return "抱歉，处理您的问题时出现了错误，请稍后再试。"

    def interactive_mode(self):
        """交互模式"""
        print("\n" + "=" * 60)
        print("[医疗RAG系统]")
        print("输入问题开始咨询，输入 'q' 退出")
        print("=" * 60)

        while True:
            try:
                question = input("\n请输入您的问题: ").strip()

                if question.lower() in ['q', 'quit', 'exit', '退出']:
                    print("再见！祝您健康！")
                    break

                if not question:
                    continue

                answer = self.query(question)
                print(f"\n助手:\n{answer}")

            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"交互模式错误: {e}")
                print("出错了，请稍后再试")
