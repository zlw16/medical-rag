"""
enhanced_hybrid_retriever.py - 增强型混合检索器
整合：TF-IDF + BM25 + 语义检索 + 同义词扩展 + 查询改写 + 精排 + 路由
继承自 HybridRetriever，复用 TF-IDF/BM25 索引构建
"""

import os
import jieba
import hashlib
import joblib
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from src.logger import logger
from config import config
from .medical_synonyms import MedicalSynonymExpander
from .semantic_retriever import SemanticRetriever
from .query_rewriter import QueryRewriter
from .cross_encoder_reranker import CrossEncoderReranker
from .document_router import DocumentRouter
from .hybrid_retriever import HybridRetriever


class EnhancedHybridRetriever(HybridRetriever):
    """增强型混合检索器（继承 HybridRetriever 复用 TF-IDF/BM25）"""

    # RRF 常数
    RRF_K = 60

    def __init__(self, documents: List[Dict],
                 use_reranker: bool = True,
                 use_router: bool = True,
                 tfidf_weight: float = 0.3,
                 bm25_weight: float = 0.4,
                 semantic_weight: float = 0.3,
                 use_synonyms: bool = True,
                 use_query_rewrite: bool = True,
                 cache_dir: Optional[str] = None):
        # 在 super().__init__ 前设置增强模块的属性，
        # 因为父类 __init__ 会依次调用 _build_index / _load_from_cache
        self.semantic_weight = semantic_weight
        self.synonym_expander = MedicalSynonymExpander() if use_synonyms else None
        self.query_rewriter = QueryRewriter() if use_query_rewrite else None
        self.semantic_retriever = None
        
        # 精排器
        self.use_reranker = use_reranker
        self.reranker = CrossEncoderReranker() if use_reranker else None
        
        # 文档路由器
        self.use_router = use_router
        self.document_router = DocumentRouter() if use_router else None

        super().__init__(documents, tfidf_weight, bm25_weight, cache_dir)

    def _get_cache_prefix(self) -> str:
        """使用独立的缓存文件名前缀，区别于父类"""
        return "enhanced_retriever_"

    def _get_cache_key(self) -> str:
        """生成缓存键（含语义权重）"""
        corpus_str = "|".join(self.corpus)
        key_str = f"{corpus_str}-{self.tfidf_weight}-{self.bm25_weight}-{self.semantic_weight}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _build_index(self):
        """构建增强索引：TF-IDF + BM25（父类） + 语义索引"""
        # 父类构建 TF-IDF 和 BM25
        super()._build_index()

        # 构建语义索引
        if self.semantic_weight > 0:
            logger.info("构建语义索引...")
            self.semantic_retriever = SemanticRetriever(self.documents)
        else:
            logger.info("语义权重为0，跳过语义索引构建")
            self.semantic_retriever = None

        logger.info(f"增强检索器构建完成，共 {len(self.corpus)} 个文档")

    def _save_to_cache(self, cache_path: str):
        """保存索引到缓存（含语义嵌入）"""
        data = {
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'bm25': self.bm25,
            'documents': self.documents,
            'corpus': self.corpus,
            'tfidf_weight': self.tfidf_weight,
            'bm25_weight': self.bm25_weight,
            'semantic_weight': self.semantic_weight,
        }
        if self.semantic_retriever is not None and hasattr(self.semantic_retriever, 'corpus_embeddings'):
            data['corpus_embeddings'] = self.semantic_retriever.corpus_embeddings
            data['model_name'] = self.semantic_retriever.model_name
        try:
            joblib.dump(data, cache_path)
            logger.info(f"增强检索器已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _load_from_cache(self, cache_path: str):
        """从缓存加载索引（含语义嵌入恢复，避免重计算）"""
        try:
            data = joblib.load(cache_path)
            self.vectorizer = data['vectorizer']
            # 恢复 tokenizer（兼容旧缓存）
            if not callable(getattr(self.vectorizer, 'tokenizer', None)):
                from .hybrid_retriever import _jieba_tokenize
                self.vectorizer.tokenizer = _jieba_tokenize
            self.tfidf_matrix = data['tfidf_matrix']
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.corpus = data['corpus']
            self.tfidf_weight = data.get('tfidf_weight', 0.3)
            self.bm25_weight = data.get('bm25_weight', 0.4)
            self.semantic_weight = data.get('semantic_weight', 0.3)

            # 恢复语义检索器：尽量从缓存恢复嵌入，避免重计算
            if 'corpus_embeddings' in data and self.semantic_weight > 0:
                self.semantic_retriever = SemanticRetriever(
                    self.documents,
                    model_name=data.get('model_name', 'all-MiniLM-L6-v2'),
                    load_embeddings=False
                )
                self.semantic_retriever.corpus_embeddings = data['corpus_embeddings']
            elif self.semantic_weight > 0:
                # 旧缓存无嵌入，单独构建
                logger.info("缓存中无语义嵌入，单独构建语义索引...")
                self.semantic_retriever = SemanticRetriever(self.documents)
            else:
                self.semantic_retriever = None

            logger.info("增强检索器缓存加载成功")
        except Exception as e:
            logger.warning(f"缓存加载失败，将重新构建索引: {e}")
            self._build_index()
            self._save_to_cache(cache_path)

    def _expand_query(self, query: str):
        """
        扩展查询（同义词）

        Returns:
            (display_str, expanded_tokens) —— display 用于日志，tokens 用于 BM25
        """
        if not self.synonym_expander:
            return query, list(jieba.cut(query))

        # display 字符串（用于日志）
        display = self.synonym_expander.expand_to_boolean_query(query)

        # expanded_tokens：原始 token + 同义词（平铺，去重，用于 BM25）
        tokens = list(jieba.cut(query))
        token_set: set = set()
        expanded_tokens = []
        for token in tokens:
            if token and token not in token_set:
                token_set.add(token)
                expanded_tokens.append(token)
            synonym_list = self.synonym_expander.synonym_dict.get(token, [])
            for syn in synonym_list:
                if syn not in token_set:
                    token_set.add(syn)
                    expanded_tokens.append(syn)

        return display, expanded_tokens

    def _aggregate_results_rrf(self, results_list: List[List[Tuple[str, Dict, float]]],
                                top_k: int) -> List[Tuple[str, Dict, float]]:
        """
        使用 Reciprocal Rank Fusion (RRF) 聚合多路检索结果

        RRF 对排序位置敏感，比简单分数加权更鲁棒。
        """
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, Dict] = {}

        for results in results_list:
            for rank, (content, meta, score) in enumerate(results):
                doc_id = f"{meta['source']}-{meta['id']}"
                if doc_id not in doc_info:
                    doc_info[doc_id] = {'content': content, 'meta': meta}
                # RRF: rank 从 0 开始，所以 +1
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank + 1)

        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for doc_id, total_score in sorted_results[:top_k]:
            info = doc_info[doc_id]
            final_results.append((info['content'], info['meta'], total_score))

        return final_results

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        增强混合检索（带 RRF 聚合 + 精排）

        Args:
            query: 用户查询
            top_k: 返回前k个结果

        Returns:
            结果列表: [(内容, 元数据, 分数), ...]
        """
        if not self.corpus or self.vectorizer is None:
            return []

        logger.info(f"原始查询: {query}")
        start_time = time.time()

        # 1. 文档路由（根据意图筛选文档）
        routed_docs = self._route_documents(query)
        
        # 2. 查询扩展
        expanded_display, expanded_tokens = self._expand_query(query)
        logger.info(f"扩展查询: {expanded_display}")

        # 3. 各路检索
        results_list = []

        # TF-IDF 检索（使用原始查询，不扩展）
        try:
            query_vec = self.vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

            tfidf_results = []
            for i, score in enumerate(tfidf_scores):
                if score > 0.05:
                    doc = self.documents[i]
                    tfidf_results.append((doc['content'], {'source': doc['source'], 'id': doc['id']}, score))

            if tfidf_results:
                results_list.append(tfidf_results)
        except Exception as e:
            logger.error(f"TF-IDF检索失败: {e}")

        # BM25 检索（使用扩展后的 token 列表，无 "OR" 噪声）
        try:
            bm25_scores = self.bm25.get_scores(expanded_tokens)

            bm25_results = []
            for i, score in enumerate(bm25_scores):
                if score > 0:
                    doc = self.documents[i]
                    bm25_results.append((doc['content'], {'source': doc['source'], 'id': doc['id']}, score))

            if bm25_results:
                results_list.append(bm25_results)
        except Exception as e:
            logger.error(f"BM25检索失败: {e}")

        # 语义检索
        try:
            if self.semantic_retriever and self.semantic_weight > 0:
                semantic_results = self.semantic_retriever.search(query, top_k=top_k * 2)
                if semantic_results:
                    results_list.append(semantic_results)
        except Exception as e:
            logger.error(f"语义检索失败: {e}")

        if not results_list:
            return []

        # 4. RRF 聚合（粗排）
        aggregated = self._aggregate_results_rrf(results_list, top_k * 2)  # 多返回一些供精排
        
        # 5. Cross-Encoder 精排
        final_results = self._rerank(query, aggregated, top_k)
        
        elapsed = time.time() - start_time
        logger.info(f"检索完成，找到 {len(final_results)} 个结果，耗时: {elapsed:.2f}s")

        return final_results

    def _route_documents(self, query: str) -> List[Dict]:
        """
        根据查询意图路由文档

        Returns:
            路由后的文档子集
        """
        if self.use_router and self.document_router:
            return self.document_router.route(query, self.documents)
        return self.documents

    def _rerank(self, query: str, candidates: List[Tuple[str, Dict, float]], 
                top_k: int) -> List[Tuple[str, Dict, float]]:
        """
        使用 Cross-Encoder 精排

        Args:
            query: 用户查询
            candidates: 粗排后的候选文档
            top_k: 返回前k个结果

        Returns:
            精排后的结果
        """
        if self.use_reranker and self.reranker:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]

    def add_documents(self, new_docs: List[Dict]):
        """
        增量添加文档

        Args:
            new_docs: 新文档列表，每个文档包含 'content' 和 'source'
        """
        if not new_docs:
            return

        logger.info(f"增量添加 {len(new_docs)} 个文档")
        
        # 添加到文档列表
        self.documents.extend(new_docs)
        self.corpus.extend([doc['content'] for doc in new_docs])
        
        # 更新索引
        self._update_index_incremental(new_docs)
        
        # 更新缓存
        if self.cache_dir:
            cache_path = self._get_cache_path()
            self._save_to_cache(cache_path)
        
        logger.info(f"增量更新完成，当前文档数: {len(self.documents)}")

    def _update_index_incremental(self, new_docs: List[Dict]):
        """
        增量更新索引

        Args:
            new_docs: 新文档列表
        """
        new_corpus = [doc['content'] for doc in new_docs]
        
        # 更新 TF-IDF
        if self.vectorizer:
            try:
                # 增量更新词汇表（简化处理：重建）
                all_texts = self.corpus + new_corpus
                self.vectorizer.fit(all_texts)
                self.tfidf_matrix = self.vectorizer.transform(self.corpus)
                logger.info("TF-IDF索引增量更新完成")
            except Exception as e:
                logger.error(f"TF-IDF增量更新失败: {e}")
        
        # 更新 BM25
        if self.bm25:
            try:
                all_tokens = [list(jieba.cut(text)) for text in self.corpus]
                self.bm25 = BM25Okapi(all_tokens)
                logger.info("BM25索引增量更新完成")
            except Exception as e:
                logger.error(f"BM25增量更新失败: {e}")
        
        # 更新语义索引
        if self.semantic_retriever:
            try:
                self.semantic_retriever.add_documents(new_docs)
                logger.info("语义索引增量更新完成")
            except Exception as e:
                logger.error(f"语义索引增量更新失败: {e}")

    def remove_documents(self, doc_ids: List[str]):
        """
        删除指定文档

        Args:
            doc_ids: 要删除的文档ID列表
        """
        doc_id_set = set(doc_ids)
        initial_count = len(self.documents)
        
        # 过滤文档
        self.documents = [doc for doc in self.documents 
                          if f"{doc['source']}-{doc['id']}" not in doc_id_set]
        self.corpus = [doc['content'] for doc in self.documents]
        
        # 重建索引
        self._build_index()
        
        # 更新缓存
        if self.cache_dir:
            cache_path = self._get_cache_path()
            self._save_to_cache(cache_path)
        
        logger.info(f"删除了 {initial_count - len(self.documents)} 个文档，当前文档数: {len(self.documents)}")

    def async_search(self, query: str, top_k: int = 5):
        """
        异步检索（协程模式）

        Args:
            query: 用户查询
            top_k: 返回前k个结果

        Returns:
            协程对象
        """
        import asyncio
        
        async def search_coroutine():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.search, query, top_k)
        
        return search_coroutine()

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, Dict, float]]]:
        """
        批量检索

        Args:
            queries: 查询列表
            top_k: 返回前k个结果

        Returns:
            结果列表的列表
        """
        results = []
        for query in queries:
            results.append(self.search(query, top_k))
        return results
