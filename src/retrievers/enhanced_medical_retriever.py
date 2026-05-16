"""
enhanced_medical_retriever.py - 优化的医学领域检索器
检索流程：
1. 第一次召回：混合检索（TF-IDF + BM25 + 语义检索）召回50个片段
2. 重排序：使用Cross-Encoder精排
3. 相似度过滤：过滤极低相似度的片段
4. 基于大模型和知识库回答
"""

import os
import hashlib
import joblib
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
# CrossEncoder 延迟导入以避免启动时下载模型
from src.logger import logger
from config import config

# 导入优化组件
from .medical_expander import MedicalQueryExpander
from src.utils.medical_tokenizer import medical_tokenize, init_medical_tokenizer

# 标记是否已初始化分词器
_tokenizer_initialized = False

def _ensure_tokenizer_initialized():
    """确保医学分词器已初始化（延迟初始化）"""
    global _tokenizer_initialized
    if not _tokenizer_initialized:
        init_medical_tokenizer()
        _tokenizer_initialized = True


class EnhancedMedicalRetriever:
    """优化的医学领域检索器"""
    
    # RRF 常数（针对医学文本优化，较小的值会使分数差距更明显）
    RRF_K = 20
    
    # 检索参数配置
    FIRST_RECALL_COUNT = 80  # 第一次召回数量（从50提升到80）
    RERANK_COUNT = 45         # 最终返回数量（提升到45以覆盖低排名但相关片段）
    SIMILARITY_THRESHOLD = 0.01  # 相似度阈值（从0.2降到0.01）
    
    def __init__(self, documents: List[Dict],
                 bm25_weight: float = 0.5,
                 semantic_weight: float = 0.5,
                 use_synonyms: bool = True,
                 use_query_rewrite: bool = True,
                 use_reranker: bool = True,
                 use_milvus: bool = False,
                 cache_dir: Optional[str] = None):
        """
        参数说明：
        - bm25_weight: BM25关键词检索权重
        - semantic_weight: BGE语义检索权重
        - use_synonyms: 是否启用同义词扩展
        - use_query_rewrite: 是否启用查询改写
        - use_reranker: 是否启用BGE重排序
        - use_milvus: 是否启用Milvus向量检索
        """
        self.documents = documents
        self.corpus = [doc["content"] for doc in documents]
        self.bm25_weight = bm25_weight
        self.semantic_weight = semantic_weight
        self.use_reranker = use_reranker
        self.use_milvus = use_milvus

        # 初始化扩展器
        self.query_expander = MedicalQueryExpander() if use_synonyms else None
        self.use_query_rewrite = use_query_rewrite

        # 延迟初始化医学分词器
        _ensure_tokenizer_initialized()

        # 初始化检索组件
        self.vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None

        # Cross-Encoder重排序器
        self.reranker = None
        if self.use_reranker:
            try:
                # 延迟导入以避免启动时下载模型
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
                logger.info("Cross-Encoder重排序器初始化完成")
            except Exception as e:
                logger.warning(f"Cross-Encoder加载失败，将使用默认重排序: {e}")
                self.use_reranker = False

        # 缓存相关
        self.cache_dir = cache_dir or config.CACHE_FOLDER
        self.cache_key = self._compute_cache_key()

        # 语义检索器（延迟加载）
        self.semantic_retriever = None
        self.semantic_weight = semantic_weight

        # Milvus检索器（延迟加载）
        self.milvus_retriever = None

        # 构建或加载索引
        self._init_index()
    
    def _compute_cache_key(self) -> str:
        """计算缓存键（基于文档内容哈希）"""
        doc_str = "|".join(doc["content"][:100] for doc in self.documents[:100])
        return hashlib.md5(doc_str.encode()).hexdigest()
    
    def _init_index(self):
        """初始化索引（尝试从缓存加载）"""
        cache_path = os.path.join(self.cache_dir, f"medical_retriever_{self.cache_key}.pkl")
        
        if os.path.exists(cache_path):
            try:
                self._load_from_cache(cache_path)
                return
            except Exception as e:
                logger.warning(f"缓存加载失败，重新构建索引: {e}")
        
        # 构建新索引
        self._build_index()
        self._save_to_cache(cache_path)
    
    def _build_index(self):
        """构建优化的检索索引"""
        logger.info("构建优化的医学检索索引...")
        
        # 优化的TF-IDF配置
        self.vectorizer = TfidfVectorizer(
            tokenizer=medical_tokenize,
            max_features=10000,      # 增加特征维度
            min_df=2,                # 降低最小文档频率
            max_df=0.85,             # 调整最大文档频率
            ngram_range=(1, 3),      # 支持1-3元语法
            analyzer="word",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True        # 亚线性TF缩放
        )
        
        # 构建TF-IDF矩阵
        logger.info("构建TF-IDF索引...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info(f"TF-IDF特征维度: {self.tfidf_matrix.shape[1]}")
        
        # 优化的BM25配置
        logger.info("构建BM25索引...")
        tokenized_corpus = [medical_tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=1.5,    # 调整文档频率影响
            b=0.75     # 调整文档长度影响
        )
        
        logger.info(f"医学检索器构建完成，共 {len(self.documents)} 个文档")
    
    def _save_to_cache(self, cache_path: str):
        """保存索引到缓存"""
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            data = {
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'bm25': self.bm25,
                'documents': self.documents,
                'corpus': self.corpus,
                'bm25_weight': self.bm25_weight,
                'semantic_weight': self.semantic_weight
            }
            joblib.dump(data, cache_path)
            logger.info(f"医学检索器已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def _load_from_cache(self, cache_path: str):
        """从缓存加载索引"""
        try:
            data = joblib.load(cache_path)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.corpus = data['corpus']
            self.bm25_weight = data.get('bm25_weight', 0.45)
            self.semantic_weight = data.get('semantic_weight', 0.2)
            logger.info("医学检索器缓存加载成功")
        except Exception as e:
            raise e
    
    def _get_semantic_retriever(self):
        """延迟初始化语义检索器"""
        if self.semantic_retriever is None and self.semantic_weight > 0:
            try:
                from .semantic_retriever import SemanticRetriever
                logger.info("语义检索器初始化中...")
                self.semantic_retriever = SemanticRetriever(
                    self.documents,
                    model_name="BAAI/bge-small-zh-v1.5",
                    cache_dir=self.cache_dir
                )
                logger.info("语义检索器初始化完成")
            except Exception as e:
                logger.warning(f"语义检索器加载失败: {e}")
                self.semantic_retriever = False  # 标记为不可用
        return self.semantic_retriever if self.semantic_retriever else None

    def _get_milvus_retriever(self):
        """延迟初始化 Milvus 检索器"""
        if self.milvus_retriever is None and self.use_milvus:
            try:
                from .milvus_retriever import MilvusRetriever
                logger.info("Milvus 检索器初始化中...")
                self.milvus_retriever = MilvusRetriever(
                    self.documents,
                    host=config.MILVUS_HOST,
                    port=config.MILVUS_PORT,
                )
            except Exception as e:
                logger.warning(f"Milvus 检索器初始化失败: {e}")
                self.milvus_retriever = False
        return self.milvus_retriever if self.milvus_retriever else None

    def _expand_query(self, query: str) -> Tuple[str, str, List[str]]:
        """扩展查询（同义词 + 缩写展开）
        Returns:
            (expanded_display, expanded_text, expanded_tokens)
            - expanded_display: 用于日志
            - expanded_text: 用于TF-IDF检索
            - expanded_tokens: 用于BM25检索
        """
        if not self.query_expander:
            tokens = medical_tokenize(query)
            return query, query, tokens

        # 获取扩展查询字符串（用于日志）
        expanded_str = self.query_expander.expand_query(query)

        # 获取扩展tokens（用于BM25和TF-IDF）
        tokens = set(medical_tokenize(query))

        # 添加同义词
        for term in list(tokens):
            synonyms = self.query_expander.synonyms.get(term, [])
            tokens.update(synonyms)

        expanded_tokens = list(tokens)
        # 生成用于TF-IDF的扩展文本（用空格连接所有扩展词）
        expanded_text = " ".join(expanded_tokens)
        # 如果扩展后和原始查询差别不大，用原始查询
        if len(expanded_tokens) <= len(medical_tokenize(query)):
            expanded_text = query

        return expanded_str, expanded_text, expanded_tokens
    
    def _retrieve_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict, float]]:
        """TF-IDF检索"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 获取Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # 降低TF-IDF阈值，允许更多结果参与RRF融合
            if scores[idx] > 0.001:
                doc = self.documents[idx]
                results.append((doc["content"], doc, float(scores[idx])))
        
        return results
    
    def _retrieve_bm25(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, Dict, float]]:
        """BM25检索"""
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            # 降低BM25阈值，允许更多结果参与RRF融合
            if scores[idx] > 0.05:
                doc = self.documents[idx]
                results.append((doc["content"], doc, float(scores[idx])))
        
        return results
    
    def _aggregate_results_rrf(self, results_list: List[List[Tuple[str, Dict, float]]],
                                top_k: int) -> List[Tuple[str, Dict, float]]:
        """使用RRF聚合结果"""
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, Dict] = {}
        
        for results in results_list:
            for rank, (content, meta, score) in enumerate(results):
                doc_id = f"{meta['source']}-{meta['id']}"
                if doc_id not in doc_info:
                    doc_info[doc_id] = {'content': content, 'meta': meta}
                # RRF公式
                doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (self.RRF_K + rank + 1)
        
        # 排序
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, total_score in sorted_results[:top_k]:
            info = doc_info[doc_id]
            final_results.append((info['content'], info['meta'], total_score))
        
        return final_results
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        优化的医学检索接口
        
        检索流程：
        1. 第一次召回：混合检索（TF-IDF + BM25）召回10个片段
        2. 重排序：使用Cross-Encoder精排，选出3个最佳片段
        3. 相似度过滤：相似度<0.2的片段不召回
        
        Args:
            query: 用户查询
            top_k: 返回前K个结果（实际会被RERANK_COUNT覆盖）
        
        Returns:
            结果列表: [(内容, 元数据, 分数), ...]
        """
        if not self.corpus or self.vectorizer is None:
            return []
        
        logger.info(f"原始查询: {query}")
        start_time = time.time()

        # 查询扩展（返回: 显示用, TF-IDF用, BM25用）
        expanded_display, expanded_text, expanded_tokens = self._expand_query(query)
        logger.info(f"扩展查询: {expanded_display}")

        # ========== 步骤1: 第一次召回（双通道：BM25 + 语义）==========
        results_list = []

        # BM25检索（使用扩展tokens）
        bm25_results = self._retrieve_bm25(expanded_tokens, self.FIRST_RECALL_COUNT)
        if bm25_results:
            results_list.append(bm25_results)
            logger.info(f"BM25检索: {len(bm25_results)} 个结果")

        # 语义检索（如果启用）
        semantic_retriever = self._get_semantic_retriever()
        if semantic_retriever:
            try:
                semantic_results = semantic_retriever.search(query, self.FIRST_RECALL_COUNT)
                if semantic_results:
                    results_list.append(semantic_results)
                    logger.info(f"语义检索: {len(semantic_results)} 个结果")
            except Exception as e:
                logger.warning(f"语义检索失败: {e}")

        # Milvus向量检索（如果启用）
        milvus_retriever = self._get_milvus_retriever()
        if milvus_retriever:
            try:
                milvus_results = milvus_retriever.search(query, self.FIRST_RECALL_COUNT)
                if milvus_results:
                    results_list.append(milvus_results)
                    logger.info(f"Milvus检索: {len(milvus_results)} 个结果")
            except Exception as e:
                logger.warning(f"Milvus检索失败: {e}")

        # 如果启用了查询改写，添加改写查询的检索结果
        if self.use_query_rewrite and self.query_expander:
            rewrites = self.query_expander.rewrite_query(query)
            for rewrite in rewrites[:3]:
                rewrite_results = self._retrieve_tfidf(rewrite, int(self.FIRST_RECALL_COUNT / 2))
                if rewrite_results:
                    results_list.append(rewrite_results)
        
        # RRF聚合得到第一次召回的10个结果
        if results_list:
            first_recall = self._aggregate_results_rrf(results_list, self.FIRST_RECALL_COUNT)
        else:
            first_recall = []
        
        logger.info(f"第一次召回完成: {len(first_recall)} 个片段")
        
        if not first_recall:
            duration = time.time() - start_time
            logger.info(f"检索完成，找到 0 个结果，耗时: {duration:.2f}s")
            return []
        
        # ========== 步骤2: 重排序（使用Cross-Encoder选出3个）==========
        if self.use_reranker and self.reranker and len(first_recall) > 1:
            # 准备Cross-Encoder输入
            pairs = [(query, content) for content, meta, score in first_recall]

            # 预测相似度分数（返回raw logits）
            raw_scores = self.reranker.predict(pairs)
            raw_scores = np.array(raw_scores)

            # 对raw logits做z-score标准化，保留相对区分度
            mean_s = np.mean(raw_scores)
            std_s = np.std(raw_scores) or 1.0
            norm_scores = (raw_scores - mean_s) / std_s
            # 用sigmoid映射到[0,1]作为相似度信号，但z-score先做标准化再sigmoid
            # 比直接sigmoid(raw_logit)有更好的区分度
            scaled_scores = 1.0 / (1.0 + np.exp(-norm_scores))

            # 融合分数：RRF分占0.4，CE标准化分占0.6
            reranked = []
            for i, (content, meta, score) in enumerate(first_recall):
                combined_score = score * 0.4 + scaled_scores[i] * 0.6
                reranked.append((content, meta, combined_score, float(scaled_scores[i])))

            # 按融合分数排序，取前RERANK_COUNT个
            reranked.sort(key=lambda x: x[2], reverse=True)
            reranked_results = reranked[:self.RERANK_COUNT]

            logger.info(f"Cross-Encoder重排序完成: {len(reranked_results)} 个片段")
        else:
            # 不使用重排序时，直接取前RERANK_COUNT个
            # 注意：当没有Cross-Encoder时，使用原始分数作为相似度分数的替代
            # 但RRF分数范围较小，所以我们使用一个合理的默认相似度值
            reranked_results = []
            for content, meta, score in first_recall[:self.RERANK_COUNT]:
                # 对于非重排序结果，使用原始分数的归一化值作为相似度估计
                normalized_score = min(score * 10, 1.0)  # 放大分数以适应阈值
                reranked_results.append((content, meta, score, normalized_score))
        
        # ========== 步骤3: 相似度过滤（相似度<0.2的不召回）==========
        final_results = []
        for content, meta, combined_score, similarity_score in reranked_results:
            if similarity_score >= self.SIMILARITY_THRESHOLD:
                final_results.append((content, meta, combined_score))
                logger.debug(f"保留片段 (相似度: {similarity_score:.4f})")
            else:
                logger.debug(f"过滤片段 (相似度: {similarity_score:.4f} < {self.SIMILARITY_THRESHOLD})")
        
        duration = time.time() - start_time
        logger.info(f"检索完成，最终找到 {len(final_results)} 个结果，耗时: {duration:.2f}s")
        
        return final_results


# 测试
if __name__ == "__main__":
    from src.loaders.document_loader import DocumentLoader
    
    # 加载文档
    loader = DocumentLoader()
    docs = loader.load_from_folder("medical_knowledge")
    print(f"加载了 {len(docs)} 个文档")
    
    # 创建检索器
    retriever = EnhancedMedicalRetriever(
        docs,
        bm25_weight=0.5,
        semantic_weight=0.5,
        use_synonyms=True,
        use_query_rewrite=True
    )
    
    # 测试查询
    test_queries = [
        "高血压需要注意什么",
        "感冒怎么办",
        "布洛芬怎么吃"
    ]
    
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"查询: {q}")
        results = retriever.search(q, top_k=3)
        print(f"找到 {len(results)} 个结果:")
        
        for i, (content, meta, score) in enumerate(results):
            print(f"\n结果{i+1} (分数:{score:.4f}):")
            print(f"来源: {meta['source']}")
            print(f"内容: {content[:150]}...")