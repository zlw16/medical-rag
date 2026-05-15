"""
optimized_retriever.py - 优化的医学检索器
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import numpy as np

class OptimizedMedicalRetriever:
    """优化的医学领域检索器"""
    
    def __init__(self, documents):
        self.documents = documents
        self.corpus = [doc["content"] for doc in documents]
        
        # 优化的TF-IDF参数
        self.tfidf = TfidfVectorizer(
            tokenizer=self._tokenize,
            max_features=10000,      # 增加特征维度
            min_df=2,                # 降低最小文档频率
            max_df=0.85,             # 调整最大文档频率
            ngram_range=(1, 3),      # 支持1-3元语法
            analyzer="word",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True        # 亚线性TF缩放
        )
        
        # 优化的BM25参数
        tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(
            tokenized_corpus,
            k1=1.5,    # 调整文档频率影响
            b=0.75     # 调整文档长度影响
        )
        
        # 构建TF-IDF矩阵
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus)
        logger.info("优化的医学检索器初始化完成")
    
    def _tokenize(self, text):
        """医学文本分词"""
        from src.utils.medical_tokenizer import medical_tokenize
        return medical_tokenize(text)
    
    def search(self, query, top_k=5, weights=[0.3, 0.5, 0.2]):
        """
        优化的混合检索
        weights: [tfidf_weight, bm25_weight, semantic_weight]
        """
        query_tokens = self._tokenize(query)
        
        # TF-IDF检索
        tfidf_scores = self.tfidf.transform([query]).toarray()[0]
        
        # BM25检索
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 归一化分数
        tfidf_scores = self._normalize(tfidf_scores)
        bm25_scores = self._normalize(bm25_scores)
        
        # 加权融合
        combined_scores = (
            weights[0] * tfidf_scores +
            weights[1] * bm25_scores
        )
        
        # 获取Top-K结果
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0.001:  # 过滤低分数结果
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "score": float(combined_scores[idx])
                })
        
        return results
    
    def _normalize(self, scores):
        """Min-Max归一化"""
        if np.max(scores) == np.min(scores):
            return scores
        return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))