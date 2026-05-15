"""
hybrid_retriever.py - TF-IDF + BM25 混合检索器
支持索引持久化缓存
"""

import os
import jieba
import hashlib
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from src.logger import logger
from config import config


def _jieba_tokenize(text):
    """jieba.cut 的模块级包装函数（可 pickle）"""
    return jieba.cut(text)


class HybridRetriever:
    """TF-IDF + BM25 混合检索器"""

    def __init__(self, documents: List[Dict],
                 tfidf_weight: float = 0.5,
                 bm25_weight: float = 0.5,
                 cache_dir: Optional[str] = None):
        self.documents = documents
        self.corpus = [doc['content'] for doc in documents]
        self.tfidf_weight = tfidf_weight
        self.bm25_weight = bm25_weight
        self.cache_dir = cache_dir or config.CACHE_FOLDER

        os.makedirs(self.cache_dir, exist_ok=True)

        # 尝试加载缓存或构建新索引
        cache_key = self._get_cache_key()
        cache_path = os.path.join(self.cache_dir, f"{self._get_cache_prefix()}{cache_key}.pkl")

        if os.path.exists(cache_path):
            logger.info("从缓存加载检索器...")
            self._load_from_cache(cache_path)
        else:
            logger.info("构建新的检索索引...")
            self._build_index()
            self._save_to_cache(cache_path)

    def _get_cache_prefix(self) -> str:
        """缓存文件名前缀（子类可覆盖以区分缓存）"""
        return "retriever_"

    def _get_cache_key(self) -> str:
        """生成缓存键"""
        corpus_str = "|".join(self.corpus)
        key_str = f"{corpus_str}-{self.tfidf_weight}-{self.bm25_weight}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _build_index(self):
        """构建检索索引"""
        if not self.corpus:
            logger.warning("语料库为空，跳过索引构建")
            self.vectorizer = None
            self.tfidf_matrix = None
            self.bm25 = None
            return

        # 构建TF-IDF索引
        logger.info("构建TF-IDF索引...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=_jieba_tokenize,
            token_pattern=None,
            max_features=5000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        logger.info(f"TF-IDF特征维度: {self.tfidf_matrix.shape[1]}")

        # 构建BM25索引
        logger.info("构建BM25索引...")
        tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"混合检索器构建完成，共 {len(self.corpus)} 个文档")

    def _save_to_cache(self, cache_path: str):
        """保存索引到缓存"""
        try:
            joblib.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'bm25': self.bm25,
                'documents': self.documents,
                'corpus': self.corpus,
                'tfidf_weight': self.tfidf_weight,
                'bm25_weight': self.bm25_weight
            }, cache_path)
            logger.info(f"检索器已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _load_from_cache(self, cache_path: str):
        """从缓存加载索引"""
        try:
            data = joblib.load(cache_path)
            self.vectorizer = data['vectorizer']
            # 恢复 tokenizer（兼容旧缓存）
            if not callable(getattr(self.vectorizer, 'tokenizer', None)):
                self.vectorizer.tokenizer = _jieba_tokenize
            self.tfidf_matrix = data['tfidf_matrix']
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.corpus = data['corpus']
            self.tfidf_weight = data.get('tfidf_weight', 0.5)
            self.bm25_weight = data.get('bm25_weight', 0.5)
            logger.info("检索器缓存加载成功")
        except Exception as e:
            logger.warning(f"缓存加载失败，将重新构建索引: {e}")
            self._build_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """混合检索"""
        if not self.corpus or self.vectorizer is None:
            return []

        # 1. TF-IDF检索
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # 2. BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 3. 归一化分数
        max_tfidf = max(tfidf_scores) if max(tfidf_scores) > 0 else 1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

        # 4. 计算混合分数
        hybrid_scores = []
        for i in range(len(self.corpus)):
            tfidf_norm = tfidf_scores[i] / max_tfidf
            bm25_norm = bm25_scores[i] / max_bm25

            final_score = (self.tfidf_weight * tfidf_norm +
                          self.bm25_weight * bm25_norm)

            hybrid_scores.append((i, final_score))

        # 5. 排序并取top_k
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = hybrid_scores[:top_k]

        # 6. 构建结果
        results = []
        for idx, score in top_results:
            if score > 0:
                doc = self.documents[idx]
                results.append((
                    doc['content'],
                    {'source': doc['source'], 'id': doc['id']},
                    score
                ))

        return results
