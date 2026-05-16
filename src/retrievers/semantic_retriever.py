"""
semantic_retriever.py - 语义检索模块
使用Sentence-Transformers进行语义匹配
"""

import os
import joblib
import hashlib
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
from src.logger import logger
from config import config

# 全局模型单例（避免重复加载几百MB的模型）
_model_instance = None
_model_name = None


class SemanticRetriever:
    """语义检索器（基于Sentence-Transformers）"""

    def __init__(self, documents: List[Dict], model_name: str = "BAAI/bge-small-zh-v1.5",
                 cache_dir: Optional[str] = None, load_embeddings: bool = True):
        global _model_instance, _model_name

        self.documents = documents
        self.corpus = [doc['content'] for doc in documents]
        self.model_name = model_name
        self.cache_dir = cache_dir or config.CACHE_FOLDER

        os.makedirs(self.cache_dir, exist_ok=True)

        # 语义模型单例
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if _model_instance is None or _model_name != model_name:
            logger.info(f"加载语义模型: {model_name} (设备: {self.device})")
            _model_instance = SentenceTransformer(model_name, device=self.device)
            _model_name = model_name
        else:
            logger.debug("使用缓存的语义模型")

        self.model = _model_instance

        if not load_embeddings:
            self.corpus_embeddings = []
            return

        # 尝试加载缓存或计算嵌入
        cache_key = self._get_cache_key()
        cache_path = os.path.join(self.cache_dir, f"semantic_{cache_key}.pkl")

        if os.path.exists(cache_path):
            logger.info("从缓存加载语义索引...")
            self._load_from_cache(cache_path)
        else:
            logger.info("计算文档语义嵌入...")
            self._build_index()
            self._save_to_cache(cache_path)

    def _get_cache_key(self) -> str:
        """生成缓存键"""
        corpus_str = "|".join(self.corpus)
        key_str = f"{corpus_str}-{self.model_name}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _build_index(self):
        """构建语义索引"""
        if self.model is None:
            self.corpus_embeddings = []
            return

        # 批量处理以节省内存
        batch_size = 32
        embeddings = []

        batches = range(0, len(self.corpus), batch_size)
        for i in tqdm(batches, desc="编码文档嵌入", unit="batch"):
            batch = self.corpus[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True)
            embeddings.append(batch_embeddings)
            logger.debug(f"处理批次 {i // batch_size + 1}/{(len(self.corpus) + batch_size - 1) // batch_size}")

        if embeddings:
            self.corpus_embeddings = torch.cat(embeddings)
        else:
            self.corpus_embeddings = []

        logger.info(f"语义索引构建完成，共 {len(self.corpus)} 个文档")

    def _save_to_cache(self, cache_path: str):
        """保存索引到缓存"""
        try:
            joblib.dump({
                'corpus_embeddings': self.corpus_embeddings,
                'documents': self.documents,
                'corpus': self.corpus
            }, cache_path)
            logger.info(f"语义索引已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _load_from_cache(self, cache_path: str):
        """从缓存加载索引"""
        try:
            data = joblib.load(cache_path)
            self.corpus_embeddings = data['corpus_embeddings']
            self.documents = data['documents']
            self.corpus = data['corpus']
            logger.info("语义索引缓存加载成功")
        except Exception as e:
            logger.warning(f"缓存加载失败，将重新计算: {e}")
            self._build_index()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """
        语义检索

        Args:
            query: 用户查询
            top_k: 返回前k个结果

        Returns:
            结果列表: [(内容, 元数据, 相似度分数), ...]
        """
        if self.model is None or not self.corpus:
            return []

        try:
            # 编码查询
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # 计算相似度（余弦相似度）
            if len(self.corpus_embeddings) == 0:
                return []

            cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

            # 获取top_k结果
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus)))

            # 构建结果
            results = []
            for score, idx in zip(top_results.values, top_results.indices):
                score_float = float(score)
                if score_float > 0.1:  # 过滤低相似度结果
                    doc = self.documents[idx]
                    results.append((
                        doc['content'],
                        {'source': doc['source'], 'id': doc['id']},
                        score_float
                    ))

            return results

        except Exception as e:
            logger.error(f"语义检索失败: {e}")
            return []

    def add_documents(self, new_docs: List[Dict]):
        """
        增量添加文档

        Args:
            new_docs: 新文档列表
        """
        if not new_docs:
            return

        logger.info(f"语义检索器增量添加 {len(new_docs)} 个文档")
        
        # 添加到文档列表
        self.documents.extend(new_docs)
        new_corpus = [doc['content'] for doc in new_docs]
        self.corpus.extend(new_corpus)
        
        # 计算新文档的嵌入并追加
        if self.model:
            new_embeddings = self.model.encode(new_corpus, convert_to_tensor=True)
            
            if hasattr(self.corpus_embeddings, 'shape') and self.corpus_embeddings.shape[0] > 0:
                self.corpus_embeddings = torch.cat([self.corpus_embeddings, new_embeddings])
            else:
                self.corpus_embeddings = new_embeddings
        
        logger.info(f"语义检索器增量更新完成，当前文档数: {len(self.documents)}")
