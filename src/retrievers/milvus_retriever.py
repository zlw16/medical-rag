"""
milvus_retriever.py - Milvus 向量检索模块
使用 Milvus 进行语义向量检索，支持中文和多语言内容。
需要先启动 Milvus 服务（docker 或 cloud），未启动时自动降级。
"""

import os
import time
from typing import List, Dict, Tuple, Optional
from src.logger import logger
from config import config


class MilvusRetriever:
    """Milvus 向量检索器"""

    def __init__(self, documents: List[Dict],
                 host: str = "localhost",
                 port: str = "19530",
                 collection_name: str = "medical_rag",
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.documents = documents
        self.collection_name = collection_name
        self.model_name = model_name
        self.available = False
        self.collection = None
        self.model = None

        try:
            self._connect(host, port)
            self._load_model()
            self._init_collection()
        except Exception as e:
            logger.warning(f"Milvus 初始化失败，将跳过 Milvus 检索: {e}")

    def _connect(self, host: str, port: str):
        """连接 Milvus 服务"""
        from pymilvus import connections, utility

        connections.connect(host=host, port=port)
        if not utility.has_collection(self.collection_name):
            logger.info(f"Milvus 连接成功，集合 {self.collection_name} 不存在，将在插入时创建")
        else:
            logger.info(f"Milvus 连接成功，集合 {self.collection_name} 已存在")
        self.available = True

    def _load_model(self):
        """加载 embedding 模型（与 semantic_retriever 共享模型单例）"""
        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Milvus embedding 模型加载完成 ({self.model_name}), 维度: {self.embedding_dim}")

    def _init_collection(self):
        """初始化或加载集合"""
        from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility

        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"Milvus 集合已加载，实体数: {self.collection.num_entities}")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="医疗RAG知识库")
        self.collection = Collection(self.collection_name, schema)

        # 插入数据
        self._insert_documents()

        # 创建 IVF_FLAT 索引（平衡性能和资源）
        index_params = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}}
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        logger.info(f"Milvus 集合创建完成，插入 {len(self.documents)} 条数据")

    def _insert_documents(self):
        """将文档插入 Milvus"""
        logger.info("正在向 Milvus 插入文档向量...")
        batch_size = 256
        total = len(self.documents)

        for start in range(0, total, batch_size):
            batch = self.documents[start:start + batch_size]
            contents = [d["content"] for d in batch]
            sources = [d["source"] for d in batch]
            chunk_ids = [d["id"] for d in batch]

            embeddings = self.model.encode(contents, show_progress=False).tolist()
            self.collection.insert([embeddings, contents, sources, chunk_ids])

            if (start + batch_size) % 1024 == 0 or start + batch_size >= total:
                logger.info(f"Milvus 插入进度: {min(start + batch_size, total)}/{total}")

        self.collection.flush()

    def search(self, query: str, top_k: int = 50) -> List[Tuple[str, Dict, float]]:
        """
        语义向量检索

        Args:
            query: 用户查询
            top_k: 返回前 k 个结果

        Returns:
            结果列表: [(内容, 元数据, 分数), ...]
        """
        if not self.available or self.model is None or self.collection is None:
            return []

        try:
            start = time.time()
            query_vec = self.model.encode([query]).tolist()

            results = self.collection.search(
                data=query_vec,
                anns_field="embedding",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=top_k,
                output_fields=["content", "source", "chunk_id"],
            )

            hits = results[0]
            ret = []
            for hit in hits:
                ret.append((
                    hit.entity.get("content"),
                    {"source": hit.entity.get("source"), "id": hit.entity.get("chunk_id")},
                    float(hit.score),
                ))

            logger.debug(f"Milvus 检索完成: {len(ret)} 个结果, 耗时: {time.time() - start:.2f}s")
            return ret

        except Exception as e:
            logger.warning(f"Milvus 检索失败: {e}")
            return []

    def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            from pymilvus import connections
            connections.disconnect("default")
        except Exception:
            pass
