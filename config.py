"""
config.py - 系统配置
使用 python-dotenv 从环境变量加载配置
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    # 检索配置（BM25关键词 + BGE语义双通道）
    TOP_K = int(os.getenv("TOP_K", 10))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.5))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.5))
    
    # 增强检索配置
    USE_SYNONYMS = os.getenv("USE_SYNONYMS", "true").lower() == "true"
    USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "true").lower() == "true"
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"  # 多语言Cross-Encoder
    USE_ROUTER = os.getenv("USE_ROUTER", "false").lower() == "true"

    # Milvus向量检索配置
    USE_MILVUS = os.getenv("USE_MILVUS", "false").lower() == "true"
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

    # 文档配置
    DOC_FOLDER = os.getenv("DOC_FOLDER", "./medical_knowledge")
    CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
    LOG_FOLDER = os.getenv("LOG_FOLDER", "./logs")

    # LLM配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    USE_LLM = bool(DEEPSEEK_API_KEY)

    # 服务器配置
    SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
    SERVER_PORT = int(os.getenv("SERVER_PORT", 5001))


# 创建配置实例
config = Config()
