"""
config.py - 系统配置
使用 python-dotenv 从环境变量加载配置
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


class Config:
    # 检索配置
    TOP_K = int(os.getenv("TOP_K", 5))
    TFIDF_WEIGHT = float(os.getenv("TFIDF_WEIGHT", 0.3))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.4))
    SEMANTIC_WEIGHT = float(os.getenv("SEMANTIC_WEIGHT", 0.3))
    
    # 增强检索配置
    USE_SYNONYMS = os.getenv("USE_SYNONYMS", "true").lower() == "true"
    USE_QUERY_REWRITE = os.getenv("USE_QUERY_REWRITE", "true").lower() == "true"
    USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"  # 默认启用重排序器
    USE_ROUTER = os.getenv("USE_ROUTER", "false").lower() == "true"      # 默认关闭

    # 文档配置
    DOC_FOLDER = os.getenv("DOC_FOLDER", "./medical_knowledge")
    CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
    LOG_FOLDER = os.getenv("LOG_FOLDER", "./logs")

    # LLM配置
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    USE_LLM = bool(DEEPSEEK_API_KEY)

    # Flask配置
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5001))


# 创建配置实例
config = Config()
