"""
logger.py - 日志记录模块
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "medical_rag",
                 log_dir: str = "./logs",
                 level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: 日志器名称
        log_dir: 日志文件目录
        level: 日志级别

    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 文件处理器 - 按日期命名
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(level)

    # 控制台处理器（Windows需显式指定utf-8编码）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    if os.name == 'nt':
        try:
            import sys
            console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except Exception:
            pass

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# 创建默认日志器
logger = setup_logger()
