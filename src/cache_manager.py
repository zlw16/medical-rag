"""
cache_manager.py - 缓存管理模块
提供缓存过期、清理和管理功能
"""

import os
import time
import shutil
from typing import Optional
from src.logger import logger


class CacheManager:
    """缓存管理器"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def clean_old_cache(self, max_age_days: int = 7):
        """
        清理过期缓存

        Args:
            max_age_days: 缓存最大保留天数，默认为7天
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0

        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            
            if os.path.isfile(filepath):
                file_mtime = os.path.getmtime(filepath)
                
                if file_mtime < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                        logger.info(f"清理过期缓存: {filename}")
                    except Exception as e:
                        logger.error(f"删除缓存文件失败 {filename}: {e}")

        if cleaned_count > 0:
            logger.info(f"共清理 {cleaned_count} 个过期缓存文件")
        else:
            logger.info("没有过期缓存需要清理")

    def clear_all_cache(self):
        """清空所有缓存"""
        try:
            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
            logger.info("所有缓存已清空")
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")

    def get_cache_size(self) -> int:
        """
        获取缓存目录大小（字节）

        Returns:
            缓存目录大小（字节）
        """
        total_size = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
        return total_size

    def get_cache_size_human(self) -> str:
        """
        获取缓存目录大小（人类可读格式）

        Returns:
            缓存目录大小（如 "12.5 MB"）
        """
        size = self.get_cache_size()
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        else:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"

    def limit_cache_size(self, max_size_mb: int = 500):
        """
        限制缓存大小

        Args:
            max_size_mb: 最大缓存大小（MB），默认为500MB
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        current_size = self.get_cache_size()

        if current_size <= max_size_bytes:
            return

        logger.info(f"缓存大小 {self.get_cache_size_human()} 超过限制 {max_size_mb}MB，开始清理...")

        # 获取所有缓存文件及其修改时间
        files = []
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                files.append((filepath, os.path.getmtime(filepath)))

        # 按修改时间排序（最早的在前）
        files.sort(key=lambda x: x[1])

        # 删除最早的文件直到缓存大小低于限制
        for filepath, _ in files:
            try:
                file_size = os.path.getsize(filepath)
                os.remove(filepath)
                current_size -= file_size
                logger.info(f"删除缓存文件: {os.path.basename(filepath)}")
                
                if current_size <= max_size_bytes:
                    break
            except Exception as e:
                logger.error(f"删除缓存文件失败 {filepath}: {e}")

        logger.info(f"缓存清理完成，当前大小: {self.get_cache_size_human()}")

    def get_cache_info(self) -> dict:
        """
        获取缓存信息

        Returns:
            缓存信息字典
        """
        files = []
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                files.append({
                    'name': filename,
                    'size': os.path.getsize(filepath),
                    'mtime': os.path.getmtime(filepath)
                })

        return {
            'directory': self.cache_dir,
            'total_files': len(files),
            'total_size_bytes': self.get_cache_size(),
            'total_size_human': self.get_cache_size_human(),
            'files': files
        }
