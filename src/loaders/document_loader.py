"""
document_loader.py - 加载和预处理文档
"""

import os
import re
import hashlib
import joblib
from typing import List, Dict, Set
from src.logger import logger
from config import config


class DocumentLoader:
    """文档加载器"""

    def __init__(self):
        self.documents: List[Dict] = []

    def get_folder_hash(self, folder_path: str) -> str:
        """计算文件夹内容的哈希值，用于缓存检测"""
        if not os.path.exists(folder_path):
            return ""

        files_info = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                files_info.append(f"{filename}-{mtime}-{size}")

        return hashlib.md5("|".join(files_info).encode()).hexdigest()

    def load_from_folder(self, folder_path: str) -> List[Dict]:
        """
        从文件夹加载所有txt文档

        返回格式:
        [
            {
                "id": 0,
                "content": "文档内容片段",
                "source": "文件名.txt",
                "chunk_id": 0
            },
            ...
        ]
        """
        if not os.path.exists(folder_path):
            logger.error(f"文件夹不存在: {folder_path}")
            return []

        # 尝试从缓存加载
        cache_hash = self._compute_folder_hash(folder_path)
        cache_path = os.path.join(config.CACHE_FOLDER, f"documents_{cache_hash}.pkl")
        if os.path.exists(cache_path):
            try:
                logger.info("从缓存加载文档索引...")
                self.documents = joblib.load(cache_path)
                logger.info(f"缓存加载成功，共 {len(self.documents)} 个文档片段")
                return self.documents
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")

        documents = []
        doc_id = 0

        # 递归遍历所有子目录中的txt文件
        txt_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.endswith('.txt'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, folder_path)
                    txt_files.append((full_path, rel_path))

        for filepath, rel_path in txt_files:
            logger.info(f"加载文件: {rel_path}")

            try:
                # 读取文件内容
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 清洗文本
                content = self._clean_text(content)

                # 切分成段落
                chunks = self._split_into_chunks(content)

                # 创建文档片段
                for chunk_id, chunk in enumerate(chunks):
                    if chunk.strip():  # 跳过空段落
                        documents.append({
                            "id": doc_id,
                            "content": chunk.strip(),
                            "source": rel_path,
                            "chunk_id": chunk_id
                        })
                        doc_id += 1

            except Exception as e:
                logger.error(f"加载文件 {rel_path} 失败: {e}")

        # 精确去重（相同内容只保留一份）
        unique_docs = self._deduplicate(documents)

        removed = len(documents) - len(unique_docs)
        if removed:
            logger.info(f"去重移除 {removed} 个重复片段")

        logger.info(f"成功加载 {len(unique_docs)} 个文档片段")
        self.documents = unique_docs

        # 保存到缓存
        try:
            os.makedirs(config.CACHE_FOLDER, exist_ok=True)
            joblib.dump(unique_docs, cache_path)
            logger.info(f"文档索引已缓存到: {cache_path}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

        return unique_docs

    def _compute_folder_hash(self, folder_path: str) -> str:
        """计算文件夹内容哈希（仅用于缓存键，不拼接全文）"""
        files_info = []
        for root, dirs, files in os.walk(folder_path):
            for f in sorted(files):
                if f.endswith('.txt'):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, folder_path)
                    mtime = os.path.getmtime(full_path)
                    size = os.path.getsize(full_path)
                    files_info.append(f"{rel_path}-{mtime}-{size}")
        return hashlib.md5("|".join(files_info).encode()).hexdigest()[:16]

    @staticmethod
    def _deduplicate(documents: List[Dict]) -> List[Dict]:
        """精确去重：相同 content 只保留第一个"""
        seen: Set[str] = set()
        result = []
        for doc in documents:
            if doc['content'] not in seen:
                seen.add(doc['content'])
                result.append(doc)
        return result

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """文本归一化：全角→半角、不可见字符移除、标点统一"""
        # 全角字母数字转半角
        result = []
        for ch in text:
            code = ord(ch)
            if 0xFF01 <= code <= 0xFF5E:  # 全角字符范围
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:  # 全角空格→半角
                result.append(' ')
            else:
                result.append(ch)
        text = ''.join(result)

        # 去除零宽字符和不可见控制字符（基于 Unicode 分类判断）
        cleaned = []
        for ch in text:
            cp = ord(ch)
            # 保留：正常字符、汉字、标点、字母数字
            # 移除：零宽字符、控制字符、格式字符
            if cp in range(0x200B, 0x2010):   # 零宽空格 ~ 零宽非连接符
                continue
            if cp in range(0x2028, 0x2030):   # 行分隔符 ~ 窄不换行空格
                continue
            if cp in range(0x2060, 0x2065):   # 单词连接符 ~ 不可见加号
                continue
            if cp in (0xFEFF, 0x00AD):        # BOM / 软连字符
                continue
            if cp < 0x20 and cp not in (0x09, 0x0A, 0x0D):  # 控制字符（保留 tab/换行/回车）
                continue
            cleaned.append(ch)
        text = ''.join(cleaned)

        return text

    def _clean_text(self, text: str) -> str:
        """清洗文本：移除引用标记、多余空白、归一化等"""
        # Unicode 归一化
        text = self._normalize_unicode(text)

        # 移除引用标记 [1], (2) 等
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)

        # 合并多个换行
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def _split_into_chunks(self, text: str) -> List[str]:
        """将文本切分成有意义的段落块"""
        # 检测是否为 【第N条】 格式（医疗QA数据集）
        if re.search(r'【第\d+条】', text):
            return self._split_qa_format(text)

        # 常规段落：按两个换行符切分
        chunks = text.split('\n\n')
        if len(chunks) <= 2:
            chunks = text.split('\n')
        return chunks

    def _split_qa_format(self, text: str) -> List[str]:
        """
        处理 【第N条】 格式的 QA 数据集

        每条包含 问/答，保留为完整的一块，避免检索时只命中问题没有答案。
        注意：_normalize_unicode 已将全角冒号转为半角，需用半角匹配。
        """
        # 移除文件头（第一行的标题等）
        body = re.sub(r'^.*?【第1条】', '【第1条】', text, count=1, flags=re.DOTALL)

        # 按 【第N条】 拆分
        blocks = re.split(r'(?=【第\d+条】)', body)
        blocks = [b.strip() for b in blocks if b.strip()]

        if not blocks:
            return []

        chunks = []
        for block in blocks:
            # 兼容全角和半角冒号（_normalize_unicode 已将全角转半角）
            q_match = re.search(r'问[：:]\s*(.+)', block)
            a_match = re.search(r'答[：:]\s*(.+)', block)

            content_parts = []
            if q_match:
                content_parts.append(q_match.group(1).strip())
            if a_match:
                content_parts.append(a_match.group(1).strip())

            if content_parts:
                chunks.append(" | ".join(content_parts))

        # 如果 QA 格式提取失败，回退到按行切分
        if not chunks:
            chunks = [line.strip() for line in text.split('\n') if line.strip()]

        return chunks
