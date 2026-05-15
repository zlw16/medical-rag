"""
query_rewriter.py - LLM查询改写模块
使用LLM将用户问题改写为多种表达方式
"""

from typing import List, Optional
from src.logger import logger
from config import config
import openai


class QueryRewriter:
    """查询改写器"""

    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = config.DEEPSEEK_BASE_URL
        self.use_llm = config.USE_LLM

        if self.use_llm:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info("查询改写器已初始化")
        else:
            logger.warning("未配置API Key，将使用简单改写")

    def rewrite(self, query: str, num_variants: int = 3) -> List[str]:
        """
        将用户查询改写为多种表达方式
        
        Args:
            query: 用户原始查询
            num_variants: 生成的改写数量
        
        Returns:
            改写后的查询列表（包含原始查询）
        """
        if not self.use_llm:
            return self._simple_rewrite(query, num_variants)

        try:
            prompt = self._build_rewrite_prompt(query, num_variants)
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗问题改写助手。请将用户的医疗问题改写成多种不同的表达方式，便于检索。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            rewrites = self._parse_rewrites(result)
            
            # 确保包含原始查询
            if query not in rewrites:
                rewrites.insert(0, query)
            
            logger.info(f"查询改写: {query} -> {rewrites}")
            return rewrites

        except Exception as e:
            logger.error(f"LLM查询改写失败，使用简单改写: {e}")
            return self._simple_rewrite(query, num_variants)

    def _build_rewrite_prompt(self, query: str, num_variants: int) -> str:
        """构建改写提示词"""
        return f"""请将以下医疗问题改写成 {num_variants} 种不同的表达方式，要求：

1. 保持原意不变
2. 使用不同的词汇和句式
3. 涵盖更多相关术语
4. 每行一个改写

原始问题：{query}

改写结果：
"""

    def _parse_rewrites(self, result: str) -> List[str]:
        """解析LLM返回的改写结果"""
        rewrites = []
        
        for line in result.split('\n'):
            line = line.strip()
            # 移除编号（如 "1. "、"- "等）
            if line.startswith(('1.', '2.', '3.', '-', '*')):
                line = line[2:].strip()
            if line and len(line) > 3:
                rewrites.append(line)
        
        return rewrites[:5]  # 最多返回5个

    def _simple_rewrite(self, query: str, num_variants: int) -> List[str]:
        """简单改写（备用方案）"""
        rewrites = [query]
        
        # 添加一些简单变体
        variations = [
            f"什么是{query}",
            f"{query}怎么办",
            f"{query}的症状",
            f"{query}的治疗方法",
            f"{query}是什么原因"
        ]
        
        for var in variations[:num_variants]:
            if var not in rewrites and var != query:
                rewrites.append(var)
        
        return rewrites

    def expand_with_keywords(self, query: str) -> List[str]:
        """
        扩展查询关键词
        
        Args:
            query: 用户查询
        
        Returns:
            扩展后的关键词列表
        """
        if not self.use_llm:
            # 简单关键词提取
            import jieba
            words = list(jieba.cut(query))
            return list(set(words))

        try:
            prompt = f"""请提取以下医疗问题中的关键术语，并列出相关的同义词和相关术语：

问题：{query}

输出格式：以逗号分隔的术语列表
"""
            
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的医疗术语提取助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            result = response.choices[0].message.content.strip()
            keywords = [k.strip() for k in result.split(',') if k.strip()]
            
            logger.info(f"关键词提取: {query} -> {keywords}")
            return keywords

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            import jieba
            return list(set(jieba.cut(query)))
