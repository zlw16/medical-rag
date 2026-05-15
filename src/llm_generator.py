"""
llm_generator.py - 使用LLM生成答案
"""

from typing import List, Dict, Optional
import openai
from src.logger import logger
from config import config


# 检索分数门槛（RRF 分数通常很小，调整为更低的值以避免过滤有效结果）
MIN_RELEVANCE_SCORE = 0.0001


class LLMAnswerGenerator:
    """LLM答案生成器"""

    def __init__(self):
        self.api_key = config.DEEPSEEK_API_KEY
        self.base_url = config.DEEPSEEK_BASE_URL
        self.use_llm = config.USE_LLM

        if self.use_llm:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logger.info("LLM生成器已初始化")
        else:
            logger.warning("未配置API Key，将使用规则生成答案")

    def generate(self, question: str, contexts: List[Dict]) -> str:
        """
        基于检索结果生成答案

        Args:
            question: 用户问题
            contexts: 检索到的上下文列表，每个包含 'content' 和 'meta'

        Returns:
            生成的答案
        """
        if not self.use_llm:
            return self._generate_rule_based(question, contexts)

        # 评估检索质量：如果最高分太低，直接跳过 LLM
        max_score = max((ctx.get('score', 1) for ctx in contexts), default=1)
        logger.info(f"检索结果数量: {len(contexts)}, 最高分: {max_score:.6f}")
        if max_score < MIN_RELEVANCE_SCORE:
            logger.info(f"检索质量过低（最高分: {max_score:.6f}），跳过 LLM 生成")
            return self._generate_rule_based(question, contexts)

        try:
            # 构建prompt
            prompt = self._build_prompt(question, contexts)

            # 调用API（使用低温度减少幻觉）
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是一个严谨的医疗问答助手。你的回答必须严格基于用户提供的"
                        "医疗资料，绝不能使用你自身的知识来补充或编造信息。\n\n"
                        "规则：\n"
                        "1. 只能引用下方「医疗资料」中的内容\n"
                        "2. 如果医疗资料中不包含能回答问题的信息，请明确说"
                        "「根据现有资料，没有找到相关信息」\n"
                        "3. 不要做任何推断、联想或补充\n"
                        "4. 引用时标注来源，如 【来源：xxx.txt】\n"
                        "5. 末尾提醒：本信息仅供参考，不能替代专业医疗诊断"
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            logger.info("LLM答案生成成功")
            return answer

        except Exception as e:
            logger.error(f"LLM调用失败: {e}，使用规则生成答案")
            return self._generate_rule_based(question, contexts)

    def _build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """构建提示词"""
        context_str = "\n".join([
            f"【来源：{ctx['meta']['source']}】\n{ctx['content']}"
            for ctx in contexts
        ])

        prompt = f"""请严格基于以下医疗资料回答用户的问题。

医疗资料：
{context_str}

用户问题：{question}

注意：
- 如果资料中不包含相关信息，请回答「根据现有资料，没有找到相关信息」
- 不要添加资料中没有的信息
- 引用格式：【来源：文件名】
"""
        return prompt

    def _generate_rule_based(self, question: str, contexts: List[Dict]) -> str:
        """规则式答案生成（备用方案）——只展示最相关的一条结果"""
        if not contexts:
            return "根据现有知识库，没有找到相关信息。"

        # 只取最相关的一条
        best = contexts[0]
        content = best['content']
        source = best['meta']['source']

        # 如果答案包含问句，尝试只取答句部分
        answer_text = content
        if '|' in content:
            parts = content.split('|', 1)
            if len(parts) == 2:
                answer_text = parts[1].strip()

        # 截取过长内容
        if len(answer_text) > 300:
            answer_text = answer_text[:300] + "..."

        return (
            f"{answer_text}\n\n"
            f"---\n"
            f"📖 来源：{source}\n"
            f"💡 温馨提示：以上信息仅供参考，不能替代专业医疗诊断。\n"
            f"如果症状持续或加重，请及时就医咨询。"
        )
