"""
main.py - 系统主入口
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_engine import MedicalRAG
from src.logger import logger


def main():
    print("-" * 50)
    print("    医疗RAG系统 - 健康助手")
    print("    基于TF-IDF + BM25混合检索")
    print("-" * 50)

    # 检查知识库是否存在
    if not os.path.exists("./medical_knowledge"):
        os.makedirs("./medical_knowledge")
        print("[INFO] 已创建 medical_knowledge 文件夹")
        print("[INFO] 请在该文件夹中添加医疗知识文档（.txt格式）")
        print("\n示例内容：")
        print("-" * 40)
        print("【疾病】普通感冒")
        print("【症状】鼻塞、流涕、咳嗽")
        print("【治疗】多休息、多喝水")
        print("-" * 40)
        return

    try:
        # 启动RAG系统
        rag = MedicalRAG()

        # 检查是否有文档
        if not rag.documents:
            print("\n[WARN] 知识库为空！")
            print("请在 medical_knowledge 文件夹中添加 .txt 格式的医疗文档")
            print("\n示例文档内容：")
            print("【疾病】普通感冒")
            print("【症状】鼻塞、流涕、咳嗽、咽痛")
            print("【治疗】多休息、多喝水、对症治疗")
            return

        # 启动交互模式
        rag.interactive_mode()

    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"[ERROR] 系统启动失败: {e}")


if __name__ == "__main__":
    main()
