"""
start_app.py - 启动脚本（带详细日志）
"""

import sys
import os

# 切换到脚本所在目录，确保相对路径正确解析
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 设置编码
sys.stdout.reconfigure(encoding='utf-8')

print("启动医疗RAG系统...")

try:
    from config import config
    print(f"配置加载成功: {config.DOC_FOLDER}")

    from src.logger import logger
    print("日志系统初始化成功")

    # 使用 fastapi_app 的惰性初始化，首次请求时自动加载
    from fastapi_app import app, get_rag
    import uvicorn

    # 预热：主动触发 RAG 初始化，避免第一个请求慢
    print("加载RAG引擎...")
    rag = get_rag()
    print(f"知识库加载完成: {len(rag.documents)}个文档片段")

    print(f"\n启动 FastAPI 服务器...")
    print(f"访问地址: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"API文档: http://{config.SERVER_HOST}:{config.SERVER_PORT}/docs")

    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)

except Exception as e:
    print(f"启动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
