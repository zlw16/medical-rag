"""
fastapi_app.py - FastAPI Web应用
功能与 flask_app.py 相同，但使用 FastAPI
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.rag_engine import MedicalRAG
from src.logger import logger
from config import config
import time

app = FastAPI(title="医疗RAG系统", version="1.0")

# 跨域支持（别人电脑访问时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 惰性初始化
_rag = None


def get_rag() -> MedicalRAG:
    global _rag
    if _rag is None:
        logger.info("首次请求触发RAG系统初始化...")
        _rag = MedicalRAG()
    return _rag


# 请求模型
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class ReloadResponse(BaseModel):
    doc_count: int
    chunk_count: int
    success: bool


# 主页
@app.get("/", response_class=HTMLResponse)
async def index():
    rag = get_rag()
    doc_count = len(set(doc['source'] for doc in rag.documents)) if rag.documents else 0
    chunk_count = len(rag.documents) if rag.documents else 0

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>医疗RAG系统</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; background: #f5f7fa; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; padding: 40px 20px; }}
            .header {{ text-align: center; margin-bottom: 40px; }}
            .header h1 {{ font-size: 28px; color: #1a73e8; margin-bottom: 8px; }}
            .header p {{ color: #666; }}
            .stats {{ display: flex; gap: 20px; justify-content: center; margin-bottom: 30px; }}
            .stat-card {{ background: white; border-radius: 12px; padding: 20px 30px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
            .stat-card .num {{ font-size: 32px; font-weight: bold; color: #1a73e8; }}
            .stat-card .label {{ font-size: 13px; color: #888; margin-top: 4px; }}
            .chat-box {{ background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); overflow: hidden; }}
            .messages {{ padding: 20px; max-height: 400px; overflow-y: auto; }}
            .msg {{ margin-bottom: 16px; padding: 12px 16px; border-radius: 8px; font-size: 14px; line-height: 1.6; }}
            .msg.user {{ background: #e3f2fd; margin-left: 40px; }}
            .msg.bot {{ background: #f5f5f5; margin-right: 40px; white-space: pre-wrap; }}
            .msg.bot .source {{ color: #1a73e8; font-size: 12px; }}
            .input-area {{ display: flex; border-top: 1px solid #eee; padding: 12px; gap: 8px; }}
            .input-area input {{ flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; outline: none; }}
            .input-area input:focus {{ border-color: #1a73e8; }}
            .input-area button {{ padding: 10px 24px; background: #1a73e8; color: white; border: none; border-radius: 8px; font-size: 14px; cursor: pointer; }}
            .input-area button:hover {{ background: #1557b0; }}
            .input-area button:disabled {{ background: #ccc; cursor: not-allowed; }}
            .loading {{ text-align: center; color: #888; padding: 10px; }}
            .error {{ color: #d32f2f; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>医疗RAG系统</h1>
                <p>基于默沙东诊疗手册的智能问答</p>
            </div>
            <div class="stats">
                <div class="stat-card">
                    <div class="num">{doc_count}</div>
                    <div class="label">医学文章</div>
                </div>
                <div class="stat-card">
                    <div class="num">{chunk_count}</div>
                    <div class="label">文档片段</div>
                </div>
            </div>
            <div class="chat-box">
                <div class="messages" id="messages">
                    <div class="msg bot">输入问题开始咨询，例如「高血压需要注意什么」</div>
                </div>
                <div class="input-area">
                    <input type="text" id="question" placeholder="请输入医学问题..."
                           onkeydown="if(event.key==='Enter') send()">
                    <button id="sendBtn" onclick="send()">发送</button>
                </div>
            </div>
        </div>
        <script>
            async function send() {{
                const input = document.getElementById('question');
                const msg = input.value.trim();
                if (!msg) return;

                const msgs = document.getElementById('messages');
                msgs.innerHTML += '<div class="msg user">' + msg + '</div>';
                input.value = '';
                document.getElementById('sendBtn').disabled = true;
                msgs.innerHTML += '<div class="msg bot loading">思考中...</div>';
                msgs.scrollTop = msgs.scrollHeight;

                try {{
                    const resp = await fetch('/api/query', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{question: msg}})
                    }});
                    const data = await resp.json();
                    msgs.querySelector('.loading').outerHTML =
                        '<div class="msg bot">' + (data.answer || data.error) + '</div>';
                }} catch(e) {{
                    msgs.querySelector('.loading').outerHTML =
                        '<div class="msg bot error">请求失败，请检查网络连接</div>';
                }}
                document.getElementById('sendBtn').disabled = false;
                msgs.scrollTop = msgs.scrollHeight;
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# 问答API
@app.post("/api/query")
async def api_query(req: QueryRequest):
    start_time = time.time()
    try:
        question = req.question.strip()
        logger.info(f"API查询: {question[:100]}")

        rag = get_rag()
        answer = rag.query(question)

        duration = time.time() - start_time
        logger.info(f"查询完成，耗时: {duration:.2f}s")

        return {"answer": answer, "question": question, "duration": round(duration, 2)}

    except Exception as e:
        logger.error(f"API错误: {e}")
        raise HTTPException(status_code=500, detail="服务内部错误")


# 重新加载知识库
@app.post("/api/reload")
async def api_reload():
    try:
        rag = get_rag()
        rag.load_knowledge_base()
        doc_count = len(set(doc['source'] for doc in rag.documents)) if rag.documents else 0
        chunk_count = len(rag.documents) if rag.documents else 0
        logger.info(f"知识库已重新加载: {doc_count}文档, {chunk_count}片段")
        return {"doc_count": doc_count, "chunk_count": chunk_count, "success": True}
    except Exception as e:
        logger.error(f"重新加载知识库失败: {e}")
        raise HTTPException(status_code=500, detail="重新加载失败")


# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}


# 启动入口
if __name__ == "__main__":
    import uvicorn
    print(f"启动FastAPI服务器...")
    print(f"访问地址: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"API文档: http://{config.SERVER_HOST}:{config.SERVER_PORT}/docs")
    uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT)
