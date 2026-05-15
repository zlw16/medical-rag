"""
flask_app.py - Flask Web应用
优化：添加异常处理、日志记录、生产配置
"""

from flask import Flask, render_template, request, jsonify, session, render_template_string
from src.rag_engine import MedicalRAG
from src.logger import logger
from config import config
import json
from datetime import datetime
from functools import wraps
import time

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# 惰性初始化RAG系统（避免 import 时阻塞）
_rag = None


def get_rag() -> MedicalRAG:
    """获取RAG实例（惰性加载）"""
    global _rag
    if _rag is None:
        logger.info("首次请求触发RAG系统初始化...")
        _rag = MedicalRAG()
    return _rag

# 请求限流装饰器
def rate_limit(limit: int = 10, per: int = 60):
    """
    简单的限流装饰器

    Args:
        limit: 限制次数
        per: 时间窗口（秒）
    """
    request_history = {}

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()

            if client_ip not in request_history:
                request_history[client_ip] = []

            # 清理过期请求
            request_history[client_ip] = [
                t for t in request_history[client_ip]
                if now - t < per
            ]

            if len(request_history[client_ip]) >= limit:
                logger.warning(f"请求限流触发: {client_ip}")
                return jsonify({'error': '请求过于频繁，请稍后再试'}), 429

            request_history[client_ip].append(now)
            return f(*args, **kwargs)

        return wrapped
    return decorator

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>医疗RAG助手</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        /* 侧边栏 */
        .sidebar {
            width: 280px;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            padding: 20px;
            overflow-y: auto;
        }

        .sidebar h2 {
            color: #667eea;
            margin-bottom: 20px;
        }

        .sidebar h3 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .doc-stats {
            background: #f0f0f0;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }

        .btn {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .btn-primary {
            background: #667eea;
            color: white;
        }

        .btn-primary:hover {
            background: #5a67d8;
        }

        .btn-danger {
            background: #e53e3e;
            color: white;
        }

        .btn-danger:hover {
            background: #c53030;
        }

        /* 主聊天区 */
        .chat-area {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #f7f7f7;
        }

        .chat-header {
            background: white;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .chat-header h1 {
            color: #333;
            font-size: 24px;
        }

        .chat-header p {
            color: #666;
            margin-top: 5px;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }

        .message {
            margin-bottom: 20px;
            display: flex;
            animation: fadeIn 0.3s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }

        .message.user .message-content {
            background: #667eea;
            color: white;
        }

        .message.assistant .message-content {
            background: white;
            color: #333;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }

        .input-area {
            background: white;
            padding: 20px;
            border-top: 1px solid #e0e0e0;
        }

        .input-container {
            display: flex;
            gap: 10px;
        }

        #question-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
        }

        #question-input:focus {
            border-color: #667eea;
        }

        .send-btn {
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
        }

        .send-btn:hover {
            background: #5a67d8;
        }

        .send-btn:disabled {
            background: #999;
            cursor: not-allowed;
        }

        .typing-indicator {
            background: white;
            padding: 10px 16px;
            border-radius: 18px;
            display: inline-block;
        }

        .typing-indicator span {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #999;
            margin: 0 2px;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-10px); opacity: 1; }
        }

        .disclaimer {
            text-align: center;
            padding: 10px;
            font-size: 12px;
            color: #999;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h2>🏥 医疗RAG助手</h2>
            <div class="doc-stats">
                <strong>📚 知识库状态</strong><br>
                文档数: <span id="doc-count">{{ doc_count }}</span><br>
                片段数: <span id="chunk-count">{{ chunk_count }}</span>
            </div>
            <button class="btn btn-primary" onclick="reloadKB()">🔄 重新加载知识库</button>
            <button class="btn btn-danger" onclick="clearChat()">🗑️ 清空对话</button>

            <h3>📋 使用提示</h3>
            <ul style="margin-left: 20px; color: #666;">
                <li>输入医疗相关问题</li>
                <li>系统会从知识库检索</li>
                <li>基于资料生成回答</li>
                <li>仅供参考，请遵医嘱</li>
            </ul>

            <h3>💡 示例问题</h3>
            <ul style="margin-left: 20px; color: #666;">
                <li onclick="setQuestion('感冒发烧怎么办')" style="cursor: pointer;">感冒发烧怎么办</li>
                <li onclick="setQuestion('高血压需要注意什么')" style="cursor: pointer;">高血压需要注意什么</li>
                <li onclick="setQuestion('布洛芬怎么吃')" style="cursor: pointer;">布洛芬怎么吃</li>
            </ul>
        </div>

        <div class="chat-area">
            <div class="chat-header">
                <h1>💬 医疗智能问答</h1>
                <p>基于知识库的专业医疗助手 | 24小时在线</p>
            </div>

            <div class="messages" id="messages">
                <div class="message assistant">
                    <div class="message-content">
                        您好！我是医疗智能助手，可以为您提供医疗知识查询服务。
                        请问有什么可以帮您？

⚠️ 温馨提示：本系统仅供参考，不能替代专业医疗诊断。
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <input type="text" id="question-input" placeholder="请输入您的医疗问题..." onkeypress="handleKeyPress(event)">
                    <button class="send-btn" id="send-btn" onclick="sendQuestion()">发送</button>
                </div>
                <div class="disclaimer">
                    ⚠️ 本信息仅供参考，不能替代专业医疗诊断
                </div>
            </div>
        </div>
    </div>

    <script>
        async function sendQuestion() {
            const input = document.getElementById('question-input');
            const sendBtn = document.getElementById('send-btn');
            const question = input.value.trim();

            if (!question) return;

            // 添加用户消息
            addMessage(question, 'user');
            input.value = '';

            // 禁用发送按钮
            sendBtn.disabled = true;

            // 显示打字指示器
            const typingId = showTypingIndicator();

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({question: question})
                });

                const data = await response.json();
                hideTypingIndicator(typingId);

                if (data.error) {
                    addMessage('抱歉: ' + data.error, 'assistant');
                } else {
                    addMessage(data.answer, 'assistant');
                }

            } catch (error) {
                hideTypingIndicator(typingId);
                addMessage('抱歉，服务暂时不可用，请稍后再试。', 'assistant');
            } finally {
                sendBtn.disabled = false;
            }
        }

        function addMessage(content, role) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            messageDiv.innerHTML = '<div class="message-content">' + content.replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g, '<br>') + '</div>';
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function showTypingIndicator() {
            const messagesDiv = document.getElementById('messages');
            const id = 'typing-' + Date.now();
            const typingDiv = document.createElement('div');
            typingDiv.id = id;
            typingDiv.className = 'message assistant';
            typingDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
            messagesDiv.appendChild(typingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return id;
        }

        function hideTypingIndicator(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }

        function reloadKB() {
            fetch('/api/reload', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    alert('知识库已重新加载！共' + data.doc_count + '个文档，' + data.chunk_count + '个片段');
                    document.getElementById('doc-count').textContent = data.doc_count;
                    document.getElementById('chunk-count').textContent = data.chunk_count;
                })
                .catch(error => {
                    alert('重新加载失败');
                });
        }

        function clearChat() {
            const messagesDiv = document.getElementById('messages');
            messagesDiv.innerHTML = '';
            addMessage('对话已清空，请问有什么可以帮您？', 'assistant');
        }

        function setQuestion(question) {
            document.getElementById('question-input').value = question;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendQuestion();
            }
        }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    """主页"""
    rag = get_rag()
    doc_count = len(set(doc['source'] for doc in rag.documents)) if rag.documents else 0
    chunk_count = len(rag.documents) if rag.documents else 0
    return render_template_string(HTML_TEMPLATE, doc_count=doc_count, chunk_count=chunk_count)


@app.route('/api/query', methods=['POST'])
@rate_limit(limit=20, per=60)
def api_query():
    """问答API"""
    start_time = time.time()

    try:
        data = request.json or {}
        question = data.get('question', '').strip()

        if not question:
            return jsonify({'error': '请提供问题'}), 400

        logger.info(f"API查询: {question[:100]}")

        rag = get_rag()
        answer = rag.query(question)

        duration = time.time() - start_time
        logger.info(f"查询完成，耗时: {duration:.2f}s")

        return jsonify({
            'answer': answer,
            'question': question,
            'duration': duration
        })

    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': '服务内部错误'}), 500


@app.route('/api/reload', methods=['POST'])
def api_reload():
    """重新加载知识库"""
    try:
        rag = get_rag()
        rag.load_knowledge_base()
        doc_count = len(set(doc['source'] for doc in rag.documents)) if rag.documents else 0
        chunk_count = len(rag.documents) if rag.documents else 0

        logger.info(f"知识库已重新加载: {doc_count}文档, {chunk_count}片段")

        return jsonify({
            'doc_count': doc_count,
            'chunk_count': chunk_count,
            'success': True
        })

    except Exception as e:
        logger.error(f"重新加载知识库失败: {e}")
        return jsonify({'error': '重新加载失败'}), 500


@app.errorhandler(404)
def not_found(e):
    """404处理"""
    return jsonify({'error': '页面不存在'}), 404


@app.errorhandler(500)
def server_error(e):
    """500处理"""
    logger.error(f"服务器错误: {e}")
    return jsonify({'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    print("[INFO] 启动Web应用...")
    print(f"[INFO] 访问地址: http://localhost:{config.FLASK_PORT}")
    print(f"[INFO] 调试模式: {'开启' if app.debug else '关闭'}")

    # 生产环境建议使用WSGI服务器，如gunicorn
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=False  # 生产环境关闭调试模式
    )
