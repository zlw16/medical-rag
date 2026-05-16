FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# Hugging Face Spaces 使用端口 7860
ENV SERVER_PORT=7860
ENV SERVER_HOST=0.0.0.0

# 首次启动时模型会自动下载并缓存
CMD ["python", "start_app.py"]
