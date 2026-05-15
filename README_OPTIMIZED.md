# 医疗RAG系统 - 优化版

## 🎉 优化内容

### 1. 🔒 安全配置
- 使用 `.env` 文件管理敏感配置（API Key、密钥等）
- 添加 `.gitignore` 保护敏感信息
- 配置文件通过环境变量加载

### 2. ⚡ 性能优化
- 索引持久化缓存（使用 joblib）
- 第二次启动直接从缓存加载，大幅提升速度
- 检测文档变化自动重建索引

### 3. 🤖 LLM答案生成
- 集成 DeepSeek API
- 智能生成自然语言回答
- 无 API Key 时自动降级到规则生成

### 4. 📝 日志记录
- 结构化日志（控制台 + 文件）
- 按日期自动归档日志
- 记录查询耗时和错误信息

### 5. 🛡️ 健壮性优化
- 完善的异常处理
- API 请求限流
- XSS 防护（HTML 转义）

### 6. 📦 项目结构
```
medical_rag/
├── src/                     # 源代码
│   ├── loaders/            # 文档加载
│   ├── retrievers/         # 检索器
│   ├── classifiers/        # 意图分类
│   ├── rag_engine.py       # 主引擎
│   ├── llm_generator.py    # LLM生成
│   └── logger.py           # 日志模块
├── tests/                  # 测试
├── cache/                  # 索引缓存
├── logs/                   # 日志文件
├── medical_knowledge/      # 知识库
├── config.py               # 配置
├── .env                    # 环境变量
└── requirements.txt        # 依赖
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量
复制 `.env.example` 为 `.env` 并填入配置：
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行交互模式
```bash
python main.py
```

### 4. 运行Web服务
```bash
python flask_app.py
```
访问 http://localhost:5000

## 📝 使用说明

### 知识库
在 `medical_knowledge` 文件夹中放入 `.txt` 格式的医疗文档。

### 配置
在 `.env` 文件中可以调整：
- `TOP_K`：检索返回文档数
- `TFIDF_WEIGHT` / `BM25_WEIGHT`：检索权重
- `FLASK_PORT`：Web服务端口

### 缓存
索引会自动缓存到 `cache` 目录，知识库变更时会自动重建。

## 🔒 安全注意

- 不要将 `.env` 文件提交到版本控制
- 生产环境务必修改 `FLASK_SECRET_KEY`
- 建议使用 HTTPS 和身份验证

## 📊 日志
日志文件按日期保存在 `logs/` 目录，包含：
- 系统启动信息
- 查询记录和耗时
- 错误和警告
