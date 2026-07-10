# RetrievalHub

> 个人智能知识库检索中间件 - RAG-Ready 检索基础设施

**只负责检索（R），不负责生成（G）**。以 MCP 协议对接任意 AI Agent，成为 Agent 的"知识外脑"。

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖（含开发依赖）
pip install -e ".[dev]"

# 3. 复制配置模板
cp .env.example .env

# 4. 运行测试
pytest

# 5. 启动服务
uvicorn retrievalhub.app:app --reload
```

## 项目结构

```
src/retrievalhub/
├── api/          # REST 路由
├── core/         # 领域模型与异常
├── protocols/    # 接口抽象层
├── parsers/      # MD/JSON 解析分块
├── embedders/    # 嵌入模型
├── storage/      # LanceDB 存储
├── retrieval/    # 检索管道
├── ingest/       # 入库流水线
├── mcp_server/   # MCP Server
├── utils/        # 工具（日志等）
├── config.py     # 配置加载
└── app.py        # FastAPI 入口
```

## 技术栈

| 层次 | 选型 |
|------|------|
| Web 框架 | FastAPI |
| MCP 框架 | mcp (官方 SDK) |
| Markdown 解析 | markdown-it-py |
| JSON 解析 | orjson |
| 向量+全文库 | LanceDB |
| 配置管理 | pydantic-settings |
| 测试 | pytest |
