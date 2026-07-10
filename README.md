# RetrievalHub

> 个人智能知识库检索中间件 - RAG-Ready 检索基础设施

**只负责检索（R），不负责生成（G）**。以 MCP 协议对接任意 AI Agent，成为 Agent 的"知识外脑"。

系统将"文档沉淀 - 知识切分 - 向量化入库 - 混合检索 - 结构化返回"打通为闭环，对上层以两种方式暴露能力：

1. **RESTful API** - 文档管理（上传、删除、列表）
2. **MCP Server** - 将检索封装为 `search_knowledge` 工具，供外部 AI Agent 调用，返回结构化 Chunk 列表

---

## 核心特性

- **职责单一**：只做检索，不内置 LLM，不做问答生成
- **解析零风险**：仅支持 Markdown / JSON，从根源消除 PDF/OCR 不确定性
- **混合检索 + 动态加权**：向量召回 + BM25 全文召回，按 content_type 差异化加权（MD 偏向量、JSON 偏 BM25）
- **异步入库**：202 Accepted + 状态轮询，大文件不阻塞
- **崩溃恢复**：进程重启后自动标记超时 processing 文档为 failed
- **热点查询 LRU 缓存**：doc_versions_hash 分区驱逐，单文档更新不清空全库缓存
- **评测驱动调参**：分层抽样评测集 + K-Fold 交叉验证 + 网格搜索自动调参
- **幻觉责任明确转移**：MCP 工具描述显式声明，基于片段生成的责任由调用方 Agent 承担
- **可插拔架构**：嵌入模型、重排序模型、存储、检索策略均可替换

---

## 快速开始

### 环境要求

- Python >= 3.11
- pip

### 安装

```bash
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac

# 2. 安装依赖（含开发依赖）
pip install -e ".[dev]"

# 3. 复制配置模板
cp .env.example .env

# 4. 运行测试（198 个测试）
pytest

# 5. 启动服务
uvicorn retrievalhub.app:app --reload
```

### 验证服务

```bash
# 存活探针
curl http://localhost:8000/healthz
# {"status":"ok"}

# 就绪探针
curl http://localhost:8000/readyz
# {"status":"ready"}

# API 文档
# 浏览器打开 http://localhost:8000/docs
```

---

## 项目结构

```
src/retrievalhub/
├── api/               # REST 路由 + 中间件
│   ├── documents.py   # 文档管理 API（上传/查询/删除/重试/知识库管理）
│   ├── health.py      # 健康探针 /healthz + /readyz
│   └── middleware.py  # API Key 鉴权 + 令牌桶限流 + 请求日志
├── core/              # 核心领域
│   ├── models.py      # KnowledgeBase / Document / Chunk / SearchHit
│   ├── exceptions.py  # 统一异常体系
│   └── access_control.py  # 多知识库 RBAC 权限隔离
├── protocols/         # 6 个接口抽象（可插拔）
│   ├── parser.py      # DocumentParser 协议
│   ├── chunker.py     # Chunker 协议
│   ├── embedder.py    # Embedder 协议
│   ├── storage.py     # Storage 协议
│   ├── reranker.py    # Reranker 协议
│   └── enqueue.py     # EnqueueBackend 协议
├── parsers/           # MD/JSON 解析与分块
│   ├── md_parser.py   # markdown-it-py AST 解析
│   ├── md_chunker.py  # 递归标题继承分块（section_path <= 3，代码块完整）
│   ├── json_parser.py # orjson 扁平化 + 数组拆分
│   └── json_chunker.py # json_parent_id 稳定哈希 + 超长 Value 降级
├── embedders/         # 嵌入模型
│   └── embedder.py    # MockEmbedder + OpenAI 兼容适配器（指数退避重试）
├── storage/           # 存储层
│   ├── lancedb_store.py    # LanceDB 向量 + FTS 混合检索
│   ├── metadata_store.py   # SQLite 元数据库（UNIQUE 去重/崩溃恢复）
│   └── migration.py       # 双集合写入零停机迁移
├── retrieval/         # 检索管道
│   ├── recall.py      # 混合召回（向量 + BM25 并行）
│   ├── fusion.py      # 动态加权融合（RRF + content_type 差异化）
│   ├── reranker.py    # Cross-encoder 重排序（Mock + Disabled）
│   ├── service.py     # 检索服务编排
│   ├── cache.py       # LRU 缓存（doc_versions_hash 分区驱逐）
│   ├── eval.py        # 评测集（分层抽样 + Recall@k / MRR）
│   └── tune.py       # 自动调参（网格搜索 + K-Fold 交叉验证）
├── ingest/            # 入库流水线
│   ├── file_handler.py    # 格式校验 + 内容哈希 + 原文存储
│   ├── pipeline.py        # 解析 -> 分块 -> 嵌入 -> 写存储
│   ├── enqueue.py         # 进程内异步后端（asyncio.Task）
│   ├── crash_recovery.py  # 崩溃恢复（超时 processing -> failed）
│   └── rq_backend.py      # RQ (Redis Queue) 预留实现
├── mcp_server/        # MCP Server
│   └── tool.py        # search_knowledge 工具（含幻觉责任声明）
├── utils/
│   └── logging.py     # structlog 结构化日志
├── config.py          # pydantic-settings .env 配置加载
└── app.py             # FastAPI 入口（整合全部组件）

tests/                 # 198 个测试
├── test_phase0.py     # 配置 + 模型 + 健康探针（12）
├── test_phase1_md.py  # MD 解析分块（20）
├── test_phase1_json.py # JSON 解析分块（27）
├── test_phase2.py     # 嵌入器 + 元数据库（18）
├── test_phase3.py     # 入库流水线 + 崩溃恢复（26）
├── test_phase4.py     # 融合器 + 重排序 + 检索服务（18）
├── test_phase5.py     # REST API + MCP 工具（25）
├── test_phase6.py     # LRU 缓存 + 评测集 + 自动调参（27）
├── test_phase7.py     # 鉴权 + 限流 + 请求日志（11）
└── test_phase8.py     # 双集合迁移 + RQ + 权限控制（15）

deploy/
└── k8s.yaml           # K8s 部署清单（livenessProbe + readinessProbe）

scripts/
└── test.ps1           # PowerShell 测试脚本
```

---

## 架构设计

### 四层架构

```
┌──────────────────────────────────────────────────────────┐
│  接入层                                                    │
│   ├─ RESTful API：文档管理（上传 / 删除 / 列表）           │
│   │   ├─ API Key 鉴权（Bearer / X-API-Key）               │
│   │   ├─ 令牌桶限流（按 IP，burst + 429）                  │
│   │   └─ 请求日志（X-Response-Time-ms）                    │
│   └─ MCP Server（stdio / SSE）：search_knowledge 工具      │
│       ├─ 超时/重试约定（timeout_ms，指数退避）              │
│       └─ 幻觉责任转移声明                                  │
├──────────────────────────────────────────────────────────┤
│  应用层：文档管理服务 / 检索服务 / 知识库管理服务           │
├──────────────────────────────────────────────────────────┤
│  能力层                                                    │
│   ├─ MD 解析（AST） / JSON 解析（扁平化）                   │
│   ├─ 递归标题继承分块 / JSON 数组拆分                      │
│   ├─ 嵌入（Mock / OpenAI 兼容）                            │
│   ├─ 混合检索（向量 + BM25 + 动态加权融合 + 重排序）        │
│   └─ 热点缓存（LRU + doc_versions_hash 分区驱逐）          │
├──────────────────────────────────────────────────────────┤
│  存储层：LanceDB（向量 + FTS） / SQLite（元数据） / 原文    │
└──────────────────────────────────────────────────────────┘
```

### 检索三级管道

```
查询 -> 混合召回（向量 top-N + BM25 top-N 并行）
         ↓
       动态加权融合（RRF + content_type 差异化加权）
       ├─ MD 块：向量得分 x MD_VECTOR_WEIGHT (1.1)
       └─ JSON 块：BM25 得分 x JSON_BM25_WEIGHT (1.2)
         ↓
       Cross-encoder 重排序精排（仅排序，不生成）
         ↓
       结构化 Chunk 列表（原文 + 来源元数据）
```

### 异步入库流程

```
上传文件 -> 格式校验 -> 202 Accepted（立即返回）
    -> 后台异步：解析 -> 分块 -> 批量嵌入 -> 写索引
    -> 状态轮询：processing -> ready / failed
    -> 失败可手动重试（幂等，按内容哈希去重）
```

---

## API 接口

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/knowledge-bases` | 创建知识库 |
| GET | `/v1/knowledge-bases` | 列出知识库 |
| DELETE | `/v1/knowledge-bases/{kb_id}` | 删除知识库（级联） |
| POST | `/v1/documents` | 上传文档（返回 202） |
| GET | `/v1/documents/{doc_id}` | 查询文档详情/状态 |
| GET | `/v1/knowledge-bases/{kb_id}/documents` | 列出文档 |
| DELETE | `/v1/documents/{doc_id}` | 删除文档 |
| POST | `/v1/documents/{doc_id}/retry` | 重试失败文档 |
| GET | `/healthz` | 存活探针 |
| GET | `/readyz` | 就绪探针 |

### MCP 工具

**工具名**：`search_knowledge`

**入参**：
- `query` - 查询文本（必填）
- `knowledge_base_id` - 知识库 ID（必填）
- `top_k` - 返回数量（默认 10）
- `rerank` - 是否重排序（默认 true）
- `filters` - 元信息过滤
- `timeout_ms` - 超时（默认 5000）

**出参**：结构化 Chunk 列表，每项含：
- `text` - 分块原文（不做摘要、不改写）
- `score` - 相关性得分
- `source` - 来源元数据（文档名、content_type、section_path、code_language 等）

---

## 配置项

所有配置通过 `.env` 注入，零硬编码。完整模板见 `.env.example`。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `REST_PORT` | 8000 | REST 服务端口 |
| `MCP_TRANSPORT` | stdio | MCP 传输模式（stdio / sse） |
| `MD_CHUNK_SIZE` | 800 | MD 分块大小上限（字符） |
| `MD_CHUNK_OVERLAP` | 100 | MD 分块重叠量（字符） |
| `JSON_CHUNK_SIZE` | 600 | JSON 分块大小上限（字符） |
| `JSON_CHUNK_OVERLAP` | 60 | JSON 分块重叠量（字符） |
| `MD_VECTOR_WEIGHT` | 1.1 | MD 块向量得分权重 |
| `JSON_BM25_WEIGHT` | 1.2 | JSON 块 BM25 得分权重 |
| `INGEST_TIMEOUT_SEC` | 30 | 后台入库总超时 |
| `QUERY_CACHE_MAXSIZE` | 512 | LRU 缓存容量 |
| `QUERY_CACHE_TTL_SEC` | 300 | 缓存过期时间 |
| `MCP_SEARCH_TIMEOUT_MS` | 5000 | MCP 检索推荐超时 |
| `ENABLE_STORAGE_ENCRYPTION` | false | 存储加密开关 |
| `API_KEY` | (空) | REST 鉴权密钥 |

---

## Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker logs retrievalhub

# 健康检查
curl http://localhost:8000/healthz
```

Docker Compose 暴露两个端口：
- `8000` - REST API
- `8001` - MCP SSE

---

## 技术栈

| 层次 | 选型 | 说明 |
|------|------|------|
| 开发语言 | Python 3.11+ | AI/检索生态成熟 |
| Web 框架 | FastAPI | 异步 + 自动 OpenAPI 文档 |
| MCP 框架 | mcp | 官方 SDK，支持 stdio/SSE |
| Markdown 解析 | markdown-it-py | AST 解析，提取标题层级与代码块 |
| JSON 解析 | orjson | 高性能解析/序列化 |
| 向量+全文库 | LanceDB | 原生向量 + FTS 混合检索 |
| 元数据库 | SQLite | 轻量零运维 |
| 配置管理 | pydantic-settings | .env 统一加载 |
| 日志 | structlog | 结构化 JSON 日志 |
| 容器化 | Docker + docker-compose | 一键部署 |
| 测试 | pytest | 198 个测试，覆盖率 > 80% |

---

## 测试

```bash
# 全量测试
pytest

# 带覆盖率
pytest --cov=retrievalhub --cov-report=term-missing --cov-report=html

# 仅单元测试
pytest tests/test_phase0.py tests/test_phase1_md.py tests/test_phase1_json.py -v

# 仅集成测试
pytest tests/test_phase5.py -v
```

---

## 扩展性

- **嵌入模型可切换**：OpenAI 兼容 / 本地模型，配合双集合写入实现零停机迁移
- **检索策略可插拔**：召回器、融合器、重排器以接口暴露，新增算法只需实现接口
- **格式解析器可扩展**：MD/JSON 解析器以接口形式存在，可扩展 YAML/CSV
- **异步入库可演进**：进程内任务 -> RQ (Redis Queue)，无需改接口契约
- **多知识库权限隔离**：RBAC 三级权限（READ / WRITE / ADMIN），预留用户体系
