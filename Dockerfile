# RetrievalHub Dockerfile - 多阶段构建
FROM python:3.12-slim AS builder

WORKDIR /build

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY pyproject.toml README.md ./
COPY src/ ./src/

# 安装依赖到构建阶段（非 editable 安装，包会被复制到 site-packages）
RUN pip install --no-cache-dir ".[dev]"

# ---- 运行阶段 ----
FROM python:3.12-slim

WORKDIR /app

# 从构建阶段复制已安装的包（含 retrievalhub 包本身，已在 site-packages 中）
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 创建数据目录
RUN mkdir -p /app/data/originals

# 环境变量默认值
ENV REST_HOST=0.0.0.0 \
    REST_PORT=8000 \
    MCP_TRANSPORT=stdio \
    LOG_LEVEL=INFO \
    LANCEDB_URI=/app/data/lancedb \
    SQLITE_PATH=/app/data/metadata.db

EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "retrievalhub.app:app", "--host", "0.0.0.0", "--port", "8000"]
