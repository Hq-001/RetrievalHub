"""MCP Server - 暴露 search_knowledge 工具供 AI Agent 调用。

支持 stdio / SSE 两种传输。
工具描述含超时/重试约定与幻觉责任转移声明。
"""

from __future__ import annotations

import json
from typing import Any

from retrievalhub.core.models import SearchRequest
from retrievalhub.retrieval.service import SearchService
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)

# 工具描述 - 含幻觉责任转移声明与超时/重试约定
TOOL_DESCRIPTION = """Search the knowledge base for relevant text chunks.

This tool returns ONLY raw text chunks retrieved from the knowledge base,
along with their source metadata. It does NOT generate any summary, answer,
or conclusion.

RESPONSIBILITY: The accuracy and hallucination risk of any content generated
based on these chunks is solely the responsibility of the calling AI Agent.

TIMEOUT & RETRY:
- Recommended timeout: 5s (timeout_ms defaults to 5000)
- On timeout or temporary error (5xx/network), retry with exponential backoff:
  max 3 retries, initial interval 200ms.
- This tool is read-only; retries are safe and idempotent.

Args:
    query: Search query text
    knowledge_base_id: Target knowledge base ID
    top_k: Max results to return (default: 10)
    rerank: Whether to apply cross-encoder reranking (default: true)
    filters: Optional metadata filters (e.g. {"content_type": "md"})
    timeout_ms: Per-call timeout in milliseconds (default: 5000)

Returns:
    List of chunks, each containing:
    - text: Original chunk text (no summarization)
    - score: Relevance score
    - source: Metadata (doc_name, content_type, section_path, etc.)
"""


class McpSearchTool:
    """MCP search_knowledge 工具封装。

    封装检索服务，提供 MCP 协议兼容的调用接口。
    实际传输层（stdio/SSE）由 mcp SDK 处理，此处提供核心逻辑。
    """

    def __init__(self, search_service: SearchService) -> None:
        self._service = search_service

    @property
    def tool_name(self) -> str:
        return "search_knowledge"

    @property
    def tool_description(self) -> str:
        return TOOL_DESCRIPTION

    def get_tool_schema(self) -> dict:
        """返回 MCP 工具的 JSON Schema。"""
        return {
            "name": self.tool_name,
            "description": self.tool_description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                    "knowledge_base_id": {
                        "type": "string",
                        "description": "Target knowledge base ID",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 10,
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Apply cross-encoder reranking",
                        "default": True,
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Per-call timeout in ms",
                        "default": 5000,
                    },
                },
                "required": ["query", "knowledge_base_id"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict:
        """执行检索工具。

        Args:
            arguments: 工具入参（query, knowledge_base_id, top_k 等）

        Returns:
            MCP 格式的工具响应
        """
        query = arguments.get("query", "")
        kb_id = arguments.get("knowledge_base_id", "")
        top_k = arguments.get("top_k", 10)
        rerank = arguments.get("rerank", True)
        filters = arguments.get("filters")
        timeout_ms = arguments.get("timeout_ms", 5000)

        if not query or not kb_id:
            return {
                "isError": True,
                "content": [{"type": "text", "text": "query and knowledge_base_id are required"}],
            }

        try:
            request = SearchRequest(
                query=query,
                kb_id=kb_id,
                top_k=top_k,
                rerank=rerank,
                filters=filters,
                timeout_ms=timeout_ms,
            )

            response = await self._service.search(request)

            # 组装 MCP 响应格式
            chunks = []
            for hit in response.hits:
                chunks.append({
                    "text": hit.text,
                    "score": hit.score,
                    "source": {
                        "doc_name": hit.doc_name,
                        "doc_id": hit.doc_id,
                        "content_type": hit.content_type.value if hasattr(hit.content_type, 'value') else hit.content_type,
                        "section_path": hit.section_path,
                        "json_parent_id": hit.json_parent_id,
                        "code_language": hit.code_language,
                        "char_offset": hit.char_offset,
                    },
                })

            logger.info(
                "mcp_search_executed",
                query=query,
                kb_id=kb_id,
                result_count=len(chunks),
            )

            return {
                "isError": False,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "query": query,
                                "total": len(chunks),
                                "chunks": chunks,
                                "cached": response.cached,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
            }

        except Exception as e:
            logger.error("mcp_search_error", query=query, error=str(e))
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Search failed: {e}"}],
            }
