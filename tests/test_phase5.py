"""Phase 5 测试 - REST API + MCP Server。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from retrievalhub.app import create_app
from retrievalhub.config import reset_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """创建测试客户端，使用临时数据目录。"""
    monkeypatch.setenv("LANCEDB_URI", str(tmp_path / "lancedb"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_settings()

    app = create_app()
    with TestClient(app) as c:
        yield c


# ---- 健康探针 ----


class TestHealth:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readyz(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200


# ---- 知识库管理 ----


class TestKnowledgeBases:
    def test_create_kb(self, client):
        resp = client.post(
            "/v1/knowledge-bases",
            data={"name": "Test KB", "description": "test"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["name"] == "Test KB"
        assert "collection" in data

    def test_list_kbs(self, client):
        client.post("/v1/knowledge-bases", data={"name": "KB1"})
        client.post("/v1/knowledge-bases", data={"name": "KB2"})
        resp = client.get("/v1/knowledge-bases")
        assert resp.status_code == 200
        kbs = resp.json()["knowledge_bases"]
        assert len(kbs) >= 2

    def test_delete_kb(self, client):
        resp = client.post("/v1/knowledge-bases", data={"name": "ToDelete"})
        kb_id = resp.json()["id"]
        resp = client.delete(f"/v1/knowledge-bases/{kb_id}")
        assert resp.status_code == 204

    def test_delete_nonexistent_kb(self, client):
        resp = client.delete("/v1/knowledge-bases/nonexistent")
        assert resp.status_code == 404


# ---- 文档上传与查询 ----


class TestDocumentUpload:
    def test_upload_md_returns_202(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "MD Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "processing"
        assert "doc_id" in data
        assert "status_url" in data

    def test_upload_json_returns_202(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "JSON Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.json", "rb") as f:
            resp = client.post(
                "/v1/documents",
                files={"file": ("sample.json", f, "application/json")},
                data={"kb_id": kb_id},
            )
        assert resp.status_code == 202

    def test_upload_jsonl_returns_202(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "JSONL Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.jsonl", "rb") as f:
            resp = client.post(
                "/v1/documents",
                files={"file": ("sample.jsonl", f, "application/json")},
                data={"kb_id": kb_id},
            )
        assert resp.status_code == 202

    def test_upload_unsupported_format(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "Format Test"}
        ).json()["id"]

        resp = client.post(
            "/v1/documents",
            files={"file": ("test.txt", b"plain text", "text/plain")},
            data={"kb_id": kb_id},
        )
        assert resp.status_code == 400

    def test_upload_to_nonexistent_kb(self, client):
        with open(FIXTURES / "sample.md", "rb") as f:
            resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": "nonexistent"},
            )
        assert resp.status_code == 404

    def test_get_document_status(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "Status Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            upload_resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )
        doc_id = upload_resp.json()["doc_id"]

        resp = client.get(f"/v1/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["doc_id"] == doc_id
        assert data["status"] in ("processing", "ready", "failed")

    def test_get_nonexistent_document(self, client):
        resp = client.get("/v1/documents/nonexistent")
        assert resp.status_code == 404

    def test_list_documents(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "List Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )

        resp = client.get(f"/v1/knowledge-bases/{kb_id}/documents")
        assert resp.status_code == 200
        docs = resp.json()["documents"]
        assert len(docs) >= 1

    def test_duplicate_upload_returns_409(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "Dup Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )

        with open(FIXTURES / "sample.md", "rb") as f:
            resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )
        assert resp.status_code == 409


# ---- 文档删除 ----


class TestDocumentDelete:
    def test_delete_document(self, client):
        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "Del Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            upload_resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )
        doc_id = upload_resp.json()["doc_id"]

        resp = client.delete(f"/v1/documents/{doc_id}")
        assert resp.status_code == 204

        # 确认已删除
        resp = client.get(f"/v1/documents/{doc_id}")
        assert resp.status_code == 404


# ---- 异步入库后状态轮询 ----


class TestAsyncIngestPolling:
    def test_upload_then_poll_until_ready(self, client):
        import time

        kb_id = client.post(
            "/v1/knowledge-bases", data={"name": "Poll Test"}
        ).json()["id"]

        with open(FIXTURES / "sample.md", "rb") as f:
            upload_resp = client.post(
                "/v1/documents",
                files={"file": ("sample.md", f, "text/markdown")},
                data={"kb_id": kb_id},
            )
        doc_id = upload_resp.json()["doc_id"]

        # 轮询状态
        max_wait = 10
        final_status = None
        for _ in range(max_wait):
            resp = client.get(f"/v1/documents/{doc_id}")
            final_status = resp.json()["status"]
            if final_status in ("ready", "failed"):
                break
            time.sleep(0.5)

        assert final_status == "ready"
        resp = client.get(f"/v1/documents/{doc_id}")
        assert resp.json()["chunk_count"] > 0


# ---- MCP 工具测试 ----


class TestMcpSearchTool:
    @pytest.fixture
    async def setup_with_data(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LANCEDB_URI", str(tmp_path / "lancedb"))
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "metadata.db"))
        reset_settings()

        app = create_app()
        with TestClient(app) as c:
            # 创建知识库
            kb_resp = c.post("/v1/knowledge-bases", data={"name": "MCP Test"})
            kb_id = kb_resp.json()["id"]

            # 上传文档
            with open(FIXTURES / "sample.md", "rb") as f:
                c.post(
                    "/v1/documents",
                    files={"file": ("sample.md", f, "text/markdown")},
                    data={"kb_id": kb_id},
                )

            # 等待入库完成
            import time

            for _ in range(10):
                docs = c.get(f"/v1/knowledge-bases/{kb_id}/documents").json()["documents"]
                if docs and docs[0]["status"] == "ready":
                    break
                time.sleep(0.5)

            yield c, kb_id

    def test_tool_name(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool
        assert mcp_tool.tool_name == "search_knowledge"

    def test_tool_description_contains_responsibility(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool
        desc = mcp_tool.tool_description
        assert "hallucination" in desc.lower() or "responsibility" in desc.lower()

    def test_tool_description_contains_timeout(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool
        desc = mcp_tool.tool_description
        assert "timeout" in desc.lower()
        assert "retry" in desc.lower()

    def test_tool_schema(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool
        schema = mcp_tool.get_tool_schema()
        assert schema["name"] == "search_knowledge"
        assert "query" in schema["inputSchema"]["properties"]
        assert "knowledge_base_id" in schema["inputSchema"]["properties"]
        assert "timeout_ms" in schema["inputSchema"]["properties"]

    async def test_execute_search(self, setup_with_data):
        client, kb_id = setup_with_data
        mcp_tool = client.app.state.mcp_tool

        result = await mcp_tool.execute({
            "query": "test",
            "knowledge_base_id": kb_id,
            "top_k": 5,
        })

        assert result["isError"] is False
        content = result["content"][0]
        data = json.loads(content["text"])
        assert "chunks" in data
        assert "total" in data

    async def test_execute_missing_params(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool

        result = await mcp_tool.execute({})
        assert result["isError"] is True

    async def test_execute_nonexistent_kb(self, setup_with_data):
        client, _ = setup_with_data
        mcp_tool = client.app.state.mcp_tool

        result = await mcp_tool.execute({
            "query": "test",
            "knowledge_base_id": "nonexistent",
        })
        assert result["isError"] is True

    async def test_execute_returns_source_metadata(self, setup_with_data):
        client, kb_id = setup_with_data
        mcp_tool = client.app.state.mcp_tool

        result = await mcp_tool.execute({
            "query": "test",
            "knowledge_base_id": kb_id,
            "top_k": 3,
        })

        data = json.loads(result["content"][0]["text"])
        if data["chunks"]:
            chunk = data["chunks"][0]
            assert "text" in chunk
            assert "score" in chunk
            assert "source" in chunk
            assert "doc_name" in chunk["source"]
            assert "content_type" in chunk["source"]
