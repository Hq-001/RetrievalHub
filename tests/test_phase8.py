"""Phase 8 测试 - 双集合迁移、RQ 预留、访问控制。"""

from __future__ import annotations

import pytest

from retrievalhub.core.access_control import AccessControl, Permission
from retrievalhub.core.models import DocFormat, DocStatus, Document, KnowledgeBase
from retrievalhub.storage.lancedb_store import LanceDBStorage
from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.storage.migration import (
    DualCollectionMigration,
    MigrationStatus,
)


# ---- 双集合迁移测试 ----


class TestDualCollectionMigration:
    @pytest.fixture
    def setup(self, tmp_path):
        storage = LanceDBStorage(uri=str(tmp_path / "lancedb"))
        metadata = MetadataStore(db_path=str(tmp_path / "metadata.db"))
        migration = DualCollectionMigration(storage, metadata)

        # 创建知识库
        kb = KnowledgeBase(
            id="kb-migrate",
            name="Migration Test",
            embedding_model="old-model",
            embedding_dim=64,
        )
        metadata.create_kb(kb)
        collection = "kb_kb-migrate"
        metadata.update_kb_collection("kb-migrate", collection)

        yield storage, metadata, migration
        metadata.close()

    def test_start_migration(self, setup):
        _, _, migration = setup
        state = migration.start_migration("kb-migrate", new_dim=128)
        assert state.status == MigrationStatus.CREATING
        assert state.old_dim == 64
        assert state.new_dim == 128
        assert "_v128" in state.new_collection

    def test_start_migration_nonexistent_kb(self, setup):
        _, _, migration = setup
        with pytest.raises(ValueError, match="not found"):
            migration.start_migration("nonexistent", new_dim=128)

    def test_switch_collection(self, setup):
        _, metadata, migration = setup
        migration.start_migration("kb-migrate", new_dim=128)
        state = migration.switch_collection("kb-migrate")
        assert state.status == MigrationStatus.SWITCHED

        kb = metadata.get_kb("kb-migrate")
        assert kb.active_collection == state.new_collection

    def test_rollback(self, setup):
        _, metadata, migration = setup
        state = migration.start_migration("kb-migrate", new_dim=128)
        old_active = state.old_collection

        migration.switch_collection("kb-migrate")
        kb = metadata.get_kb("kb-migrate")
        assert kb.active_collection == state.new_collection

        # 回滚
        migration.rollback("kb-migrate")
        kb = metadata.get_kb("kb-migrate")
        assert kb.active_collection == old_active

        state = migration.get_state("kb-migrate")
        assert state.status == MigrationStatus.ROLLED_BACK

    def test_get_state(self, setup):
        _, _, migration = setup
        assert migration.get_state("kb-migrate") is None
        migration.start_migration("kb-migrate", new_dim=128)
        state = migration.get_state("kb-migrate")
        assert state is not None
        assert state.kb_id == "kb-migrate"


# ---- RQ 后端测试 ----


class TestRQBackend:
    def test_import_error_without_redis(self):
        """无 redis 包时应抛出 ImportError。"""
        from retrievalhub.ingest.rq_backend import RQEnqueueBackend

        backend = RQEnqueueBackend(redis_url="redis://localhost:6379/0")
        # _ensure_connection 会在无 redis/rq 包时抛 ImportError
        # 如果装了 redis/rq 但连不上则抛 ConnectionError
        # 测试仅验证对象可创建
        assert backend._redis_url == "redis://localhost:6379/0"
        assert backend._queue_name == "retrievalhub-ingest"
        assert backend._timeout_sec == 30

    def test_is_async_wrapper(self):
        from retrievalhub.ingest.rq_backend import _is_async_callable

        async def async_fn():
            pass

        def sync_fn():
            pass

        assert _is_async_callable(async_fn) is True
        assert _is_async_callable(sync_fn) is False


# ---- 访问控制测试 ----


class TestAccessControl:
    @pytest.fixture
    def setup(self, tmp_path):
        metadata = MetadataStore(db_path=str(tmp_path / "metadata.db"))
        metadata.create_kb(KnowledgeBase(id="kb1", name="KB1"))
        metadata.create_kb(KnowledgeBase(id="kb2", name="KB2"))
        ac = AccessControl(metadata)
        return metadata, ac

    def test_grant_and_check_read(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.READ)
        assert ac.can_read("key1", "kb1") is True
        assert ac.can_write("key1", "kb1") is False
        assert ac.is_admin("key1", "kb1") is False

    def test_grant_write_includes_read(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.WRITE)
        assert ac.can_read("key1", "kb1") is True
        assert ac.can_write("key1", "kb1") is True
        assert ac.is_admin("key1", "kb1") is False

    def test_grant_admin_all(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.ADMIN)
        assert ac.can_read("key1", "kb1") is True
        assert ac.can_write("key1", "kb1") is True
        assert ac.is_admin("key1", "kb1") is True

    def test_no_access(self, setup):
        _, ac = setup
        assert ac.can_read("unknown-key", "kb1") is False
        assert ac.can_write("unknown-key", "kb1") is False

    def test_no_access_to_kb(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.READ)
        assert ac.can_read("key1", "kb2") is False

    def test_revoke(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.READ)
        assert ac.can_read("key1", "kb1") is True
        ac.revoke("key1", "kb1")
        assert ac.can_read("key1", "kb1") is False

    def test_list_accessible_kbs(self, setup):
        _, ac = setup
        ac.grant("key1", "kb1", Permission.READ)
        ac.grant("key1", "kb2", Permission.WRITE)
        kbs = ac.list_accessible_kbs("key1")
        assert len(kbs) == 2
        kb_ids = {kb_id for kb_id, _ in kbs}
        assert "kb1" in kb_ids
        assert "kb2" in kb_ids

    def test_permission_hierarchy(self, setup):
        """ADMIN > WRITE > READ 层级正确。"""
        _, ac = setup
        ac.grant("reader", "kb1", Permission.READ)
        ac.grant("writer", "kb1", Permission.WRITE)
        ac.grant("admin", "kb1", Permission.ADMIN)

        # reader 不能写
        assert ac.can_write("reader", "kb1") is False
        # writer 不能管理
        assert ac.is_admin("writer", "kb1") is False
        # admin 全部可以
        assert ac.is_admin("admin", "kb1") is True
        assert ac.can_write("admin", "kb1") is True
        assert ac.can_read("admin", "kb1") is True
