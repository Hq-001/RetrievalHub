"""多知识库权限隔离 - 预留位。

数据模型已保留知识库 ID 维度，后续可扩展用户体系与知识库级权限。
当前实现基于 API Key 的全局访问，预留 RBAC 扩展接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from retrievalhub.storage.metadata_store import MetadataStore
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class Permission(str, Enum):
    """权限级别。"""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class KbAccess:
    """知识库访问控制记录。"""

    kb_id: str
    api_key: str
    permission: Permission = Permission.READ
    created_at: str = ""


class AccessControl:
    """知识库级访问控制。

    当前实现：简单内存映射（api_key -> [kb_id, permission]）。
    后续可扩展为数据库持久化的 RBAC/ABAC 模型。
    """

    def __init__(self, metadata: MetadataStore) -> None:
        self._metadata = metadata
        # api_key -> {kb_id -> Permission}
        self._access_map: dict[str, dict[str, Permission]] = {}

    def grant(self, api_key: str, kb_id: str, permission: Permission) -> None:
        """授予 API Key 对知识库的访问权限。"""
        if api_key not in self._access_map:
            self._access_map[api_key] = {}
        self._access_map[api_key][kb_id] = permission
        logger.info(
            "access_granted",
            api_key=api_key[:8] + "...",
            kb_id=kb_id,
            permission=permission.value,
        )

    def revoke(self, api_key: str, kb_id: str) -> None:
        """撤销访问权限。"""
        if api_key in self._access_map:
            self._access_map[api_key].pop(kb_id, None)

    def check(self, api_key: str, kb_id: str, required: Permission) -> bool:
        """检查权限。

        Args:
            api_key: 调用方 API Key
            kb_id: 目标知识库 ID
            required: 所需权限级别

        Returns:
            是否有权限
        """
        if api_key not in self._access_map:
            return False
        kb_perms = self._access_map[api_key]
        if kb_id not in kb_perms:
            return False

        granted = kb_perms[kb_id]
        # 权限层级：ADMIN > WRITE > READ
        levels = {Permission.READ: 1, Permission.WRITE: 2, Permission.ADMIN: 3}
        return levels[granted] >= levels[required]

    def list_accessible_kbs(self, api_key: str) -> list[tuple[str, Permission]]:
        """列出 API Key 可访问的所有知识库。"""
        if api_key not in self._access_map:
            return []
        return list(self._access_map[api_key].items())

    def is_admin(self, api_key: str, kb_id: str) -> bool:
        """是否为管理员。"""
        return self.check(api_key, kb_id, Permission.ADMIN)

    def can_write(self, api_key: str, kb_id: str) -> bool:
        """是否可写。"""
        return self.check(api_key, kb_id, Permission.WRITE)

    def can_read(self, api_key: str, kb_id: str) -> bool:
        """是否可读。"""
        return self.check(api_key, kb_id, Permission.READ)
