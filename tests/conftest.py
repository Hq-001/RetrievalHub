"""pytest 全局夹具 - 确保 src 在 path 中，提供通用测试工具。"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 src 目录加入 Python path（确保 pip install -e 之前的开发模式可用）
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest

from retrievalhub.config import reset_settings


@pytest.fixture(autouse=True)
def reset_settings_fixture():
    """每个测试前后重置全局配置，避免测试间状态泄漏。"""
    reset_settings()
    yield
    reset_settings()
