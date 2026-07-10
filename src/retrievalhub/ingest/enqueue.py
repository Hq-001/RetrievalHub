"""异步入库后端 - 进程内实现（asyncio.Task）。

首版默认进程内后台任务，零外部依赖。
后续可平滑切换为 RQ（Redis Queue）。
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Awaitable

from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class InProcessEnqueueBackend:
    """进程内异步入库后端。

    使用 asyncio.Task 管理后台任务，零外部依赖。
    任务状态存储在内存中（进程重启后丢失，由崩溃恢复机制兜底）。
    """

    def __init__(self, timeout_sec: int = 30) -> None:
        self._timeout_sec = timeout_sec
        self._tasks: dict[str, asyncio.Task] = {}
        self._statuses: dict[str, str] = {}  # task_id -> running|completed|failed

    async def submit(
        self,
        task: Callable[..., Awaitable[Any]],
        *args: Any,
        task_id: str = "",
        **kwargs: Any,
    ) -> str:
        """提交异步任务。

        Args:
            task: async 可调用对象
            task_id: 任务 ID（如文档 ID），为空则自动生成

        Returns:
            任务 ID
        """
        import uuid

        if not task_id:
            task_id = str(uuid.uuid4())

        self._statuses[task_id] = "running"

        async def _wrapped():
            try:
                result = await asyncio.wait_for(
                    task(*args, **kwargs),
                    timeout=self._timeout_sec,
                )
                self._statuses[task_id] = "completed"
                return result
            except asyncio.TimeoutError:
                self._statuses[task_id] = "failed"
                logger.error("task_timeout", task_id=task_id, timeout=self._timeout_sec)
                raise
            except Exception as e:
                self._statuses[task_id] = "failed"
                logger.error("task_failed", task_id=task_id, error=str(e))
                raise

        self._tasks[task_id] = asyncio.create_task(_wrapped())
        logger.info("task_submitted", task_id=task_id)
        return task_id

    def get_status(self, task_id: str) -> str:
        """查询任务状态。"""
        return self._statuses.get(task_id, "unknown")

    def is_running(self, task_id: str) -> bool:
        return self.get_status(task_id) == "running"

    async def wait_for_completion(self, task_id: str, timeout: float = 60) -> Any:
        """等待任务完成（测试用）。"""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        return await asyncio.wait_for(task, timeout=timeout)

    def pending_tasks(self) -> list[str]:
        """获取所有运行中的任务 ID。"""
        return [tid for tid, status in self._statuses.items() if status == "running"]
