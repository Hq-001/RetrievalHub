"""异步入库后端协议 - EnqueueBackend。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EnqueueBackend(Protocol):
    """异步入库后端接口。

    首版默认进程内后台任务（asyncio.Task / 线程池），零外部依赖。
    后续可平滑切换为 RQ（Redis Queue）支撑大规模批量场景。
    """

    async def submit(self, task: object, *args, **kwargs) -> str:
        """提交异步任务。

        Args:
            task: 可调用对象（async 函数或普通函数）

        Returns:
            任务 ID（用于状态查询）
        """
        ...

    def get_status(self, task_id: str) -> str:
        """查询任务状态。

        Returns:
            running | completed | failed
        """
        ...
