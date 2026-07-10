"""RQ (Redis Queue) 异步入库后端 - 预留实现。

首版走进程内后台任务（InProcessEnqueueBackend），零外部依赖。
大规模批量场景可平滑切换为 RQ，无需改动上传/轮询接口契约。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from retrievalhub.protocols.enqueue import EnqueueBackend
from retrievalhub.utils.logging import get_logger

logger = get_logger(__name__)


class RQEnqueueBackend:
    """RQ (Redis Queue) 异步入库后端。

    预留实现：接口已就绪，需配置 Redis 连接后启用。
    与 InProcessEnqueueBackend 实现相同接口契约，可平滑替换。

    启用步骤：
    1. pip install rq redis
    2. .env 配置 REDIS_URL=redis://localhost:6379/0
    3. 启动 worker: rq worker retrievalhub-ingest
    4. 将 app.py 中 InProcessEnqueueBackend 替换为 RQEnqueueBackend
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_name: str = "retrievalhub-ingest",
        timeout_sec: int = 30,
    ) -> None:
        self._redis_url = redis_url
        self._queue_name = queue_name
        self._timeout_sec = timeout_sec
        self._queue = None
        self._redis = None

    def _ensure_connection(self):
        """建立 Redis 连接（惰性初始化）。"""
        if self._redis is None:
            try:
                import redis
                from rq import Queue

                self._redis = redis.from_url(self._redis_url)
                self._queue = Queue(self._queue_name, connection=self._redis)
                logger.info("rq_connected", url=self._redis_url, queue=self._queue_name)
            except ImportError:
                raise ImportError(
                    "RQ backend requires: pip install rq redis"
                )

    async def submit(
        self,
        task: Callable[..., Any],
        *args: Any,
        task_id: str = "",
        **kwargs: Any,
    ) -> str:
        """提交任务到 RQ 队列。

        Args:
            task: 可调用对象（async 函数需包装为 sync）
            task_id: 任务 ID

        Returns:
            任务 ID
        """
        import uuid

        if not task_id:
            task_id = str(uuid.uuid4())

        self._ensure_connection()

        # RQ 处理 sync 函数，async 需包装
        if _is_async_callable(task):
            sync_wrapper = _make_sync_wrapper(task)
        else:
            sync_wrapper = task

        job = self._queue.enqueue(
            sync_wrapper,
            *args,
            job_id=task_id,
            job_timeout=self._timeout_sec,
            **kwargs,
        )

        logger.info("rq_task_submitted", task_id=task_id, job_id=job.id)
        return task_id

    def get_status(self, task_id: str) -> str:
        """查询任务状态。"""
        self._ensure_connection()
        from rq.job import Job

        job = Job.fetch(task_id, connection=self._redis)
        if job is None:
            return "unknown"

        status_map = {
            "queued": "running",
            "started": "running",
            "finished": "completed",
            "failed": "failed",
        }
        return status_map.get(job.get_status(), "unknown")

    def is_running(self, task_id: str) -> bool:
        return self.get_status(task_id) == "running"

    def pending_tasks(self) -> list[str]:
        """获取队列中等待的任务。"""
        self._ensure_connection()
        return [job.id for job in self._queue.get_jobs()]


def _is_async_callable(fn) -> bool:
    """判断是否为 async 可调用对象。"""
    import asyncio

    return asyncio.iscoroutinefunction(fn)


def _make_sync_wrapper(async_fn):
    """将 async 函数包装为 sync 函数（供 RQ worker 调用）。"""
    import asyncio

    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(async_fn(*args, **kwargs))
        finally:
            loop.close()

    return wrapper
