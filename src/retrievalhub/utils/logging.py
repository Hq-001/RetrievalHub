"""结构化日志 - JSON 格式，含 request_id 追踪。"""

from __future__ import annotations

import logging
import sys
import uuid

import structlog
from structlog.types import EventDict, Processor


def add_request_id(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """注入 request_id（如未在上下文中设置则生成新的）。"""
    if "request_id" not in event_dict:
        event_dict["request_id"] = str(uuid.uuid4())[:8]
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """配置结构化日志。

    使用 structlog 输出 JSON 格式日志，含 request_id、模块名、耗时等字段。
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # 共享处理器链
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        add_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # 配置标准 logging 转发到 structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors[:-1],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # 降低第三方库日志级别
    for noisy in ("uvicorn.access", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """获取结构化日志器。"""
    return structlog.get_logger(name)


def set_request_context(request_id: str) -> None:
    """在上下文中绑定 request_id（用于请求级别追踪）。"""
    structlog.contextvars.bind_contextvars(request_id=request_id)
