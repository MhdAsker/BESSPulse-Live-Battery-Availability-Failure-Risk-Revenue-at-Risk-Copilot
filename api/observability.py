"""Minimal production JSON logging without secret-bearing configuration."""

import json
import logging
from datetime import UTC, datetime
from typing import Any


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": "besspulse-api",
            "event": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status",
            "duration_ms",
            "database_backend",
            "exception_type",
            "asset_id",
            "model_version",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(level: str, json_logs: bool) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    if json_logs:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        root.handlers.clear()
        root.addHandler(handler)
