"""Domain-to-HTTP error boundary with secret-safe messages."""

import re
from collections.abc import Mapping
from typing import Any

_SECRET_PATTERNS = (
    re.compile(r"(?i)(ENTSOE_API_TOKEN|OPENAI_API_KEY|GEMINI_API_KEY|authorization)\s*[=:]\s*\S+"),
    re.compile(r"(?i)bearer\s+\S+"),
)


def redact_secrets(value: str) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Any | None = None,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = redact_secrets(message)
        self.details = _safe_details(details)
        super().__init__(self.message)


def _safe_details(details: Any | None) -> Any | None:
    if details is None:
        return None
    if isinstance(details, str):
        return redact_secrets(details)
    if isinstance(details, Mapping):
        return {
            str(key): "[REDACTED]"
            if any(
                token in str(key).lower() for token in ("key", "token", "secret", "authorization")
            )
            else _safe_details(value)
            for key, value in details.items()
        }
    if isinstance(details, list | tuple):
        return [_safe_details(item) for item in details]
    return details


class NotFoundError(APIError):
    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(404, "not_found", f"{resource} '{identifier}' was not found.")


class DataUnavailableError(APIError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "data_unavailable", message)


class FeatureUnavailableError(APIError):
    def __init__(self, message: str) -> None:
        super().__init__(503, "feature_unavailable", message)


class SimulationDisabledError(APIError):
    def __init__(self) -> None:
        super().__init__(403, "simulation_control_disabled", "Simulation control API is disabled.")
