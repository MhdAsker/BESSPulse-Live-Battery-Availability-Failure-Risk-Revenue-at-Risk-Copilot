from datetime import UTC, datetime

import pytest


@pytest.fixture
def start() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)
