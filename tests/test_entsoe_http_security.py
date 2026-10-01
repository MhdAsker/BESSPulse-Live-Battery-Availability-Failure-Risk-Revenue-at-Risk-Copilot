import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from data.entsoe import EntsoeClient, redact_sensitive
from data.exceptions import EntsoeAuthenticationError, EntsoeRateLimitError
from data.schemas import Dataset
from data.storage import RawEntsoeStore

FIXTURES = Path(__file__).parent / "fixtures" / "entsoe"
START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 2, tzinfo=UTC)
TOKEN = "unit-test-secret-token-value"


def client_for(handler, tmp_path, **kwargs):
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport)
    return EntsoeClient(
        TOKEN,
        http_client=http,
        raw_store=RawEntsoeStore(tmp_path),
        sleeper=lambda _: None,
        **kwargs,
    )


def test_success_request_uses_central_de_lu_domain_and_safe_raw_filename(tmp_path) -> None:
    captured = {}

    def handler(request):
        captured.update(dict(request.url.params))
        return httpx.Response(200, content=(FIXTURES / "day_ahead_prices.xml").read_bytes())

    client = client_for(handler, tmp_path)
    result = client.fetch_day_ahead_prices(START, END)
    assert captured["in_Domain"] == "10Y1001A1001A82H"
    assert captured["out_Domain"] == "10Y1001A1001A82H"
    assert captured["securityToken"] == TOKEN
    assert TOKEN not in result.raw_artifact.xml_path.name
    assert TOKEN not in result.raw_artifact.metadata_path.read_text(encoding="utf-8")


def test_token_is_not_logged_or_exposed_by_authentication_failure(tmp_path, caplog) -> None:
    def handler(request):
        return httpx.Response(401, request=request)

    client = client_for(handler, tmp_path)
    with caplog.at_level(logging.DEBUG), pytest.raises(EntsoeAuthenticationError) as error:
        client.fetch_actual_load(START, END)
    assert TOKEN not in str(error.value)
    assert TOKEN not in caplog.text


def test_sanitizer_redacts_sensitive_query_parameters() -> None:
    unsafe = f"https://example.test/api?securityToken={TOKEN}&documentType=A44"
    sanitized = redact_sensitive(unsafe, TOKEN)
    assert TOKEN not in sanitized
    assert "securityToken=[REDACTED]" in sanitized


def test_rate_limit_respects_retry_after_then_fails_transparently(tmp_path) -> None:
    delays = []
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, headers={"Retry-After": "2"}, request=request)

    transport = httpx.MockTransport(handler)
    client = EntsoeClient(
        TOKEN,
        http_client=httpx.Client(transport=transport),
        raw_store=RawEntsoeStore(tmp_path),
        max_retries=2,
        sleeper=delays.append,
    )
    with pytest.raises(EntsoeRateLimitError):
        client.fetch_actual_load(START, END)
    assert calls == 3
    assert delays == [2.0, 2.0]


def test_raw_store_deduplicates_identical_content_and_hashes_metadata(tmp_path) -> None:
    content = (FIXTURES / "day_ahead_prices.xml").read_bytes()
    store = RawEntsoeStore(tmp_path)
    kwargs = {
        "dataset": Dataset.DAY_AHEAD_PRICES,
        "region": "DE-LU",
        "requested_start": START,
        "requested_end": END,
        "retrieved_at_utc": START,
        "http_status": 200,
    }
    first = store.save(content, **kwargs)
    second = store.save(content, **kwargs)
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert first.created is True
    assert second.created is False
    assert first.xml_path.read_bytes() == content
    assert metadata["content_hash_sha256"] == first.content_hash
    assert metadata["data_provenance"] == "REAL"
    assert len(list(tmp_path.glob("*.xml"))) == 1
