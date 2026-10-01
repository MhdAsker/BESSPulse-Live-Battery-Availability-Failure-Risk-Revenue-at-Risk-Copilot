"""Secure ENTSO-E DE-LU client, ingestion service, and development CLI."""

import argparse
import logging
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from email.utils import parsedate_to_datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy.orm import Session

from besspulse.config import MarketConfig, SimulationConfig
from besspulse.database import create_schema, create_session_factory
from data.exceptions import (
    EntsoeAuthenticationError,
    EntsoeDataUnavailableError,
    EntsoeError,
    EntsoeRateLimitError,
    EntsoeRequestError,
    EntsoeValidationError,
)
from data.parser import aggregate_wind, parse_entsoe_xml
from data.persistence import persist_market_observations
from data.schemas import (
    Dataset,
    DatasetFailure,
    FetchResult,
    IngestionSummary,
    MarketObservation,
)
from data.storage import RawEntsoeStore
from data.validation import validate_series_continuity

LOGGER = logging.getLogger(__name__)
_SENSITIVE_QUERY = re.compile(r"(?i)(securityToken|api[_-]?key|token)=([^&\s]+)")


def redact_sensitive(value: str, secret: str | None = None) -> str:
    """Remove an exact credential and common token-like query parameters."""

    safe = value.replace(secret, "[REDACTED]") if secret else value
    return _SENSITIVE_QUERY.sub(r"\1=[REDACTED]", safe)


class _SecretRedactionFilter(logging.Filter):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        record.msg = redact_sensitive(rendered, self.secret)
        record.args = ()
        return True


def _install_http_log_redaction(secret: str) -> None:
    """Sanitize URLs produced by optional httpx/httpcore diagnostic logging."""

    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).addFilter(_SecretRedactionFilter(secret))
    # httpcore child loggers can expose request targets at DEBUG; block that channel.
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _utc_boundary(value: datetime, name: str) -> datetime:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise EntsoeValidationError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class _DatasetQuery:
    document_type: str
    process_type: str | None
    domain_parameter: str


_QUERIES: dict[Dataset, _DatasetQuery] = {
    Dataset.DAY_AHEAD_PRICES: _DatasetQuery("A44", None, "price"),
    Dataset.ACTUAL_LOAD: _DatasetQuery("A65", "A16", "outBiddingZone_Domain"),
    Dataset.LOAD_FORECAST: _DatasetQuery("A65", "A01", "outBiddingZone_Domain"),
    Dataset.ACTUAL_GENERATION: _DatasetQuery("A75", "A16", "in_Domain"),
    Dataset.GENERATION_FORECAST: _DatasetQuery("A69", "A01", "in_Domain"),
}


class EntsoeClient:
    """Bounded-retry HTTP retrieval kept separate from deterministic parsing."""

    def __init__(
        self,
        token: str,
        *,
        market_config: MarketConfig | None = None,
        raw_store: RawEntsoeStore | None = None,
        http_client: httpx.Client | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 0.5,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not token.strip():
            raise EntsoeAuthenticationError("ENTSOE_API_TOKEN is required in the environment")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self._token = token
        _install_http_log_redaction(token)
        self.market_config = market_config or MarketConfig()
        self.raw_store = raw_store or RawEntsoeStore()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleeper
        self._owns_client = http_client is None
        self._raw_artifacts_created = 0
        self._http = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "EntsoeClient":
        """Load the sole credential only from ENTSOE_API_TOKEN."""

        return cls(os.getenv("ENTSOE_API_TOKEN", ""), **kwargs)

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> "EntsoeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch_day_ahead_prices(self, start: datetime, end: datetime) -> FetchResult:
        return self.fetch(Dataset.DAY_AHEAD_PRICES, start, end)

    def fetch_actual_load(self, start: datetime, end: datetime) -> FetchResult:
        return self.fetch(Dataset.ACTUAL_LOAD, start, end)

    def fetch_load_forecast(self, start: datetime, end: datetime) -> FetchResult:
        return self.fetch(Dataset.LOAD_FORECAST, start, end)

    def fetch_generation(self, start: datetime, end: datetime) -> FetchResult:
        return self.fetch(Dataset.ACTUAL_GENERATION, start, end)

    def fetch_generation_forecast(self, start: datetime, end: datetime) -> FetchResult:
        return self.fetch(Dataset.GENERATION_FORECAST, start, end)

    def fetch(self, dataset: Dataset, start: datetime, end: datetime) -> FetchResult:
        start_utc, end_utc = self._validate_interval(start, end)
        content, status, retrieved_at = self._request(dataset, start_utc, end_utc)
        artifact = self.raw_store.save(
            content,
            dataset=dataset,
            region=self.market_config.market_region,
            requested_start=start_utc,
            requested_end=end_utc,
            retrieved_at_utc=retrieved_at,
            http_status=status,
        )
        self._raw_artifacts_created += int(artifact.created)
        observations = parse_entsoe_xml(
            content,
            dataset,
            self.market_config.market_region,
            retrieved_at,
            artifact.content_hash,
        )
        if dataset in {Dataset.ACTUAL_GENERATION, Dataset.GENERATION_FORECAST}:
            observations = (*observations, *aggregate_wind(observations))
            observations = tuple(
                sorted(
                    observations,
                    key=lambda row: (row.timestamp_utc, row.metric, row.timeseries_mrid),
                )
            )
        return FetchResult(dataset=dataset, observations=observations, raw_artifact=artifact)

    def _request(
        self, dataset: Dataset, start_utc: datetime, end_utc: datetime
    ) -> tuple[bytes, int, datetime]:
        params = self._query_parameters(dataset, start_utc, end_utc)
        for attempt in range(self.max_retries + 1):
            started = time.monotonic()
            try:
                response = self._http.get(self.market_config.entsoe_endpoint, params=params)
            except httpx.RequestError:
                LOGGER.warning(
                    "ENTSO-E network failure: dataset=%s region=%s attempt=%d",
                    dataset.value,
                    self.market_config.market_region,
                    attempt + 1,
                )
                if attempt >= self.max_retries:
                    raise EntsoeRequestError(
                        "ENTSO-E network request failed after bounded retries"
                    ) from None
                self._sleep(self.backoff_seconds * (2**attempt))
                continue
            duration = time.monotonic() - started
            LOGGER.info(
                "ENTSO-E response: dataset=%s region=%s attempt=%d status=%d duration_s=%.3f",
                dataset.value,
                self.market_config.market_region,
                attempt + 1,
                response.status_code,
                duration,
            )
            if response.status_code in {401, 403}:
                raise EntsoeAuthenticationError("ENTSO-E authentication failed")
            if response.status_code == 429:
                if attempt >= self.max_retries:
                    raise EntsoeRateLimitError("ENTSO-E rate limit persisted after bounded retries")
                self._sleep(self._retry_delay(response, attempt))
                continue
            if 500 <= response.status_code <= 599:
                if attempt >= self.max_retries:
                    raise EntsoeRequestError(
                        "ENTSO-E server failure after bounded retries: "
                        f"status={response.status_code}"
                    )
                self._sleep(self.backoff_seconds * (2**attempt))
                continue
            if response.status_code >= 400:
                raise EntsoeRequestError(f"ENTSO-E request rejected: status={response.status_code}")
            return response.content, response.status_code, datetime.now(UTC)
        raise AssertionError("bounded retry loop exhausted unexpectedly")

    def _query_parameters(
        self, dataset: Dataset, start_utc: datetime, end_utc: datetime
    ) -> dict[str, str]:
        query = _QUERIES[dataset]
        params = {
            "securityToken": self._token,
            "documentType": query.document_type,
            "periodStart": start_utc.strftime("%Y%m%d%H%M"),
            "periodEnd": end_utc.strftime("%Y%m%d%H%M"),
        }
        if query.process_type:
            params["processType"] = query.process_type
        domain = self.market_config.entsoe_domain
        if query.domain_parameter == "price":
            params["in_Domain"] = domain
            params["out_Domain"] = domain
        else:
            params[query.domain_parameter] = domain
        return params

    def _retry_delay(self, response: httpx.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                try:
                    retry_at: datetime = parsedate_to_datetime(retry_after).astimezone(UTC)
                    return min(
                        60.0,
                        max(0.0, (retry_at - datetime.now(UTC)).total_seconds()),
                    )
                except (TypeError, ValueError):
                    pass
        return self.backoff_seconds * float(2**attempt)

    @property
    def raw_artifacts_created(self) -> int:
        return self._raw_artifacts_created

    @staticmethod
    def _validate_interval(start: datetime, end: datetime) -> tuple[datetime, datetime]:
        start_utc = _utc_boundary(start, "start")
        end_utc = _utc_boundary(end, "end")
        if start_utc >= end_utc:
            raise EntsoeValidationError("start must be before exclusive end")
        return start_utc, end_utc


def _continuity_warnings(observations: Iterable[MarketObservation]) -> list[str]:
    groups: dict[tuple[str, str], list[MarketObservation]] = {}
    for row in observations:
        if row.data_provenance != "REAL":
            continue
        groups.setdefault((row.metric.value, row.timeseries_mrid), []).append(row)
    warnings: list[str] = []
    for (metric, series_id), rows in groups.items():
        report = validate_series_continuity(rows)
        if report.missing_timestamps:
            warnings.append(
                f"metric={metric} series={series_id} missing={len(report.missing_timestamps)}"
            )
        if report.duplicate_timestamps:
            warnings.append(
                f"metric={metric} series={series_id} duplicates={len(report.duplicate_timestamps)}"
            )
        negative_nonprice = sum(
            row.value < 0 and row.metric.value != "day_ahead_price" for row in rows
        )
        if negative_nonprice:
            warnings.append(
                f"metric={metric} series={series_id} negative_values={negative_nonprice}"
            )
    return warnings


def ingest_entsoe_market_data(
    start: datetime,
    end: datetime,
    *,
    datasets: tuple[Dataset, ...] = tuple(Dataset),
    client: EntsoeClient | None = None,
    session: Session | None = None,
) -> IngestionSummary:
    """Fetch, archive, normalize, validate, and idempotently persist `[start, end)`."""

    owned_client = client is None
    active_client = client or EntsoeClient.from_env()
    owned_session = session is None
    if session is None:
        create_schema()
        session = create_session_factory()()
    succeeded: list[Dataset] = []
    unavailable: list[Dataset] = []
    failed: list[DatasetFailure] = []
    warnings: list[str] = []
    observations_written = 0
    initial_raw_count = active_client.raw_artifacts_created
    try:
        start_utc, end_utc = active_client._validate_interval(start, end)
        for dataset in datasets:
            try:
                result = active_client.fetch(dataset, start_utc, end_utc)
                warnings.extend(_continuity_warnings(result.observations))
                observations_written += persist_market_observations(session, result.observations)
                succeeded.append(dataset)
            except EntsoeAuthenticationError:
                raise
            except EntsoeDataUnavailableError:
                unavailable.append(dataset)
            except EntsoeError as exc:
                failed.append(
                    DatasetFailure(
                        dataset=dataset,
                        error_type=type(exc).__name__,
                        message=redact_sensitive(str(exc), active_client._token),
                    )
                )
        return IngestionSummary(
            requested_start=start_utc,
            requested_end=end_utc,
            market_region=active_client.market_config.market_region,
            datasets_requested=datasets,
            datasets_succeeded=tuple(succeeded),
            datasets_unavailable=tuple(unavailable),
            datasets_failed=tuple(failed),
            observations_written=observations_written,
            raw_files_written=active_client.raw_artifacts_created - initial_raw_count,
            validation_warnings=tuple(warnings),
        )
    finally:
        if owned_session:
            session.close()
        if owned_client:
            active_client.close()


def _parse_cli_boundary(value: str, timezone_name: str) -> datetime:
    """Interpret date-only CLI values as local market midnight; require TZ on datetimes."""

    try:
        if "T" not in value:
            return datetime.combine(
                date.fromisoformat(value), datetime_time.min, ZoneInfo(timezone_name)
            )
        result = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date/datetime: {value}") from exc
    if result.tzinfo is None:
        raise argparse.ArgumentTypeError("Datetime values must include a UTC offset")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest REAL ENTSO-E DE-LU market data")
    parser.add_argument("--start", required=True, help="Inclusive date or aware datetime")
    parser.add_argument("--end", required=True, help="Exclusive date or aware datetime")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate interval/configuration without network"
    )
    args = parser.parse_args(argv)
    market = SimulationConfig().market
    start = _parse_cli_boundary(args.start, market.market_timezone)
    end = _parse_cli_boundary(args.end, market.market_timezone)
    start_utc, end_utc = EntsoeClient._validate_interval(start, end)
    if args.dry_run:
        print(
            f"valid interval region={market.market_region} "
            f"start={start_utc.isoformat()} end={end_utc.isoformat()}"
        )
        return 0
    summary = ingest_entsoe_market_data(start, end)
    print(summary.model_dump_json(indent=2))
    return 0 if not summary.datasets_failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
