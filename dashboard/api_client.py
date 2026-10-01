"""Typed FastAPI client with bounded GET retries and controlled errors."""

from time import sleep
from typing import Any, TypeVar

import httpx
from api.schemas.alerts import AlertPage
from api.schemas.anomalies import AnomalyPage
from api.schemas.assets import AssetStatus, AssetSummary
from api.schemas.availability import AvailabilityResponse
from api.schemas.commercial import RevenueRiskResponse
from api.schemas.copilot import CopilotQueryResponse
from api.schemas.health import HealthResponse
from api.schemas.market import MarketLatestResponse
from api.schemas.racks import RackResponse
from api.schemas.risk import DeliveryRiskResponse
from pydantic import BaseModel, TypeAdapter, ValidationError

T = TypeVar("T", bound=BaseModel)


class DashboardAPIError(Exception):
    def __init__(self, category: str, message: str, *, status_code: int | None = None) -> None:
        self.category = category
        self.status_code = status_code
        super().__init__(message)


class BESSPulseAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 5.0,
        retries: int = 1,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._transport = transport

    def health(self) -> HealthResponse:
        return self._get("/health", HealthResponse)

    def assets(self) -> tuple[AssetSummary, ...]:
        payload = self._request("GET", "/assets")
        try:
            return TypeAdapter(tuple[AssetSummary, ...]).validate_python(payload)
        except ValidationError as exc:
            raise DashboardAPIError(
                "malformed_response", "API returned an invalid asset list."
            ) from exc

    def asset_status(self, asset_id: str) -> AssetStatus:
        return self._get(f"/assets/{asset_id}/status", AssetStatus)

    def availability(self, asset_id: str) -> AvailabilityResponse:
        return self._get(f"/assets/{asset_id}/availability", AvailabilityResponse)

    def delivery_risk(self, asset_id: str) -> DeliveryRiskResponse:
        return self._get(f"/assets/{asset_id}/delivery-risk", DeliveryRiskResponse)

    def revenue_risk(self, asset_id: str) -> RevenueRiskResponse:
        return self._get(f"/assets/{asset_id}/revenue-risk", RevenueRiskResponse)

    def rack(self, rack_id: str) -> RackResponse:
        return self._get(f"/racks/{rack_id}", RackResponse)

    def anomalies(self, rack_id: str, **filters: Any) -> AnomalyPage:
        return self._get(f"/racks/{rack_id}/anomalies", AnomalyPage, params=_clean(filters))

    def alerts(self, **filters: Any) -> AlertPage:
        return self._get("/alerts", AlertPage, params=_clean(filters))

    def market_latest(self) -> MarketLatestResponse:
        return self._get("/market/latest", MarketLatestResponse)

    def copilot_query(
        self, asset_id: str, query: str, conversation_id: str | None = None
    ) -> CopilotQueryResponse:
        return self._post(
            "/copilot/query",
            CopilotQueryResponse,
            {"asset_id": asset_id, "query": query, "conversation_id": conversation_id},
        )

    def _get(self, path: str, model: type[T], *, params: dict[str, Any] | None = None) -> T:
        payload = self._request("GET", path, params=params)
        try:
            return model.model_validate(payload)
        except ValidationError as exc:
            raise DashboardAPIError(
                "malformed_response", "API returned an invalid response."
            ) from exc

    def _post(self, path: str, model: type[T], payload: dict[str, Any]) -> T:
        response = self._request("POST", path, json=payload)
        try:
            return model.model_validate(response)
        except ValidationError as exc:
            raise DashboardAPIError(
                "malformed_response", "API returned an invalid response."
            ) from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        attempts = self.retries + 1 if method == "GET" else 1
        for attempt in range(attempts):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout_seconds,
                    transport=self._transport,
                ) as client:
                    response = client.request(method, path, **kwargs)
                if response.is_success:
                    return response.json()
                message = _error_message(response)
                raise DashboardAPIError(
                    "not_found"
                    if response.status_code == 404
                    else "service_unavailable"
                    if response.status_code == 503
                    else "api_error",
                    message,
                    status_code=response.status_code,
                )
            except httpx.TimeoutException as exc:
                error = DashboardAPIError("timeout", "BESSPulse API request timed out.")
                if attempt == attempts - 1:
                    raise error from exc
            except httpx.RequestError as exc:
                error = DashboardAPIError("offline", "BESSPulse API is unreachable.")
                if attempt == attempts - 1:
                    raise error from exc
            if attempt < attempts - 1:
                sleep(0.1)
        raise DashboardAPIError("offline", "BESSPulse API is unreachable.")


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"API request failed with HTTP {response.status_code}."
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return str(body["message"])
    return f"API request failed with HTTP {response.status_code}."


def _clean(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != "ALL"}
