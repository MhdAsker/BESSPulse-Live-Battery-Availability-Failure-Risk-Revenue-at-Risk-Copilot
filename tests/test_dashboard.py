"""Dashboard client and presentation contracts without live services."""

import importlib
import math
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from dashboard.api_client import BESSPulseAPIClient, DashboardAPIError
from dashboard.components.badges import PROVENANCE_STYLES, badge_html, calibration_badge
from dashboard.config import DashboardSettings
from dashboard.pages.revenue_at_risk import DISCLAIMER
from dashboard.utils import (
    format_eur,
    format_number,
    format_percent,
    format_timestamp,
    operational_status,
    priority_presentation,
)

NOW = "2026-10-01T12:00:00Z"
PROVENANCE = {
    "provenance_type": "DERIVED",
    "source": "test",
    "source_version": "v1",
    "generated_at_utc": NOW,
}
FRESHNESS = {"as_of_utc": NOW, "age_seconds": 0, "is_stale": False}


def _responses() -> dict[str, Any]:
    return {
        "/health": {
            "status": "healthy",
            "service": "BESSPulse API",
            "version": "0.1.0",
            "liveness": "alive",
            "readiness": "ready",
            "database_status": "available",
            "model_status": "available",
            "market_data_status": "available",
            "components": {"database": "available"},
            "timestamp_utc": NOW,
        },
        "/assets": [
            {
                "asset_id": "BESS-001",
                "name": "Asset",
                "rated_power_mw": 20,
                "rated_energy_mwh": 40,
                "pcs_count": 4,
                "rack_count": 32,
                "status": "configured",
                "data_provenance": {**PROVENANCE, "provenance_type": "SIMULATED"},
            }
        ],
        "/assets/BESS-001/status": {
            "asset_id": "BESS-001",
            "timestamp_utc": NOW,
            "rated_power_mw": 20,
            "rated_energy_mwh": 40,
            "available_power_mw": 18,
            "available_energy_mwh": 20,
            "site_soc": 0.5,
            "rte": 0.91,
            "technical_availability": 0.95,
            "requested_power_availability": 1,
            "available_racks": 31,
            "available_pcs": 4,
            "active_alert_count": 1,
            "highest_priority_alert": "HIGH",
            "failure_probability_6h": 0.1,
            "failure_probability_12h": 0.2,
            "failure_probability_24h": 0.3,
            "revenue_at_risk_eur": 100,
            "telemetry_timestamp_utc": NOW,
            "risk_prediction_timestamp_utc": NOW,
            "availability_timestamp_utc": NOW,
            "market_timestamp_utc": NOW,
            "commercial_timestamp_utc": NOW,
            "freshness": {"telemetry": FRESHNESS},
            "provenance": [{**PROVENANCE, "provenance_type": "SIMULATED"}],
        },
        "/assets/BESS-001/availability": {
            "asset_id": "BESS-001",
            "timestamp_utc": NOW,
            "technical_availability": 0.95,
            "rack_technical_availability": 0.96,
            "pcs_technical_availability": 1,
            "available_discharge_power_mw": 18,
            "available_charge_power_mw": 17,
            "discharge_power_availability": 0.9,
            "charge_power_availability": 0.85,
            "available_discharge_energy_mwh": 20,
            "available_charge_energy_mwh": 12,
            "discharge_energy_availability": 0.625,
            "charge_energy_availability": 0.375,
            "requested_power_availability": 1,
            "sustainable_request_duration_hours": 4,
            "limiting_factor": "POWER",
            "limiting_factors": ["POWER"],
            "limiting_components": ["PCS-01"],
            "available_racks": 31,
            "available_pcs": 4,
            "freshness": FRESHNESS,
            "data_provenance": PROVENANCE,
        },
        "/assets/BESS-001/delivery-risk": {
            "asset_id": "BESS-001",
            "timestamp_utc": NOW,
            "failure_probability_6h": 0.1,
            "failure_probability_12h": 0.2,
            "failure_probability_24h": 0.3,
            "model_version_6h": "risk-6",
            "model_version_12h": "risk-12",
            "model_version_24h": "risk-24",
            "feature_set_version": "v1",
            "model_confidence": None,
            "calibration": [
                {"horizon_hours": 6, "status": "validated", "limitation": None},
                {"horizon_hours": 12, "status": "validated", "limitation": None},
                {"horizon_hours": 24, "status": "limited", "limitation": "poor calibration"},
            ],
            "freshness": FRESHNESS,
            "data_provenance": {**PROVENANCE, "provenance_type": "MODEL_PREDICTION"},
        },
        "/assets/BESS-001/revenue-risk": {
            "asset_id": "BESS-001",
            "start_timestamp_utc": NOW,
            "end_timestamp_utc": NOW,
            "market_mode": "historical",
            "price_source": "REAL",
            "healthy_gross_revenue_eur": 1000,
            "healthy_degradation_cost_eur": 20,
            "healthy_net_revenue_eur": 980,
            "current_gross_revenue_eur": 900,
            "current_degradation_cost_eur": 18,
            "current_net_revenue_eur": 882,
            "revenue_at_risk_eur": 98,
            "revenue_at_risk_fraction": 0.1,
            "attribution": {
                "power_derating_impact_eur": 40,
                "energy_capacity_impact_eur": 20,
                "efficiency_impact_eur": 25,
                "availability_impact_eur": 10,
                "interaction_unattributed_eur": 3,
            },
            "calculation_version": "v1",
            "freshness": FRESHNESS,
            "data_provenance": {**PROVENANCE, "provenance_type": "COUNTERFACTUAL"},
            "disclaimer": DISCLAIMER,
        },
        "/racks/PCS-01-RACK-01": {
            "rack_id": "PCS-01-RACK-01",
            "pcs_id": "PCS-01",
            "timestamp_utc": NOW,
            "soc": 0.5,
            "soh_proxy": 0.98,
            "voltage_v": 1200,
            "current_a": 10,
            "temperature_mean_c": 28,
            "temperature_min_c": 27,
            "temperature_max_c": 29,
            "temperature_spread_c": 2,
            "voltage_spread_v": 3,
            "requested_power_mw": 0.5,
            "actual_power_mw": 0.48,
            "rte": 0.91,
            "availability": True,
            "alarm_code": None,
            "operating_state": "DISCHARGING",
            "peer_temperature_mean_c": 27.8,
            "peer_voltage_mean_v": 1201,
            "expected_temperature_c": 27.5,
            "thermal_residual_c": 0.5,
            "anomaly_summary": {"active": False, "peak_score": None, "detector": None},
            "freshness": FRESHNESS,
            "data_provenance": {**PROVENANCE, "provenance_type": "SIMULATED"},
        },
        "/racks/PCS-01-RACK-01/anomalies": {"total": 0, "limit": 50, "offset": 0, "items": []},
        "/alerts": {"total": 0, "limit": 50, "offset": 0, "items": []},
        "/market/latest": {
            "market_region": "DE-LU",
            "observed": {
                "timestamp_utc": NOW,
                "day_ahead_price_eur_per_mwh": 80,
                "load_mw": None,
                "wind_generation_mw": None,
                "solar_generation_mw": None,
                "renewable_generation_mw": None,
            },
            "forecast": None,
            "derived": {"residual_load_mw": None, "renewable_share": None},
            "freshness": FRESHNESS,
            "observed_provenance": {**PROVENANCE, "provenance_type": "REAL"},
            "forecast_provenance": None,
            "derived_provenance": PROVENANCE,
        },
    }


@pytest.fixture
def client() -> BESSPulseAPIClient:
    responses = _responses()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses[request.url.path])

    return BESSPulseAPIClient("http://test", transport=httpx.MockTransport(handler), retries=0)


def test_app_and_all_ten_pages_import() -> None:
    importlib.import_module("dashboard.app")
    for module in (
        "fleet_overview",
        "live_asset",
        "rack_heatmap",
        "ml_health",
        "delivery_risk",
        "market_context",
        "revenue_at_risk",
        "alerts",
        "ai_copilot",
        "model_monitoring",
    ):
        importlib.import_module(f"dashboard.pages.{module}")


def test_api_configuration_and_all_parsers(client: BESSPulseAPIClient) -> None:
    assert DashboardSettings(
        besspulse_api_url="https://example.test/v1"
    ).besspulse_api_url.endswith("/v1")
    assert client.health().readiness == "ready"
    assert client.assets()[0].rack_count == 32
    assert client.asset_status("BESS-001").site_soc == 0.5
    assert client.availability("BESS-001").available_charge_power_mw == 17
    risk = client.delivery_risk("BESS-001")
    assert risk.failure_probability_24h == 0.3
    assert risk.calibration[2].status == "limited"
    assert client.revenue_risk("BESS-001").disclaimer == DISCLAIMER
    assert client.rack("PCS-01-RACK-01").thermal_residual_c == 0.5
    assert client.anomalies("PCS-01-RACK-01").total == 0
    assert client.alerts(status="OPEN").total == 0
    assert client.market_latest().observed.day_ahead_price_eur_per_mwh == 80  # type: ignore[union-attr]


def test_filters_serialize_without_all_or_none() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"total": 0, "limit": 50, "offset": 0, "items": []})

    client = BESSPulseAPIClient("http://test", transport=httpx.MockTransport(handler), retries=0)
    client.alerts(status="OPEN", priority_level="ALL", component_id=None, limit=50)
    assert "status=OPEN" in seen[0] and "limit=50" in seen[0]
    assert "priority_level" not in seen[0] and "component_id" not in seen[0]


def test_client_503_404_timeout_offline_and_malformed() -> None:
    for status, category in ((503, "service_unavailable"), (404, "not_found")):
        transport = httpx.MockTransport(
            lambda request, code=status: httpx.Response(code, json={"message": "safe"})
        )
        with pytest.raises(DashboardAPIError, match="safe") as error:
            BESSPulseAPIClient("http://test", transport=transport, retries=0).health()
        assert error.value.category == category

    timeout = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("late")))
    with pytest.raises(DashboardAPIError) as error:
        BESSPulseAPIClient("http://test", transport=timeout, retries=0).health()
    assert error.value.category == "timeout"

    offline = httpx.MockTransport(lambda request: (_ for _ in ()).throw(httpx.ConnectError("down")))
    with pytest.raises(DashboardAPIError) as error:
        BESSPulseAPIClient("http://test", transport=offline, retries=0).health()
    assert error.value.category == "offline"

    malformed = httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "healthy"}))
    with pytest.raises(DashboardAPIError) as error:
        BESSPulseAPIClient("http://test", transport=malformed, retries=0).health()
    assert error.value.category == "malformed_response"

    copilot_unavailable = httpx.MockTransport(
        lambda request: httpx.Response(503, json={"message": "AI Copilot is not configured."})
    )
    with pytest.raises(DashboardAPIError, match="AI Copilot is not configured") as error:
        BESSPulseAPIClient("http://test", transport=copilot_unavailable, retries=0).copilot_query(
            "BESS-001", "What changed?"
        )
    assert error.value.status_code == 503


def test_presentation_mappings_and_scientific_labels() -> None:
    assert set(PROVENANCE_STYLES) == {
        "REAL",
        "SIMULATED",
        "DERIVED",
        "MODEL_PREDICTION",
        "COUNTERFACTUAL",
        "DECISION_SUPPORT",
    }
    assert "MODEL PREDICTION" in badge_html("MODEL_PREDICTION")
    assert "LIMITED" in calibration_badge("limited")
    assert priority_presentation("CRITICAL") == ("✹", "critical")
    assert operational_status("HIGH", 1) == "DEGRADED"
    assert operational_status(None, 1) == "NORMAL"
    assert DISCLAIMER == "Counterfactual historical simulation, not actual commercial P&L."


def test_safe_number_and_timezone_formatting() -> None:
    assert format_number(None) == "—"
    assert format_number(math.nan) == "—"
    assert format_number(math.inf) == "—"
    assert format_percent(0.524) == "52.4%"
    assert format_eur(1480) == "€1,480"
    assert format_timestamp(datetime(2026, 1, 1, tzinfo=UTC), "UTC").endswith("UTC")
    assert format_timestamp("2026-01-01T00:00:00Z", "Europe/Berlin").endswith("Europe/Berlin")
