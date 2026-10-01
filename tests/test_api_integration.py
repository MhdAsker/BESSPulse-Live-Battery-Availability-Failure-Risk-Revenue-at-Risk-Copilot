"""CI-sized API integration and contract tests with no live services."""

import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from api.config import APISettings
from api.dependencies import get_asset_service
from api.main import create_app
from api.services import AssetService
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from besspulse.database import (
    AlertModel,
    AnomalyEventModel,
    Asset,
    AvailabilitySnapshotModel,
    Base,
    BatteryTelemetryModel,
    MarketDataModel,
    ModelPredictionModel,
    PCSModel,
    PricePredictionModel,
    RackModel,
    RackTelemetryModel,
    RevenueAtRiskModel,
)


@pytest.fixture
def api_bundle(tmp_path: Path) -> Iterator[tuple[TestClient, sessionmaker[Session], APISettings]]:
    url = f"sqlite:///{tmp_path / 'api.db'}"
    engine = create_engine(url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    _seed(factory)
    settings = APISettings(
        database_url=url,
        enable_simulation_control_api=True,
        api_stale_after_seconds=3600,
        api_max_page_size=100,
    )
    app = create_app(settings, factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, factory, settings
    engine.dispose()


def _seed(factory: sessionmaker[Session]) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    with factory() as session:
        asset = Asset(
            asset_id="BESS-001",
            rated_power_mw=20,
            rated_energy_mwh=40,
            data_provenance="SIMULATED",
        )
        session.add(asset)
        session.flush()
        pcs = PCSModel(pcs_id="PCS-01", asset_pk=asset.id, rated_power_mw=5)
        session.add(pcs)
        session.flush()
        session.add(
            RackModel(
                rack_id="PCS-01-RACK-01",
                pcs_pk=pcs.id,
                rated_power_mw=0.625,
                rated_energy_mwh=1.25,
            )
        )
        session.add(
            BatteryTelemetryModel(
                timestamp_utc=now,
                asset_id="BESS-001",
                site_requested_power_mw=5,
                site_actual_power_mw=4.8,
                available_power_mw=18,
                available_energy_mwh=20,
                site_soc=0.6,
                rte=0.91,
                ambient_temperature_c=20,
                available_racks=31,
                available_pcs=4,
                operating_mode="DISCHARGING",
                alarm_count=1,
                data_provenance="SIMULATED",
            )
        )
        session.add(
            RackTelemetryModel(
                timestamp_utc=now,
                rack_id="PCS-01-RACK-01",
                pcs_id="PCS-01",
                soc=0.61,
                soh_proxy=0.97,
                voltage_v=1198,
                current_a=50,
                temperature_mean_c=28,
                temperature_min_c=27,
                temperature_max_c=29,
                temperature_spread_c=2,
                voltage_spread_v=3,
                requested_power_mw=0.5,
                actual_power_mw=0.48,
                charge_energy_mwh=1,
                discharge_energy_mwh=2,
                cumulative_throughput_mwh=50,
                equivalent_full_cycles=20,
                rte=0.9,
                availability=True,
                alarm_code=None,
                operating_state="DISCHARGING",
                data_provenance="SIMULATED",
            )
        )
        session.add(
            AvailabilitySnapshotModel(
                timestamp_utc=now,
                asset_id="BESS-001",
                technical_availability=0.95,
                rack_technical_availability=0.96875,
                pcs_technical_availability=1,
                known_component_fraction=1,
                available_discharge_power_mw=18,
                available_charge_power_mw=17,
                available_discharge_energy_mwh=20,
                available_charge_energy_mwh=12,
                discharge_power_availability=0.9,
                charge_power_availability=0.85,
                discharge_energy_availability=0.625,
                charge_energy_availability=0.375,
                requested_power_mw=5,
                requested_power_availability=1,
                requested_energy_availability=1,
                sustainable_request_duration_hours=4,
                available_racks=31,
                unknown_racks=0,
                available_pcs=4,
                unknown_pcs=0,
                limiting_factor="POWER_DERATING",
                limiting_factors_json='["POWER_DERATING"]',
                limiting_component_ids_json='["PCS-01"]',
                data_provenance="DERIVED ENGINEERING ANALYTIC",
                created_at=now,
            )
        )
        for horizon, probability in ((6, 0.1), (12, 0.2), (24, 0.3)):
            session.add(
                ModelPredictionModel(
                    timestamp_utc=now,
                    entity_id="BESS-001",
                    model_name="delivery_risk",
                    model_version=f"delivery_risk_{horizon}h_v1",
                    prediction_name=f"failure_probability_{horizon}h",
                    prediction_value=probability,
                    actual_value=None,
                    residual_value=None,
                    prediction_horizon_hours=horizon,
                    decision_threshold=0.5,
                    predicted_class=False,
                    prediction_provenance="MODEL_PREDICTION",
                    residual_provenance=None,
                    feature_set_version="v1",
                    created_at=now,
                )
            )
        session.add(
            ModelPredictionModel(
                timestamp_utc=now,
                entity_id="PCS-01-RACK-01",
                model_name="expected_temperature",
                model_version="expected_temperature_v1",
                prediction_name="expected_temperature_c",
                prediction_value=27.5,
                actual_value=28,
                residual_value=0.5,
                prediction_horizon_hours=None,
                decision_threshold=None,
                predicted_class=None,
                prediction_provenance="MODEL_PREDICTION",
                residual_provenance="DERIVED",
                feature_set_version="v1",
                created_at=now,
            )
        )
        session.add(
            AnomalyEventModel(
                anomaly_event_id="ANOM-001",
                component_id="PCS-01-RACK-01",
                component_type="RACK",
                detector_name="vote_ensemble",
                detector_version="vote_ensemble_v1",
                start_timestamp=now - timedelta(minutes=10),
                end_timestamp=now + timedelta(minutes=10),
                peak_score=0.8,
                mean_score=0.6,
                supporting_signals_json='["thermal_residual"]',
                feature_set_version="v1",
                data_provenance="DERIVED",
                created_at=now,
            )
        )
        session.add(_commercial(now))
        session.add(_alert(now))
        for metric, value, unit in (
            ("day_ahead_price", 80.0, "EUR/MWh"),
            ("actual_load", 50000.0, "MW"),
            ("renewable_generation", 20000.0, "MW"),
        ):
            session.add(
                MarketDataModel(
                    timestamp_utc=now,
                    market_region="DE-LU",
                    metric=metric,
                    value=value,
                    unit=unit,
                    series_type="ACTUAL" if metric != "day_ahead_price" else "DAY_AHEAD",
                    source="ENTSO-E Transparency Platform",
                    data_provenance="REAL",
                    retrieved_at_utc=now,
                    raw_content_hash=f"hash-{metric}",
                    document_id="doc",
                    timeseries_mrid=metric,
                    business_type=None,
                    process_type=None,
                    psr_type=None,
                    resolution="PT15M",
                    created_at=now,
                )
            )
        session.add(
            PricePredictionModel(
                forecast_origin_utc=now,
                target_timestamp_utc=now + timedelta(hours=1),
                market_region="DE-LU",
                model_name="price_forecast",
                model_version="price_forecast_v1",
                quantile=-1,
                predicted_price_eur_per_mwh=85,
                horizon_minutes=60,
                feature_set_version="price_features_v1",
                data_provenance="MODEL_PREDICTION",
                created_at=now,
            )
        )
        session.commit()


def _commercial(now: datetime) -> RevenueAtRiskModel:
    return RevenueAtRiskModel(
        asset_id="BESS-001",
        start_timestamp_utc=now - timedelta(days=1),
        end_timestamp_utc=now,
        market_mode="historical",
        price_source="REAL ENTSO-E",
        healthy_gross_revenue_eur=1000,
        healthy_degradation_cost_eur=20,
        healthy_net_revenue_eur=980,
        current_gross_revenue_eur=900,
        current_degradation_cost_eur=18,
        current_net_revenue_eur=882,
        revenue_at_risk_eur=98,
        revenue_at_risk_fraction=0.1,
        power_derating_impact_eur=40,
        energy_capacity_impact_eur=20,
        efficiency_impact_eur=25,
        availability_impact_eur=10,
        interaction_unattributed_eur=3,
        model_or_price_version="entsoe-v1",
        capability_version="availability-v1",
        price_dataset_hash="price-hash",
        capability_dataset_hash="capability-hash",
        optimization_config_hash="optimization-hash",
        calculation_hash="calculation-hash",
        attribution_json="{}",
        data_provenance="COUNTERFACTUAL",
        disclaimer="Counterfactual historical simulation, not actual commercial P&L.",
        created_at=now,
    )


def _alert(now: datetime) -> AlertModel:
    return AlertModel(
        alert_key="alert-key",
        asset_id="BESS-001",
        component_id="PCS-01-RACK-01",
        component_type="RACK",
        alert_type="THERMAL_ISSUE",
        status="OPEN",
        opened_at_utc=now - timedelta(minutes=15),
        updated_at_utc=now,
        resolved_at_utc=None,
        priority_score=0.7,
        priority_level="HIGH",
        failure_probability=0.2,
        risk_horizon_hours=12,
        model_confidence=0.8,
        technical_severity=0.75,
        commercial_severity=0.1,
        affected_power_mw=0.5,
        affected_energy_mwh=1,
        revenue_at_risk_eur=98,
        anomaly_score=0.8,
        limiting_factor="THERMAL",
        supporting_signals_json='["thermal_residual"]',
        source_versions_json='{"alert_formula":"alert-formula-v1"}',
        data_provenance="DERIVED DECISION-SUPPORT OUTPUT",
        created_at=now,
    )


def test_health_assets_and_unknown_asset(api_bundle: tuple[TestClient, Any, Any]) -> None:
    client, _, _ = api_bundle
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["liveness"] == "alive"
    assert client.get("/api/v1/health/live").status_code == 200
    assert client.get("/api/v1/health/ready").json()["readiness"] == "ready"
    assets = client.get("/api/v1/assets")
    assert assets.status_code == 200
    assert assets.json()[0]["asset_id"] == "BESS-001"
    missing = client.get("/api/v1/assets/UNKNOWN/status")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "not_found"


def test_asset_analytics_contracts_and_provenance(api_bundle: tuple[TestClient, Any, Any]) -> None:
    client, _, _ = api_bundle
    status = client.get("/api/v1/assets/BESS-001/status")
    assert status.status_code == 200
    body = status.json()
    assert body["available_power_mw"] == 18
    assert body["active_alert_count"] == 1
    assert body["telemetry_timestamp_utc"].endswith("Z")
    assert "ground_truth_label" not in status.text

    availability = client.get("/api/v1/assets/BESS-001/availability").json()
    assert availability["available_discharge_power_mw"] != availability["available_charge_power_mw"]
    assert (
        availability["discharge_energy_availability"] != availability["charge_energy_availability"]
    )
    assert availability["data_provenance"]["provenance_type"] == "DERIVED"

    risk = client.get("/api/v1/assets/BESS-001/delivery-risk").json()
    assert all(0 <= risk[f"failure_probability_{h}h"] <= 1 for h in (6, 12, 24))
    assert risk["model_version_6h"] == "delivery_risk_6h_v1"
    assert risk["calibration"][2]["status"] == "limited"
    assert risk["data_provenance"]["provenance_type"] == "MODEL_PREDICTION"
    assert "failure_within" not in str(risk)

    commercial = client.get("/api/v1/assets/BESS-001/revenue-risk").json()
    assert (
        commercial["disclaimer"]
        == "Counterfactual historical simulation, not actual commercial P&L."
    )
    assert commercial["data_provenance"]["provenance_type"] == "COUNTERFACTUAL"
    assert "power_derating_impact_eur" in commercial["attribution"]


def test_rack_anomaly_and_alert_pagination(api_bundle: tuple[TestClient, Any, Any]) -> None:
    client, _, _ = api_bundle
    rack = client.get("/api/v1/racks/PCS-01-RACK-01")
    assert rack.status_code == 200
    assert rack.json()["expected_temperature_c"] == 27.5
    assert "fault" not in rack.text.lower()
    assert client.get("/api/v1/racks/UNKNOWN").status_code == 404

    anomalies = client.get("/api/v1/racks/PCS-01-RACK-01/anomalies?limit=1&offset=0")
    assert anomalies.status_code == 200
    assert anomalies.json()["total"] == 1
    assert anomalies.json()["items"][0]["provenance"]["provenance_type"] == "DERIVED"
    invalid_range = client.get(
        "/api/v1/racks/PCS-01-RACK-01/anomalies",
        params={"start": "2026-01-02T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
    )
    assert invalid_range.status_code == 422
    naive = client.get(
        "/api/v1/racks/PCS-01-RACK-01/anomalies", params={"start": "2026-01-01T00:00:00"}
    )
    assert naive.status_code == 422

    alerts = client.get("/api/v1/alerts?status=OPEN&priority_level=HIGH&limit=1")
    assert alerts.status_code == 200
    alert = alerts.json()["items"][0]
    assert alert["priority_score"] <= 1
    assert alert["affected_power_mw"] >= 0 and alert["affected_energy_mwh"] >= 0
    assert alert["data_provenance"]["provenance_type"] == "DECISION_SUPPORT"
    assert alert["anomaly_score_semantics"] == "detector score; not a probability"


def test_market_actual_and_forecast_are_separate(api_bundle: tuple[TestClient, Any, Any]) -> None:
    client, _, _ = api_bundle
    body = client.get("/api/v1/market/latest").json()
    assert body["observed"]["day_ahead_price_eur_per_mwh"] == 80
    assert body["forecast"]["day_ahead_price_eur_per_mwh"] == 85
    assert body["observed_provenance"]["provenance_type"] == "REAL"
    assert body["forecast_provenance"]["provenance_type"] == "MODEL_PREDICTION"
    assert body["derived_provenance"]["provenance_type"] == "DERIVED"


def test_simulation_controls_validation_and_disable(
    api_bundle: tuple[TestClient, Any, Any],
) -> None:
    client, factory, settings = api_bundle
    valid = client.post(
        "/api/v1/simulation/fault",
        json={
            "fault_type": "THERMAL_DRIFT",
            "component_id": "PCS-01-RACK-01",
            "severity": 0.5,
        },
    )
    assert valid.status_code == 201
    assert valid.json()["fault_type"] == "THERMAL_DRIFT"
    assert (
        client.post(
            "/api/v1/simulation/fault",
            json={"fault_type": "EXEC_CODE", "component_id": "SITE", "severity": 0.5},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/simulation/fault",
            json={"fault_type": "RACK_OFFLINE", "component_id": "PCS-99", "severity": 1},
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/simulation/reset", json={"asset_id": "BESS-001", "seed": 7}
        ).status_code
        == 200
    )
    assert client.post("/api/v1/simulation/reset", json={"reset_database": True}).status_code == 422

    disabled = create_app(
        settings.model_copy(update={"enable_simulation_control_api": False}), factory
    )
    with TestClient(disabled) as disabled_client:
        response = disabled_client.post(
            "/api/v1/simulation/fault",
            json={"fault_type": "NORMAL", "component_id": "SITE", "severity": 0},
        )
    assert response.status_code == 403


def test_copilot_openapi_and_secret_safe_errors(
    api_bundle: tuple[TestClient, Any, Any], caplog: pytest.LogCaptureFixture
) -> None:
    client, _, _ = api_bundle
    unavailable = client.post(
        "/api/v1/copilot/query", json={"asset_id": "BESS-001", "query": "status?"}
    )
    assert unavailable.status_code == 503
    assert unavailable.json()["message"] == "AI Copilot is not configured."

    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths = schema.json()["paths"]
    for path in (
        "/api/v1/health",
        "/api/v1/assets",
        "/api/v1/assets/{asset_id}/status",
        "/api/v1/racks/{rack_id}",
        "/api/v1/alerts",
        "/api/v1/market/latest",
        "/api/v1/simulation/fault",
        "/api/v1/copilot/query",
    ):
        assert path in paths
    assert "AQ.Ab8RN6" not in schema.text
    assert "831fc90f" not in schema.text

    class BrokenAssetService(AssetService):
        def list_assets(self) -> tuple[Any, ...]:
            raise RuntimeError("GEMINI_API_KEY=super-secret")

    def broken() -> BrokenAssetService:
        raise RuntimeError("GEMINI_API_KEY=super-secret")

    client.app.dependency_overrides[get_asset_service] = broken
    caplog.set_level(logging.ERROR, logger="besspulse.api")
    response = client.get("/api/v1/assets")
    assert response.status_code == 500
    assert "super-secret" not in response.text
    assert "super-secret" not in caplog.text
    client.app.dependency_overrides.clear()


def test_read_endpoints_do_not_mutate_database(
    api_bundle: tuple[TestClient, sessionmaker[Session], Any],
) -> None:
    client, factory, _ = api_bundle
    with factory() as session:
        before = session.query(AlertModel).count()
    assert client.get("/api/v1/assets/BESS-001/status").status_code == 200
    assert client.get("/api/v1/alerts").status_code == 200
    with factory() as session:
        after = session.query(AlertModel).count()
    assert after == before


def test_request_session_closes_and_database_failure_is_controlled(tmp_path: Path) -> None:
    class TrackingSession(Session):
        close_count = 0

        def close(self) -> None:
            TrackingSession.close_count += 1
            super().close()

    engine = create_engine(
        f"sqlite:///{tmp_path / 'sessions.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, class_=TrackingSession, expire_on_commit=False)
    app = create_app(APISettings(database_url="sqlite:///unused.db"), factory)
    with TestClient(app, raise_server_exceptions=False) as client:
        before = TrackingSession.close_count
        assert client.get("/api/v1/assets").status_code == 200
        assert TrackingSession.close_count == before + 1

        def database_failure() -> AssetService:
            raise OperationalError("statement withheld", {}, Exception("database secret withheld"))

        app.dependency_overrides[get_asset_service] = database_failure
        response = client.get("/api/v1/assets")
        assert response.status_code == 503
        assert response.json()["error_code"] == "database_unavailable"
        assert "secret" not in response.text.lower()
    engine.dispose()


def test_missing_risk_artifact_returns_controlled_503(
    api_bundle: tuple[TestClient, sessionmaker[Session], Any],
) -> None:
    client, factory, _ = api_bundle
    with factory() as session:
        row = (
            session.query(ModelPredictionModel)
            .filter_by(entity_id="BESS-001", prediction_name="failure_probability_24h")
            .one()
        )
        session.delete(row)
        session.commit()
    response = client.get("/api/v1/assets/BESS-001/delivery-risk")
    assert response.status_code == 503
    assert response.json()["error_code"] == "data_unavailable"
