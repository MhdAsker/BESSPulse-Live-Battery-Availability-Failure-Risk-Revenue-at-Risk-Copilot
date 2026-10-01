from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from alerts.config import AlertConfig
from alerts.schemas import AlertInput, AlertStatus, AlertType
from alerts.service import AlertEngine, evaluate_alert, replay_alerts
from alerts.storage import persist_alert
from besspulse.database import AlertModel, Base


def row(
    index: int, *, power: float = 10, energy: float = 20, risk: float = 0.7, anomaly: float = 0.6
) -> AlertInput:
    return AlertInput(
        timestamp_utc=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index),
        component_id="PCS-02",
        component_type="PCS",
        available_discharge_power_mw=power,
        available_usable_energy_mwh=energy,
        technical_availability=power / 20,
        unavailable_racks=1 if power < 20 else 0,
        failure_probability_6h=risk,
        failure_probability_24h=1,
        anomaly_score=anomaly,
        detector_count=3,
        revenue_at_risk_eur=1000 if power < 20 else 0,
        revenue_at_risk_fraction=0.2 if power < 20 else 0,
        market_mode="historical",
        price_source="REAL",
        benchmark_disclaimer="Counterfactual historical simulation, not actual commercial P&L.",
        source_versions={"delivery_risk": "6h_v1", "commercial": "commercial_dispatch_v1"},
    )


def test_alert_generation_evidence_and_input_separation() -> None:
    alert = evaluate_alert(row(0), AlertConfig(open_persistence_intervals=1))
    assert alert is not None
    assert alert.alert_type == AlertType.RACK_UNAVAILABLE
    assert alert.affected_power_mw == 10
    assert alert.affected_energy_mwh == 12
    assert "anomaly score (not a probability)" in alert.supporting_signals
    assert alert.failure_probability != alert.anomaly_score
    assert alert.data_provenance == "DERIVED DECISION-SUPPORT OUTPUT"
    assert alert.benchmark_disclaimer is not None


def test_deduplication_open_update_hysteresis_and_resolution() -> None:
    config = AlertConfig(open_persistence_intervals=2, recovery_persistence_intervals=2)
    engine = AlertEngine(config)
    assert engine.process(row(0)) is None
    opened = engine.process(row(1))
    assert opened is not None and opened.status == AlertStatus.OPEN
    updated = engine.process(row(2))
    assert updated is not None and updated.alert_key == opened.alert_key
    assert updated.opened_at_utc == opened.opened_at_utc
    assert engine.process(row(3, power=20, energy=32, risk=0, anomaly=0)) is None
    resolved = engine.process(row(4, power=20, energy=32, risk=0, anomaly=0))
    assert resolved is not None and resolved.status == AlertStatus.RESOLVED


def test_replay_is_deterministic() -> None:
    inputs = [row(i) for i in range(4)] + [
        row(i, power=20, energy=32, risk=0, anomaly=0) for i in range(4, 8)
    ]
    first = replay_alerts(inputs)
    second = replay_alerts(inputs)
    assert first == second


def test_storage_updates_without_duplicate_and_preserves_versions() -> None:
    alert = evaluate_alert(row(0), AlertConfig(open_persistence_intervals=1))
    assert alert is not None
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert persist_alert(session, alert)
        assert not persist_alert(session, alert)
        stored = session.scalars(select(AlertModel)).all()
    assert len(stored) == 1
    assert "delivery_risk" in stored[0].source_versions_json
    assert stored[0].data_provenance == "DERIVED DECISION-SUPPORT OUTPUT"


def test_cooldown_blocks_immediate_reopen() -> None:
    config = AlertConfig(
        open_persistence_intervals=1,
        recovery_persistence_intervals=1,
        cooldown_minutes=60,
    )
    engine = AlertEngine(config)
    assert engine.process(row(0)) is not None
    assert (
        engine.process(row(1, power=20, energy=32, risk=0, anomaly=0)).status
        == AlertStatus.RESOLVED
    )  # type: ignore[union-attr]
    assert engine.process(row(2)) is None
