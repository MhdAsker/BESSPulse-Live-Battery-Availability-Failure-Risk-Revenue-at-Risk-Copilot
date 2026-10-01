from datetime import UTC, datetime

import pytest

from alerts.config import AlertConfig
from alerts.priority import calculate_priority
from alerts.schemas import AlertInput


def item(**updates: object) -> AlertInput:
    values = {
        "timestamp_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "available_discharge_power_mw": 16.0,
        "available_usable_energy_mwh": 28.0,
        "technical_availability": 0.8,
        "failure_probability_6h": 0.4,
        "failure_probability_12h": 0.5,
        "failure_probability_24h": 1.0,
        "anomaly_score": 0.5,
        "detector_count": 2,
        "revenue_at_risk_eur": 1000,
        "revenue_at_risk_fraction": 0.2,
    }
    values.update(updates)
    return AlertInput.model_validate(values)


def test_priority_bounded_components_and_24h_downweighting() -> None:
    score = calculate_priority(item())
    assert 0 <= score.selected <= 1
    assert score.risk_horizon_hours == 12
    assert score.risk == pytest.approx(0.5 * 0.85)
    assert score.capacity == pytest.approx(0.2)
    assert score.affected_power_mw == 4
    assert score.affected_energy_mwh == 4


@pytest.mark.parametrize(
    "field,low,high,component",
    [
        ("failure_probability_6h", 0.1, 0.8, "risk"),
        ("available_discharge_power_mw", 19.0, 10.0, "capacity"),
        ("available_usable_energy_mwh", 31.0, 20.0, "capacity"),
        ("revenue_at_risk_fraction", 0.1, 0.8, "commercial"),
    ],
)
def test_priority_components_are_monotonic(
    field: str, low: float, high: float, component: str
) -> None:
    first = calculate_priority(item(**{field: low}))
    second = calculate_priority(item(**{field: high}))
    assert getattr(second, component) >= getattr(first, component)


def test_confidence_scales_weighted_priority() -> None:
    low = calculate_priority(item(feature_completeness=0.2, detector_count=0))
    high = calculate_priority(item(feature_completeness=1, detector_count=3))
    assert high.weighted >= low.weighted


def test_technical_override_prevents_zero_commercial_suppression() -> None:
    score = calculate_priority(
        item(
            available_discharge_power_mw=0,
            available_usable_energy_mwh=0,
            revenue_at_risk_fraction=0,
            failure_probability_6h=0,
            failure_probability_12h=0,
            anomaly_score=0,
        )
    )
    assert score.selected >= AlertConfig().technical_override_floor
    assert score.level == "HIGH"


def test_weight_and_hysteresis_validation() -> None:
    with pytest.raises(ValueError, match="weights"):
        AlertConfig(risk_weight=1)
    with pytest.raises(ValueError, match="close threshold"):
        AlertConfig(close_score_threshold=0.4)


def test_ground_truth_fields_are_schema_rejected() -> None:
    with pytest.raises(ValueError):
        item(fault_type="injected", fault_severity=1)
