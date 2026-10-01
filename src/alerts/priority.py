"""Bounded, monotonic, explicit alert-priority components."""

from dataclasses import dataclass

import numpy as np

from alerts.config import AlertConfig
from alerts.schemas import AlertInput


@dataclass(frozen=True)
class PriorityComponents:
    risk: float
    risk_horizon_hours: int | None
    confidence: float
    capacity: float
    commercial: float
    anomaly: float
    technical: float
    affected_power_mw: float
    affected_energy_mwh: float
    product: float
    weighted: float
    selected: float
    level: str


def _clip(value: float) -> float:
    return float(np.clip(value, 0, 1))


def priority_level(score: float, config: AlertConfig) -> str:
    if score >= config.critical_threshold:
        return "CRITICAL"
    if score >= config.high_threshold:
        return "HIGH"
    if score >= config.medium_threshold:
        return "MEDIUM"
    if score >= config.low_threshold:
        return "LOW"
    return "INFO"


def calculate_priority(item: AlertInput, config: AlertConfig | None = None) -> PriorityComponents:
    cfg = config or AlertConfig()
    candidates = [
        (6, item.failure_probability_6h, cfg.risk_reliability_6h),
        (12, item.failure_probability_12h, cfg.risk_reliability_12h),
        (24, item.failure_probability_24h, cfg.risk_reliability_24h),
    ]
    available = [(h, float(p), r) for h, p, r in candidates if p is not None]
    if available:
        horizon, probability, reliability = max(available, key=lambda value: value[1] * value[2])
        risk = _clip(probability * reliability)
        confidence = _clip(
            item.feature_completeness * reliability * min(1, 0.8 + 0.05 * item.detector_count)
        )
    else:
        horizon, probability, risk = None, 0.0, 0.0
        confidence = _clip(item.feature_completeness * min(1, 0.8 + 0.05 * item.detector_count))
    affected_power = max(0.0, cfg.reference_power_mw - item.available_discharge_power_mw)
    affected_energy = max(0.0, cfg.reference_usable_energy_mwh - item.available_usable_energy_mwh)
    power_fraction = affected_power / cfg.reference_power_mw
    energy_fraction = affected_energy / cfg.reference_usable_energy_mwh
    capacity = _clip(max(power_fraction, energy_fraction))
    technical = _clip(max(capacity, 1 - item.technical_availability))
    commercial = _clip(max(0.0, item.revenue_at_risk_fraction or 0.0))
    anomaly = _clip(item.anomaly_score or 0.0)
    product = _clip(risk * capacity * commercial * confidence)
    weighted = _clip(
        (
            cfg.risk_weight * risk
            + cfg.capacity_weight * capacity
            + cfg.commercial_weight * commercial
            + cfg.anomaly_weight * anomaly
        )
        * confidence
    )
    if technical >= cfg.technical_override_threshold:
        weighted = max(weighted, cfg.technical_override_floor)
        product = max(product, cfg.technical_override_floor)
    selected = product if cfg.default_formula == "priority_v1_product" else weighted
    return PriorityComponents(
        risk,
        horizon,
        confidence,
        capacity,
        commercial,
        anomaly,
        technical,
        affected_power,
        affected_energy,
        product,
        weighted,
        selected,
        priority_level(selected, cfg),
    )
