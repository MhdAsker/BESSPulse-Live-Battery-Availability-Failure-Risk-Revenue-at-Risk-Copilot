"""Traceable evidence construction."""

from alerts.schemas import AlertInput


def evidence_for(item: AlertInput) -> tuple[str, ...]:
    evidence = list(item.supporting_signals)
    if item.available_discharge_power_mw < 20:
        evidence.append("deterministic available-power reduction")
    if item.available_usable_energy_mwh < 32:
        evidence.append("deterministic available-energy reduction")
    if item.failure_probability_6h is not None or item.failure_probability_12h is not None:
        evidence.append("delivery-risk model prediction")
    if item.failure_probability_24h is not None:
        evidence.append("24h risk contextual only; poor out-of-time calibration")
    if item.anomaly_score is not None:
        evidence.append("anomaly score (not a probability)")
    if item.revenue_at_risk_eur != 0:
        evidence.append("Prompt 9 counterfactual RevenueAtRisk")
    return tuple(dict.fromkeys(evidence))
