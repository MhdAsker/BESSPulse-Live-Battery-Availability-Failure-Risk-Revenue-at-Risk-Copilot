"""Observable-evidence alert family selection without fault labels."""

from alerts.schemas import AlertInput, AlertType


def select_alert_type(item: AlertInput) -> AlertType | None:
    if item.unavailable_pcs > 0:
        return AlertType.PCS_UNAVAILABLE
    if item.unavailable_racks > 0:
        return AlertType.RACK_UNAVAILABLE
    if item.available_discharge_power_mw < 19.999:
        return AlertType.POWER_DERATING
    if item.available_usable_energy_mwh < 31.999:
        return AlertType.ENERGY_CAPACITY_REDUCTION
    if max(item.failure_probability_6h or 0, item.failure_probability_12h or 0) > 0:
        return AlertType.DELIVERY_RISK
    if (item.anomaly_score or 0) > 0:
        return AlertType.ANOMALY
    return None
