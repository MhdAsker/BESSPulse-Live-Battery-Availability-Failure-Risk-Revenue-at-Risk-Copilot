"""Transparent interval and aggregate commercial accounting."""

import pandas as pd

from optimization.dispatch import DispatchResult


def combine_dispatch_revenue(healthy: DispatchResult, current: DispatchResult) -> pd.DataFrame:
    left = healthy.intervals.add_prefix("healthy_").rename(
        columns={
            "healthy_timestamp_utc": "timestamp_utc",
            "healthy_price_eur_per_mwh": "price_eur_per_mwh",
        }
    )
    right = current.intervals.add_prefix("current_").rename(
        columns={
            "current_timestamp_utc": "timestamp_utc",
            "current_price_eur_per_mwh": "price_eur_per_mwh",
        }
    )
    keep = [column for column in right if column not in {"price_eur_per_mwh", "timestamp_utc"}]
    result = left.merge(right[["timestamp_utc", *keep]], on="timestamp_utc", validate="one_to_one")
    result["interval_revenue_at_risk_eur"] = (
        result["healthy_net_cashflow_eur"] - result["current_net_cashflow_eur"]
    )
    result["cumulative_healthy_revenue_eur"] = result["healthy_net_cashflow_eur"].cumsum()
    result["cumulative_current_revenue_eur"] = result["current_net_cashflow_eur"].cumsum()
    result["cumulative_revenue_at_risk_eur"] = result["interval_revenue_at_risk_eur"].cumsum()
    result["data_provenance"] = "COUNTERFACTUAL"
    return result
