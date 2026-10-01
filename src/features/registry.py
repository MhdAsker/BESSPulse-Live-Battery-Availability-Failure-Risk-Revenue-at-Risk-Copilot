"""Lightweight traceability metadata for implemented feature families."""

from dataclasses import dataclass

FEATURE_SET_VERSION = "v1"


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    entity_level: str
    unit: str
    provenance: str
    causal: bool
    source_fields: tuple[str, ...]
    description: str


def _feature(
    name: str,
    level: str,
    unit: str,
    sources: tuple[str, ...],
    description: str,
) -> FeatureMetadata:
    return FeatureMetadata(name, level, unit, "DERIVED", True, sources, description)


FEATURE_REGISTRY: dict[str, FeatureMetadata] = {
    item.name: item
    for item in (
        _feature(
            "delivery_ratio",
            "site",
            "fraction",
            ("actual_power_mw", "requested_power_mw"),
            "Absolute delivered/requested power on active intervals.",
        ),
        _feature(
            "power_residual_mw",
            "site",
            "MW",
            ("actual_power_mw", "requested_power_mw"),
            "Signed actual minus requested power.",
        ),
        _feature(
            "normalized_power_residual",
            "site",
            "fraction",
            ("power_residual_mw", "rated_power_mw"),
            "Residual divided by site nameplate power.",
        ),
        _feature(
            "soc_change_rate",
            "site/rack",
            "fraction/hour",
            ("soc", "timestamp_utc"),
            "Backward SOC change divided by elapsed time.",
        ),
        _feature(
            "soc_volatility", "site/rack", "fraction", ("soc",), "Trailing SOC standard deviation."
        ),
        _feature(
            "thermal_exposure_above_threshold",
            "rack",
            "fraction",
            ("temperature_mean_c",),
            "Trailing fraction of observations above configured temperature.",
        ),
        _feature(
            "rte_trend",
            "rack",
            "fraction/hour",
            ("rte",),
            "Exact time-lag RTE slope over configured window.",
        ),
        _feature(
            "availability_rate",
            "rack",
            "fraction",
            ("availability",),
            "Trailing observed availability fraction.",
        ),
        _feature(
            "alarm_frequency",
            "site/rack",
            "fraction",
            ("alarm_indicator",),
            "Trailing fraction of intervals with an observable alarm.",
        ),
        _feature(
            "temperature_peer_robust_zscore",
            "rack",
            "dimensionless",
            ("temperature_mean_c",),
            "Leave-one-out deviation divided by 1.4826 peer MAD.",
        ),
        _feature(
            "price_lag_24h",
            "market",
            "EUR/MWh",
            ("price_eur_per_mwh",),
            "Strict exact timestamp lag by 24 hours.",
        ),
        _feature(
            "price_lag_168h",
            "market",
            "EUR/MWh",
            ("price_eur_per_mwh",),
            "Strict exact timestamp lag by 168 hours.",
        ),
        _feature(
            "rolling_price_mean",
            "market",
            "EUR/MWh",
            ("price_eur_per_mwh",),
            "Causal trailing price mean.",
        ),
        _feature(
            "price_volatility",
            "market",
            "EUR/MWh",
            ("price_eur_per_mwh",),
            "Causal trailing price standard deviation.",
        ),
        _feature(
            "renewable_generation_mw",
            "market",
            "MW",
            ("wind_generation_mw", "solar_generation_mw"),
            "Wind plus solar only when all required components exist.",
        ),
        _feature(
            "residual_load_mw",
            "market",
            "MW",
            ("load_mw", "renewable_generation_mw"),
            "Actual load minus actual renewable generation.",
        ),
        _feature(
            "renewable_share",
            "market",
            "fraction",
            ("renewable_generation_mw", "load_mw"),
            "Renewable generation divided by nonzero load.",
        ),
        _feature(
            "market_data_age_minutes",
            "combined",
            "minutes",
            ("timestamp_utc", "market_timestamp_utc"),
            "Backward-aligned market observation age.",
        ),
    )
}
