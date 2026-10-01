"""Compatibility exports for the implemented top-level feature package."""

from features import (
    FEATURE_REGISTRY,
    FEATURE_SET_VERSION,
    add_peer_features,
    align_market_to_battery,
    build_market_features,
    build_rack_features,
    build_site_features,
    summarize_site_kpis,
)

__all__ = [
    "FEATURE_REGISTRY",
    "FEATURE_SET_VERSION",
    "add_peer_features",
    "align_market_to_battery",
    "build_market_features",
    "build_rack_features",
    "build_site_features",
    "summarize_site_kpis",
]
