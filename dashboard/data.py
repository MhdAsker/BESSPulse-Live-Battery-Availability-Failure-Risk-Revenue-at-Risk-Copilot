"""Explicit local-artifact demo adapter for data not exposed as API history.

This adapter performs display-oriented selection/aggregation only. It never trains,
scores, recalculates availability, optimizes dispatch, or prioritizes alerts.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class LocalArtifactData:
    label = "DEMO / LOCAL DATA MODE"

    def availability_history(self) -> pd.DataFrame:
        return self._parquet("reports/availability/availability_snapshots.parquet")

    def latest_availability(self) -> dict[str, Any]:
        frame = self.availability_history()
        return {} if frame.empty else _record(frame.sort_values("timestamp_utc").iloc[-1])

    def rack_history(self, rack_id: str | None = None) -> pd.DataFrame:
        frame = self._parquet("reports/anomaly/causal_feature_cache_2880_seed42.parquet")
        if rack_id:
            frame = frame.loc[frame["rack_id"] == rack_id]
        return frame

    def latest_racks(self) -> list[dict[str, Any]]:
        frame = self.rack_history()
        if frame.empty:
            return []
        latest = frame.sort_values("timestamp_utc").groupby("rack_id", as_index=False).tail(1)
        events = self._parquet("reports/anomaly/intervals_vote_ensemble.parquet")
        if not events.empty and "timestamp_utc" in events:
            scores = (
                events.sort_values("timestamp_utc").groupby("component_id", as_index=False).tail(1)
            )
            score_values: list[Any] = (
                scores["anomaly_score"].tolist() if "anomaly_score" in scores else [0] * len(scores)
            )
            mapping = dict(zip(scores["component_id"].tolist(), score_values, strict=False))
            latest["anomaly_score"] = latest["rack_id"].map(mapping).fillna(0)
        latest["peer_temperature_deviation_c"] = latest.get("temperature_peer_deviation")
        return [_record(row) for _, row in latest.iterrows()]

    def delivery_predictions(self, horizon: int) -> pd.DataFrame:
        return self._parquet(f"artifacts/models/delivery_risk/{horizon}h/test_predictions.parquet")

    def latest_risk(self) -> dict[str, Any]:
        result: dict[str, Any] = {"calibration": []}
        timestamps = []
        for horizon in (6, 12, 24):
            frame = self.delivery_predictions(horizon)
            if frame.empty:
                continue
            row = frame.sort_values("timestamp_utc").iloc[-1]
            result[f"failure_probability_{horizon}h"] = float(
                row[f"failure_probability_{horizon}h"]
            )
            result[f"model_version_{horizon}h"] = str(row[f"model_version_{horizon}h"])
            result["feature_set_version"] = str(row["feature_set_version"])
            timestamps.append(row["timestamp_utc"])
            result["calibration"].append(
                {"horizon_hours": horizon, "status": "limited" if horizon == 24 else "validated"}
            )
        if timestamps:
            result["timestamp_utc"] = max(timestamps).isoformat()
        return result

    def alerts(self) -> list[dict[str, Any]]:
        frame = self._csv("reports/alerts/alert_history.csv")
        if frame.empty:
            return []
        aliases = {
            "component_id": "component",
            "opened_at_utc": "opened_at",
            "updated_at_utc": "updated_at",
        }
        frame = frame.rename(columns=aliases)
        return [_record(row) for _, row in frame.iterrows()]

    def commercial_summary(self, mode: str = "historical") -> dict[str, Any]:
        return self._json(f"reports/commercial/{mode}_benchmark_summary.json")

    def commercial_dispatch(self, mode: str = "historical") -> pd.DataFrame:
        return self._csv(f"reports/commercial/{mode}_dispatch.csv")

    def market_history(self) -> pd.DataFrame:
        frame = self.commercial_dispatch("historical")
        if "timestamp_utc" in frame:
            frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        return frame

    def price_forecast(self) -> pd.DataFrame:
        return self._parquet("reports/price_forecast/next_24h_forecast.parquet")

    def model_metadata(self, name: str) -> dict[str, Any]:
        paths = {
            "expected_power": "artifacts/models/expected_power/metadata.json",
            "expected_temperature": "artifacts/models/expected_temperature/metadata.json",
            "price": "artifacts/models/price/price_forecast_v1/metadata.json",
        }
        return self._json(paths[name])

    def delivery_metrics(self) -> dict[str, Any]:
        return self._json("reports/delivery_risk/test_metrics.json")

    def anomaly_metadata(self) -> dict[str, Any]:
        return self._json("reports/anomaly/experiment_metadata.json")

    def _parquet(self, relative: str) -> pd.DataFrame:
        path = ROOT / relative
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _csv(self, relative: str) -> pd.DataFrame:
        path = ROOT / relative
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def _json(self, relative: str) -> dict[str, Any]:
        path = ROOT / relative
        if not path.exists():
            return {}
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}


def _record(row: Any) -> dict[str, Any]:
    raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        if not isinstance(value, list | tuple | dict):
            try:
                if bool(pd.isna(value)):
                    value = None
            except (TypeError, ValueError):
                pass
        result[str(key)] = value
    return result
