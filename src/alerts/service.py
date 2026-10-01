"""High-level alert evaluation, deterministic deduplication, and lifecycle."""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from alerts.config import AlertConfig
from alerts.evidence import evidence_for
from alerts.priority import calculate_priority
from alerts.rules import select_alert_type
from alerts.schemas import AlertInput, AlertRecord, AlertStatus


def evaluate_alert(
    item: AlertInput,
    config: AlertConfig | None = None,
    *,
    minimum_score: float | None = None,
) -> AlertRecord | None:
    cfg = config or AlertConfig()
    alert_type = select_alert_type(item)
    if alert_type is None:
        return None
    priority = calculate_priority(item, cfg)
    deterministic_capability_alert = alert_type.value in {
        "POWER_DERATING",
        "ENERGY_CAPACITY_REDUCTION",
        "RACK_UNAVAILABLE",
        "PCS_UNAVAILABLE",
    }
    if (
        not deterministic_capability_alert
        and priority.selected
        < (cfg.open_score_threshold if minimum_score is None else minimum_score)
        and priority.technical < cfg.technical_override_threshold
    ):
        return None
    stamp = item.timestamp_utc
    bucket_minutes = max(1, cfg.merge_window_minutes)
    bucket = int(stamp.timestamp() // (bucket_minutes * 60))
    raw_key = f"{item.asset_id}|{item.component_id}|{alert_type.value}|{bucket}"
    key = hashlib.sha256(raw_key.encode()).hexdigest()[:24]
    return AlertRecord(
        alert_key=key,
        asset_id=item.asset_id,
        component_id=item.component_id,
        component_type=item.component_type,
        alert_type=alert_type,
        status=AlertStatus.OPEN,
        opened_at_utc=item.timestamp_utc,
        updated_at_utc=item.timestamp_utc,
        priority_score=priority.selected,
        priority_product=priority.product,
        priority_weighted=priority.weighted,
        priority_level=priority.level,
        failure_probability=priority.risk,
        risk_horizon_hours=priority.risk_horizon_hours,
        model_confidence=priority.confidence,
        technical_severity=priority.technical,
        commercial_severity=priority.commercial,
        capacity_severity=priority.capacity,
        anomaly_component=priority.anomaly,
        affected_power_mw=priority.affected_power_mw,
        affected_energy_mwh=priority.affected_energy_mwh,
        revenue_at_risk_eur=item.revenue_at_risk_eur,
        anomaly_score=item.anomaly_score,
        limiting_factor=item.limiting_factor,
        supporting_signals=evidence_for(item),
        source_versions={**item.source_versions, "alert_engine": cfg.engine_version},
        market_mode=item.market_mode,
        price_source=item.price_source,
        benchmark_disclaimer=item.benchmark_disclaimer,
    )


@dataclass
class AlertEngine:
    config: AlertConfig = field(default_factory=AlertConfig)
    active: dict[str, AlertRecord] = field(default_factory=dict)
    pending: dict[str, int] = field(default_factory=dict)
    recovery: dict[str, int] = field(default_factory=dict)
    last_resolved: dict[str, datetime] = field(default_factory=dict)

    def process(self, item: AlertInput) -> AlertRecord | None:
        candidate = evaluate_alert(item, self.config)
        component_keys = [
            key for key, value in self.active.items() if value.component_id == item.component_id
        ]
        if candidate is None and component_keys:
            candidate = evaluate_alert(
                item, self.config, minimum_score=self.config.close_score_threshold
            )
        if candidate is None:
            for key in component_keys:
                self.recovery[key] = self.recovery.get(key, 0) + 1
                if self.recovery[key] >= self.config.recovery_persistence_intervals:
                    previous = self.active.pop(key)
                    self.recovery.pop(key, None)
                    identity = f"{previous.component_id}|{previous.alert_type.value}"
                    self.last_resolved[identity] = item.timestamp_utc
                    return previous.model_copy(
                        update={
                            "status": AlertStatus.RESOLVED,
                            "updated_at_utc": item.timestamp_utc,
                            "resolved_at_utc": item.timestamp_utc,
                        }
                    )
            return None
        self.recovery[candidate.alert_key] = 0
        for key in component_keys:
            previous = self.active[key]
            if previous.alert_type == candidate.alert_type:
                continue
            self.recovery[key] = self.recovery.get(key, 0) + 1
            if self.recovery[key] >= self.config.recovery_persistence_intervals:
                self.active.pop(key)
                self.recovery.pop(key, None)
                identity = f"{previous.component_id}|{previous.alert_type.value}"
                self.last_resolved[identity] = item.timestamp_utc
                return previous.model_copy(
                    update={
                        "status": AlertStatus.RESOLVED,
                        "updated_at_utc": item.timestamp_utc,
                        "resolved_at_utc": item.timestamp_utc,
                    }
                )
        active_key = next(
            (
                key
                for key, value in self.active.items()
                if value.component_id == candidate.component_id
                and value.alert_type == candidate.alert_type
            ),
            None,
        )
        if active_key is not None:
            previous = self.active[active_key]
            updated = candidate.model_copy(
                update={"alert_key": active_key, "opened_at_utc": previous.opened_at_utc}
            )
            self.active[active_key] = updated
            return updated
        identity = f"{candidate.component_id}|{candidate.alert_type.value}"
        if identity in self.last_resolved:
            cooldown = timedelta(minutes=self.config.cooldown_minutes)
            if item.timestamp_utc - self.last_resolved[identity] < cooldown:
                return None
        count = self.pending.get(candidate.alert_key, 0) + 1
        self.pending[candidate.alert_key] = count
        if count < self.config.open_persistence_intervals:
            return None
        self.pending.pop(candidate.alert_key, None)
        self.active[candidate.alert_key] = candidate
        return candidate


def replay_alerts(inputs: list[AlertInput], config: AlertConfig | None = None) -> list[AlertRecord]:
    engine = AlertEngine(config or AlertConfig())
    output: list[AlertRecord] = []
    for item in sorted(inputs, key=lambda value: value.timestamp_utc):
        result = engine.process(item)
        if result is not None:
            output.append(result)
    return output
