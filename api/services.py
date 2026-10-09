"""API-facing query and control services.

These adapters compose existing persisted domain outputs. They do not reproduce
battery, ML, availability, optimization, or alert-priority calculations.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from alerts.schemas import AlertStatus, AlertType
from api.config import APISettings
from api.errors import DataUnavailableError, FeatureUnavailableError, NotFoundError
from api.schemas.alerts import AlertPage, AlertResponse
from api.schemas.anomalies import AnomalyEventResponse, AnomalyPage
from api.schemas.assets import AssetStatus, AssetSummary
from api.schemas.availability import AvailabilityResponse
from api.schemas.commercial import CommercialAttributionResponse, RevenueRiskResponse
from api.schemas.common import Freshness, Provenance, ProvenanceType
from api.schemas.copilot import CopilotQueryRequest, CopilotQueryResponse
from api.schemas.market import (
    DerivedMarketFields,
    ForecastMarketFields,
    MarketLatestResponse,
    ObservedMarketFields,
)
from api.schemas.racks import AnomalySummary, RackResponse
from api.schemas.risk import CalibrationMetadata, DeliveryRiskResponse
from api.schemas.simulation import (
    FaultInjectionRequest,
    FaultInjectionResponse,
    SimulationResetRequest,
    SimulationResetResponse,
    default_fault_window,
)
from besspulse.config import SimulationConfig
from besspulse.database import (
    AlertModel,
    AnomalyEventModel,
    Asset,
    AvailabilitySnapshotModel,
    BatteryTelemetryModel,
    MarketDataModel,
    ModelPredictionModel,
    PCSModel,
    PricePredictionModel,
    RackModel,
    RackTelemetryModel,
    RevenueAtRiskModel,
)
from commercial.schemas import DISCLAIMER


def utc(value: datetime) -> datetime:
    """Treat SQLite's timezone-stripped values as UTC at the persistence boundary."""

    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def provenance(
    kind: ProvenanceType,
    source: str,
    version: str,
    generated_at: datetime,
    *,
    model_version: str | None = None,
    feature_set_version: str | None = None,
) -> Provenance:
    return Provenance(
        provenance_type=kind,
        source=source,
        source_version=version,
        generated_at_utc=utc(generated_at),
        model_version=model_version,
        feature_set_version=feature_set_version,
    )


def freshness(value: datetime, settings: APISettings) -> Freshness:
    as_of = utc(value)
    age = max(0.0, (datetime.now(UTC) - as_of).total_seconds())
    return Freshness(
        as_of_utc=as_of,
        age_seconds=age,
        is_stale=age > settings.api_stale_after_seconds,
    )


class AssetService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings
        self.config = SimulationConfig()

    def ensure_exists(self, asset_id: str) -> None:
        if asset_id == "BESS-001":
            return
        if self.session.scalar(select(Asset.id).where(Asset.asset_id == asset_id)) is None:
            raise NotFoundError("asset", asset_id)

    def list_assets(self) -> tuple[AssetSummary, ...]:
        assets = self.session.scalars(select(Asset).order_by(Asset.asset_id)).all()
        if not assets:
            cfg = self.config.battery
            return (
                AssetSummary(
                    asset_id="BESS-001",
                    name="BESSPulse reference asset",
                    rated_power_mw=cfg.site_rated_power_mw,
                    rated_energy_mwh=cfg.site_rated_energy_mwh,
                    pcs_count=cfg.pcs_count,
                    rack_count=cfg.pcs_count * cfg.racks_per_pcs,
                    status="configured",
                    data_provenance=provenance(
                        ProvenanceType.SIMULATED,
                        "BESSPulse configuration",
                        "simulation-config-v1",
                        datetime.now(UTC),
                    ),
                ),
            )
        result = []
        for asset in assets:
            pcs_count = (
                self.session.scalar(
                    select(func.count()).select_from(PCSModel).where(PCSModel.asset_pk == asset.id)
                )
                or 0
            )
            rack_count = (
                self.session.scalar(
                    select(func.count())
                    .select_from(RackModel)
                    .join(PCSModel, RackModel.pcs_pk == PCSModel.id)
                    .where(PCSModel.asset_pk == asset.id)
                )
                or 0
            )
            result.append(
                AssetSummary(
                    asset_id=asset.asset_id,
                    name=asset.asset_id,
                    rated_power_mw=asset.rated_power_mw,
                    rated_energy_mwh=asset.rated_energy_mwh,
                    pcs_count=pcs_count,
                    rack_count=rack_count,
                    status="configured",
                    data_provenance=provenance(
                        ProvenanceType.SIMULATED,
                        "assets table",
                        "schema-v1",
                        datetime.now(UTC),
                    ),
                )
            )
        return tuple(result)

    def status(self, asset_id: str) -> AssetStatus:
        self.ensure_exists(asset_id)
        telemetry = self.session.scalar(
            select(BatteryTelemetryModel)
            .where(BatteryTelemetryModel.asset_id == asset_id)
            .order_by(desc(BatteryTelemetryModel.timestamp_utc))
            .limit(1)
        )
        availability = _latest_availability(self.session, asset_id)
        risks = _latest_risk_rows(self.session, asset_id)
        commercial = _latest_commercial(self.session, asset_id)
        market_time = self.session.scalar(select(func.max(MarketDataModel.timestamp_utc)))
        alerts = self.session.scalars(
            select(AlertModel).where(AlertModel.asset_id == asset_id, AlertModel.status == "OPEN")
        ).all()
        timestamps = [
            utc(value)
            for value in (
                telemetry.timestamp_utc if telemetry else None,
                availability.timestamp_utc if availability else None,
                *(row.timestamp_utc for row in risks.values()),
                commercial.created_at if commercial else None,
                market_time,
            )
            if value is not None
        ]
        if not timestamps:
            raise DataUnavailableError(f"No operational data is available for asset '{asset_id}'.")
        cfg = self.config.battery
        highest = max(alerts, key=lambda item: item.priority_score, default=None)
        freshness_map: dict[str, Freshness] = {}
        for name, value in (
            ("telemetry", telemetry.timestamp_utc if telemetry else None),
            ("availability", availability.timestamp_utc if availability else None),
            ("risk", max((r.timestamp_utc for r in risks.values()), default=None)),
            ("market", market_time),
            ("commercial", commercial.created_at if commercial else None),
        ):
            if value is not None:
                freshness_map[name] = freshness(value, self.settings)
        prov = []
        if telemetry:
            prov.append(
                provenance(
                    ProvenanceType.SIMULATED,
                    "battery telemetry",
                    "schema-v1",
                    telemetry.timestamp_utc,
                )
            )
        if availability:
            prov.append(
                provenance(
                    ProvenanceType.DERIVED,
                    "availability engine",
                    "availability-v1",
                    availability.timestamp_utc,
                )
            )
        if risks:
            row = next(iter(risks.values()))
            prov.append(
                provenance(
                    ProvenanceType.MODEL_PREDICTION,
                    "delivery-risk models",
                    "delivery-risk-v1",
                    row.timestamp_utc,
                    feature_set_version=row.feature_set_version,
                )
            )
        if commercial:
            prov.append(
                provenance(
                    ProvenanceType.COUNTERFACTUAL,
                    "commercial benchmark",
                    commercial.capability_version,
                    commercial.created_at,
                    model_version=commercial.model_or_price_version,
                )
            )
        return AssetStatus(
            asset_id=asset_id,
            timestamp_utc=max(timestamps),
            rated_power_mw=cfg.site_rated_power_mw,
            rated_energy_mwh=cfg.site_rated_energy_mwh,
            available_power_mw=availability.available_discharge_power_mw
            if availability
            else (telemetry.available_power_mw if telemetry else None),
            available_energy_mwh=availability.available_discharge_energy_mwh
            if availability
            else (telemetry.available_energy_mwh if telemetry else None),
            site_soc=telemetry.site_soc if telemetry else None,
            rte=telemetry.rte if telemetry else None,
            technical_availability=availability.technical_availability if availability else None,
            requested_power_availability=availability.requested_power_availability
            if availability
            else None,
            available_racks=availability.available_racks
            if availability
            else (telemetry.available_racks if telemetry else None),
            available_pcs=availability.available_pcs
            if availability
            else (telemetry.available_pcs if telemetry else None),
            active_alert_count=len(alerts),
            highest_priority_alert=highest.priority_level if highest else None,
            failure_probability_6h=risks[6].prediction_value if 6 in risks else None,
            failure_probability_12h=risks[12].prediction_value if 12 in risks else None,
            failure_probability_24h=risks[24].prediction_value if 24 in risks else None,
            revenue_at_risk_eur=commercial.revenue_at_risk_eur if commercial else None,
            telemetry_timestamp_utc=utc(telemetry.timestamp_utc) if telemetry else None,
            risk_prediction_timestamp_utc=max(
                (utc(r.timestamp_utc) for r in risks.values()), default=None
            ),
            availability_timestamp_utc=utc(availability.timestamp_utc) if availability else None,
            market_timestamp_utc=utc(market_time) if market_time else None,
            commercial_timestamp_utc=utc(commercial.created_at) if commercial else None,
            freshness=freshness_map,
            provenance=tuple(prov),
        )


class AvailabilityService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings

    def latest(self, asset_id: str) -> AvailabilityResponse:
        AssetService(self.session, self.settings).ensure_exists(asset_id)
        row = _latest_availability(self.session, asset_id)
        if row is None:
            raise DataUnavailableError(
                f"Availability output is unavailable for asset '{asset_id}'."
            )
        return AvailabilityResponse(
            asset_id=row.asset_id,
            timestamp_utc=utc(row.timestamp_utc),
            technical_availability=row.technical_availability,
            rack_technical_availability=row.rack_technical_availability,
            pcs_technical_availability=row.pcs_technical_availability,
            available_discharge_power_mw=row.available_discharge_power_mw,
            available_charge_power_mw=row.available_charge_power_mw,
            discharge_power_availability=row.discharge_power_availability,
            charge_power_availability=row.charge_power_availability,
            available_discharge_energy_mwh=row.available_discharge_energy_mwh,
            available_charge_energy_mwh=row.available_charge_energy_mwh,
            discharge_energy_availability=row.discharge_energy_availability,
            charge_energy_availability=row.charge_energy_availability,
            requested_power_availability=row.requested_power_availability,
            sustainable_request_duration_hours=row.sustainable_request_duration_hours,
            limiting_factor=row.limiting_factor,
            limiting_factors=tuple(json.loads(row.limiting_factors_json)),
            limiting_components=tuple(json.loads(row.limiting_component_ids_json)),
            available_racks=row.available_racks,
            available_pcs=row.available_pcs,
            freshness=freshness(row.timestamp_utc, self.settings),
            data_provenance=provenance(
                ProvenanceType.DERIVED, "availability engine", "availability-v1", row.timestamp_utc
            ),
        )


class RiskService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings

    def latest(self, asset_id: str) -> DeliveryRiskResponse:
        AssetService(self.session, self.settings).ensure_exists(asset_id)
        rows = _latest_risk_rows(self.session, asset_id)
        if set(rows) != {6, 12, 24}:
            raise DataUnavailableError(
                f"Complete delivery-risk output is unavailable for asset '{asset_id}'."
            )
        latest = max(utc(row.timestamp_utc) for row in rows.values())
        return DeliveryRiskResponse(
            asset_id=asset_id,
            timestamp_utc=latest,
            failure_probability_6h=rows[6].prediction_value,
            failure_probability_12h=rows[12].prediction_value,
            failure_probability_24h=rows[24].prediction_value,
            model_version_6h=rows[6].model_version,
            model_version_12h=rows[12].model_version,
            model_version_24h=rows[24].model_version,
            feature_set_version=rows[6].feature_set_version,
            model_confidence=None,
            calibration=(
                CalibrationMetadata(horizon_hours=6, status="validated"),
                CalibrationMetadata(horizon_hours=12, status="validated"),
                CalibrationMetadata(
                    horizon_hours=24,
                    status="limited",
                    limitation=(
                        "Poor out-of-time calibration; interpret the 24h probability cautiously."
                    ),
                ),
            ),
            freshness=freshness(latest, self.settings),
            data_provenance=provenance(
                ProvenanceType.MODEL_PREDICTION,
                "delivery-risk models",
                "delivery-risk-v1",
                latest,
                feature_set_version=rows[6].feature_set_version,
            ),
        )


class CommercialService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings

    def latest(self, asset_id: str) -> RevenueRiskResponse:
        AssetService(self.session, self.settings).ensure_exists(asset_id)
        row = _latest_commercial(self.session, asset_id)
        if row is None:
            raise DataUnavailableError(
                f"Persisted Revenue-at-Risk is unavailable for asset '{asset_id}'."
            )
        return RevenueRiskResponse(
            asset_id=row.asset_id,
            start_timestamp_utc=utc(row.start_timestamp_utc),
            end_timestamp_utc=utc(row.end_timestamp_utc),
            market_mode=row.market_mode,
            price_source=row.price_source,
            healthy_gross_revenue_eur=row.healthy_gross_revenue_eur,
            healthy_degradation_cost_eur=row.healthy_degradation_cost_eur,
            healthy_net_revenue_eur=row.healthy_net_revenue_eur,
            current_gross_revenue_eur=row.current_gross_revenue_eur,
            current_degradation_cost_eur=row.current_degradation_cost_eur,
            current_net_revenue_eur=row.current_net_revenue_eur,
            revenue_at_risk_eur=row.revenue_at_risk_eur,
            revenue_at_risk_fraction=row.revenue_at_risk_fraction,
            attribution=CommercialAttributionResponse(
                power_derating_impact_eur=row.power_derating_impact_eur,
                energy_capacity_impact_eur=row.energy_capacity_impact_eur,
                efficiency_impact_eur=row.efficiency_impact_eur,
                availability_impact_eur=row.availability_impact_eur,
                interaction_unattributed_eur=row.interaction_unattributed_eur,
            ),
            calculation_version=row.capability_version,
            freshness=freshness(row.created_at, self.settings),
            data_provenance=provenance(
                ProvenanceType.COUNTERFACTUAL,
                "commercial benchmark",
                row.capability_version,
                row.created_at,
                model_version=row.model_or_price_version,
            ),
            disclaimer=DISCLAIMER,
        )


def _latest_availability(session: Session, asset_id: str) -> AvailabilitySnapshotModel | None:
    return session.scalar(
        select(AvailabilitySnapshotModel)
        .where(AvailabilitySnapshotModel.asset_id == asset_id)
        .order_by(desc(AvailabilitySnapshotModel.timestamp_utc))
        .limit(1)
    )


def _latest_risk_rows(session: Session, asset_id: str) -> dict[int, ModelPredictionModel]:
    result: dict[int, ModelPredictionModel] = {}
    for horizon in (6, 12, 24):
        name = f"failure_probability_{horizon}h"
        row = session.scalar(
            select(ModelPredictionModel)
            .where(
                ModelPredictionModel.entity_id == asset_id,
                ModelPredictionModel.prediction_name == name,
            )
            .order_by(desc(ModelPredictionModel.timestamp_utc))
            .limit(1)
        )
        if row is not None:
            result[horizon] = row
    return result


def _latest_commercial(session: Session, asset_id: str) -> RevenueAtRiskModel | None:
    return session.scalar(
        select(RevenueAtRiskModel)
        .where(RevenueAtRiskModel.asset_id == asset_id)
        .order_by(desc(RevenueAtRiskModel.created_at))
        .limit(1)
    )


class RackService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings

    def latest(self, rack_id: str) -> RackResponse:
        row = self.session.scalar(
            select(RackTelemetryModel)
            .where(RackTelemetryModel.rack_id == rack_id)
            .order_by(desc(RackTelemetryModel.timestamp_utc))
            .limit(1)
        )
        if row is None:
            raise NotFoundError("rack", rack_id)
        expected = self.session.scalar(
            select(ModelPredictionModel)
            .where(
                ModelPredictionModel.entity_id == rack_id,
                ModelPredictionModel.prediction_name == "expected_temperature_c",
                ModelPredictionModel.timestamp_utc <= row.timestamp_utc,
            )
            .order_by(desc(ModelPredictionModel.timestamp_utc))
            .limit(1)
        )
        anomaly = self.session.scalar(
            select(AnomalyEventModel)
            .where(
                AnomalyEventModel.component_id == rack_id,
                AnomalyEventModel.start_timestamp <= row.timestamp_utc,
                AnomalyEventModel.end_timestamp >= row.timestamp_utc,
            )
            .order_by(desc(AnomalyEventModel.peak_score))
            .limit(1)
        )
        peer_temp, peer_voltage = self.session.execute(
            select(
                func.avg(RackTelemetryModel.temperature_mean_c),
                func.avg(RackTelemetryModel.voltage_v),
            ).where(
                RackTelemetryModel.pcs_id == row.pcs_id,
                RackTelemetryModel.timestamp_utc == row.timestamp_utc,
                RackTelemetryModel.rack_id != rack_id,
            )
        ).one()
        return RackResponse(
            rack_id=row.rack_id,
            pcs_id=row.pcs_id,
            timestamp_utc=utc(row.timestamp_utc),
            soc=row.soc,
            soh_proxy=row.soh_proxy,
            voltage_v=row.voltage_v,
            current_a=row.current_a,
            temperature_mean_c=row.temperature_mean_c,
            temperature_min_c=row.temperature_min_c,
            temperature_max_c=row.temperature_max_c,
            temperature_spread_c=row.temperature_spread_c,
            voltage_spread_v=row.voltage_spread_v,
            requested_power_mw=row.requested_power_mw,
            actual_power_mw=row.actual_power_mw,
            rte=row.rte,
            availability=row.availability,
            alarm_code=row.alarm_code,
            operating_state=row.operating_state,
            peer_temperature_mean_c=(
                float(cast(float, peer_temp)) if peer_temp is not None else None
            ),
            peer_voltage_mean_v=(
                float(cast(float, peer_voltage)) if peer_voltage is not None else None
            ),
            expected_temperature_c=expected.prediction_value if expected else None,
            thermal_residual_c=expected.residual_value if expected else None,
            anomaly_summary=AnomalySummary(
                active=anomaly is not None,
                peak_score=anomaly.peak_score if anomaly else None,
                detector=anomaly.detector_name if anomaly else None,
            ),
            freshness=freshness(row.timestamp_utc, self.settings),
            data_provenance=provenance(
                ProvenanceType.SIMULATED,
                "rack telemetry",
                "schema-v1",
                row.timestamp_utc,
            ),
        )

    def anomalies(
        self,
        rack_id: str,
        *,
        start: datetime | None,
        end: datetime | None,
        detector: str | None,
        active_only: bool,
        limit: int,
        offset: int,
    ) -> AnomalyPage:
        exists = self.session.scalar(
            select(RackTelemetryModel.id).where(RackTelemetryModel.rack_id == rack_id).limit(1)
        )
        if exists is None:
            raise NotFoundError("rack", rack_id)
        filters: list[Any] = [AnomalyEventModel.component_id == rack_id]
        if start is not None:
            filters.append(AnomalyEventModel.end_timestamp >= start)
        if end is not None:
            filters.append(AnomalyEventModel.start_timestamp < end)
        if detector:
            filters.append(AnomalyEventModel.detector_name == detector)
        if active_only:
            filters.append(AnomalyEventModel.end_timestamp >= datetime.now(UTC))
        total = (
            self.session.scalar(select(func.count()).select_from(AnomalyEventModel).where(*filters))
            or 0
        )
        rows = self.session.scalars(
            select(AnomalyEventModel)
            .where(*filters)
            .order_by(desc(AnomalyEventModel.start_timestamp))
            .limit(limit)
            .offset(offset)
        ).all()
        now = datetime.now(UTC)
        items = tuple(
            AnomalyEventResponse(
                event_id=row.anomaly_event_id,
                rack_id=row.component_id,
                detector=row.detector_name,
                detector_version=row.detector_version,
                start=utc(row.start_timestamp),
                end=utc(row.end_timestamp),
                peak_score=row.peak_score,
                mean_score=row.mean_score,
                supporting_signals=tuple(json.loads(row.supporting_signals_json)),
                status="ACTIVE" if utc(row.end_timestamp) >= now else "ENDED",
                provenance=provenance(
                    ProvenanceType.DERIVED,
                    "anomaly detector",
                    row.detector_version,
                    row.created_at,
                    model_version=row.detector_version,
                    feature_set_version=row.feature_set_version,
                ),
            )
            for row in rows
        )
        return AnomalyPage(total=total, limit=limit, offset=offset, items=items)


class AlertQueryService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def list(
        self,
        *,
        asset_id: str | None,
        component_id: str | None,
        status: str | None,
        priority_level: str | None,
        alert_type: str | None,
        start: datetime | None,
        end: datetime | None,
        limit: int,
        offset: int,
    ) -> AlertPage:
        filters: list[Any] = []
        for value, column in (
            (asset_id, AlertModel.asset_id),
            (component_id, AlertModel.component_id),
            (status, AlertModel.status),
            (priority_level, AlertModel.priority_level),
            (alert_type, AlertModel.alert_type),
        ):
            if value is not None:
                filters.append(column == value)
        if start is not None:
            filters.append(AlertModel.updated_at_utc >= start)
        if end is not None:
            filters.append(AlertModel.opened_at_utc < end)
        total = (
            self.session.scalar(select(func.count()).select_from(AlertModel).where(*filters)) or 0
        )
        rows = self.session.scalars(
            select(AlertModel)
            .where(*filters)
            .order_by(desc(AlertModel.updated_at_utc), desc(AlertModel.priority_score))
            .limit(limit)
            .offset(offset)
        ).all()
        items = tuple(
            AlertResponse(
                alert_id=row.alert_key,
                asset_id=row.asset_id,
                component=row.component_id,
                component_type=row.component_type,
                alert_type=AlertType(row.alert_type),
                status=AlertStatus(row.status),
                priority_score=row.priority_score,
                priority_level=row.priority_level,
                failure_probability=row.failure_probability,
                risk_horizon=row.risk_horizon_hours,
                model_confidence=row.model_confidence,
                technical_severity=row.technical_severity,
                commercial_severity=row.commercial_severity,
                affected_power_mw=row.affected_power_mw,
                affected_energy_mwh=row.affected_energy_mwh,
                revenue_at_risk_eur=row.revenue_at_risk_eur,
                anomaly_score=row.anomaly_score,
                limiting_factor=row.limiting_factor,
                supporting_signals=tuple(json.loads(row.supporting_signals_json)),
                opened_at=utc(row.opened_at_utc),
                updated_at=utc(row.updated_at_utc),
                resolved_at=utc(row.resolved_at_utc) if row.resolved_at_utc else None,
                source_versions=json.loads(row.source_versions_json),
                data_provenance=provenance(
                    ProvenanceType.DECISION_SUPPORT,
                    "alert priority service",
                    json.loads(row.source_versions_json).get("alert_formula", "alert-formula-v1"),
                    row.updated_at_utc,
                ),
            )
            for row in rows
        )
        return AlertPage(total=total, limit=limit, offset=offset, items=items)


class MarketService:
    def __init__(self, session: Session, settings: APISettings) -> None:
        self.session = session
        self.settings = settings

    def latest(self) -> MarketLatestResponse:
        observed_at = self.session.scalar(
            select(func.max(MarketDataModel.timestamp_utc)).where(
                MarketDataModel.market_region == "DE-LU"
            )
        )
        observed_rows: list[MarketDataModel] = []
        if observed_at is not None:
            observed_rows = list(
                self.session.scalars(
                    select(MarketDataModel).where(
                        MarketDataModel.market_region == "DE-LU",
                        MarketDataModel.timestamp_utc == observed_at,
                    )
                ).all()
            )
        forecast = self.session.scalar(
            select(PricePredictionModel)
            .where(
                PricePredictionModel.market_region == "DE-LU", PricePredictionModel.quantile == -1.0
            )
            .order_by(desc(PricePredictionModel.target_timestamp_utc))
            .limit(1)
        )
        if observed_at is None and forecast is None:
            raise DataUnavailableError("Market data is unavailable.")
        values = {row.metric: row.value for row in observed_rows}
        load = values.get("actual_load")
        renewable = values.get("renewable_generation")
        latest_time = max(
            value
            for value in (
                utc(observed_at) if observed_at else None,
                utc(forecast.target_timestamp_utc) if forecast else None,
            )
            if value is not None
        )
        return MarketLatestResponse(
            market_region="DE-LU",
            observed=ObservedMarketFields(
                timestamp_utc=utc(observed_at),
                day_ahead_price_eur_per_mwh=values.get("day_ahead_price"),
                load_mw=load,
                wind_generation_mw=values.get("wind_generation"),
                solar_generation_mw=values.get("solar_generation"),
                renewable_generation_mw=renewable,
            )
            if observed_at
            else None,
            forecast=ForecastMarketFields(
                forecast_origin_utc=utc(forecast.forecast_origin_utc),
                target_timestamp_utc=utc(forecast.target_timestamp_utc),
                day_ahead_price_eur_per_mwh=forecast.predicted_price_eur_per_mwh,
            )
            if forecast
            else None,
            derived=DerivedMarketFields(
                residual_load_mw=load - renewable
                if load is not None and renewable is not None
                else None,
                renewable_share=renewable / load if load and renewable is not None else None,
            ),
            freshness=freshness(latest_time, self.settings),
            observed_provenance=provenance(
                ProvenanceType.REAL,
                "ENTSO-E Transparency Platform",
                "market-ingestion-v1",
                observed_at,
            )
            if observed_at
            else None,
            forecast_provenance=provenance(
                ProvenanceType.MODEL_PREDICTION,
                "price forecast model",
                forecast.model_version,
                forecast.created_at,
                model_version=forecast.model_version,
                feature_set_version=forecast.feature_set_version,
            )
            if forecast
            else None,
            derived_provenance=provenance(
                ProvenanceType.DERIVED, "market feature service", "market-features-v1", latest_time
            ),
        )


class SimulationService:
    """In-memory demo control state, owned by the FastAPI application lifespan."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        self._faults: dict[UUID, FaultInjectionResponse] = {}
        self._lock = Lock()

    def inject(self, request: FaultInjectionRequest) -> FaultInjectionResponse:
        if not self._component_is_valid(request.component_id):
            raise NotFoundError("simulation component", request.component_id)
        if request.start_timestamp is None:
            start, end = default_fault_window(request.duration_minutes)
        else:
            start = request.start_timestamp
            end = start + timedelta(minutes=request.duration_minutes)
        fault_id = uuid4()
        response = FaultInjectionResponse(
            fault_id=fault_id,
            fault_type=request.fault_type,
            component_id=request.component_id,
            start_timestamp=start,
            end_timestamp=end,
            severity=request.severity,
            status="SCHEDULED",
        )
        with self._lock:
            self._faults[fault_id] = response
        return response

    def reset(self, request: SimulationResetRequest) -> SimulationResetResponse:
        if request.asset_id != "BESS-001":
            raise NotFoundError("asset", request.asset_id)
        seed = request.seed if request.seed is not None else self.config.simulation_seed
        self.config = self.config.model_copy(update={"simulation_seed": seed})
        with self._lock:
            self._faults.clear()
        return SimulationResetResponse(asset_id=request.asset_id, seed=seed, status="RESET")

    def _component_is_valid(self, component_id: str) -> bool:
        if component_id in {"SITE", "BESS-001"}:
            return True
        cfg = self.config.battery
        valid = {f"PCS-{index:02d}" for index in range(1, cfg.pcs_count + 1)}
        valid.update(
            f"PCS-{pcs:02d}-RACK-{rack:02d}"
            for pcs in range(1, cfg.pcs_count + 1)
            for rack in range(1, cfg.racks_per_pcs + 1)
        )
        return component_id in valid


class CopilotService:
    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled

    def query(self, request: CopilotQueryRequest) -> CopilotQueryResponse:
        if not self.enabled:
            raise FeatureUnavailableError("AI Copilot is not configured.")
        from besspulse.agents.knowledge import QueryRoute, route_query
        from rag.service import KnowledgeService

        route = route_query(request.query)
        if route == QueryRoute.OPERATIONAL_TOOL:
            raise FeatureUnavailableError(
                "Prompt 13 operational Copilot tools are not present; "
                "current state was not inferred from documentation."
            )
        result = KnowledgeService().search_knowledge_base(request.query)
        if not result.citations:
            return CopilotQueryResponse(status="NO_EVIDENCE", answer=None)
        citations = tuple(
            f"[{item.source_path} - {item.heading}] relevance={score:.3f} id={item.citation_id}"
            for item, score in zip(result.citations, result.scores, strict=True)
        )
        answer = "\n\n".join(item.excerpt for item in result.citations[:2])
        status = "GROUNDED_RETRIEVAL"
        if route == QueryRoute.MIXED:
            status = "PARTIAL_GROUNDED_RETRIEVAL"
            answer = (
                "Current operational evidence is unavailable because Prompt 13 tools are absent. "
                "Documentation context follows; it does not represent current asset state.\n\n"
                f"{answer}"
            )
        return CopilotQueryResponse(
            status=status,
            answer=answer,
            citations=citations,
            tool_calls=(
                {
                    "tool": "search_knowledge_base",
                    "arguments": {"query": request.query},
                    "result_count": len(result.citations),
                },
            ),
        )


class HealthService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def components(self) -> dict[str, str]:
        self.session.execute(select(1)).scalar_one()
        checks = {
            "database": "available",
            "telemetry": self._table_status(BatteryTelemetryModel),
            "availability": self._table_status(AvailabilitySnapshotModel),
            "delivery_risk": self._risk_status(),
            "anomaly": self._table_status(AnomalyEventModel),
            "commercial": self._table_status(RevenueAtRiskModel),
            "market_data": self._table_status(MarketDataModel),
            "copilot": "optional_unavailable",
            "model_artifacts": self._artifact_status(),
        }
        return checks

    @staticmethod
    def _artifact_status() -> str:
        root = Path(__file__).resolve().parents[1] / "artifacts" / "models"
        required = (
            root / "expected_power" / "metadata.json",
            root / "expected_temperature" / "metadata.json",
            root / "delivery_risk" / "6h" / "metadata.json",
            root / "delivery_risk" / "12h" / "metadata.json",
            root / "delivery_risk" / "24h" / "metadata.json",
            root / "price" / "metadata.json",
        )
        return "available" if all(path.is_file() for path in required) else "unavailable"

    def _table_status(self, model: type[Any]) -> str:
        count = self.session.scalar(select(func.count()).select_from(model)) or 0
        return "available" if count else "no_data"

    def _risk_status(self) -> str:
        count = (
            self.session.scalar(
                select(func.count())
                .select_from(ModelPredictionModel)
                .where(
                    ModelPredictionModel.prediction_name.in_(
                        (
                            "failure_probability_6h",
                            "failure_probability_12h",
                            "failure_probability_24h",
                        )
                    )
                )
            )
            or 0
        )
        return "available" if count else "no_data"
