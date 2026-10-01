"""SQLAlchemy 2.x persistence foundation, independent of simulator physics."""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class Asset(Base):
    __tablename__ = "assets"
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    rated_power_mw: Mapped[float] = mapped_column(Float)
    rated_energy_mwh: Mapped[float] = mapped_column(Float)
    data_provenance: Mapped[str] = mapped_column(String(32), default="SIMULATED")
    pcs_units: Mapped[list["PCSModel"]] = relationship(back_populates="asset")


class PCSModel(Base):
    __tablename__ = "pcs"
    id: Mapped[int] = mapped_column(primary_key=True)
    pcs_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    asset_pk: Mapped[int] = mapped_column(ForeignKey("assets.id"))
    rated_power_mw: Mapped[float] = mapped_column(Float)
    asset: Mapped[Asset] = relationship(back_populates="pcs_units")
    racks: Mapped[list["RackModel"]] = relationship(back_populates="pcs")


class RackModel(Base):
    __tablename__ = "racks"
    id: Mapped[int] = mapped_column(primary_key=True)
    rack_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    pcs_pk: Mapped[int] = mapped_column(ForeignKey("pcs.id"))
    rated_power_mw: Mapped[float] = mapped_column(Float)
    rated_energy_mwh: Mapped[float] = mapped_column(Float)
    pcs: Mapped[PCSModel] = relationship(back_populates="racks")


class BatteryTelemetryModel(Base):
    __tablename__ = "battery_telemetry"
    __table_args__ = (Index("ix_battery_telemetry_asset_time", "asset_id", "timestamp_utc"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    site_requested_power_mw: Mapped[float] = mapped_column(Float)
    site_actual_power_mw: Mapped[float] = mapped_column(Float)
    available_power_mw: Mapped[float] = mapped_column(Float)
    available_energy_mwh: Mapped[float] = mapped_column(Float)
    site_soc: Mapped[float] = mapped_column(Float)
    rte: Mapped[float] = mapped_column(Float)
    ambient_temperature_c: Mapped[float] = mapped_column(Float)
    available_racks: Mapped[int] = mapped_column(Integer)
    available_pcs: Mapped[int] = mapped_column(Integer)
    operating_mode: Mapped[str] = mapped_column(String(32))
    alarm_count: Mapped[int] = mapped_column(Integer)
    data_provenance: Mapped[str] = mapped_column(String(32))


class PCSTelemetryModel(Base):
    __tablename__ = "pcs_telemetry"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    pcs_id: Mapped[str] = mapped_column(String(64), index=True)
    requested_power_mw: Mapped[float] = mapped_column(Float)
    actual_power_mw: Mapped[float] = mapped_column(Float)
    efficiency: Mapped[float] = mapped_column(Float)
    temperature_c: Mapped[float] = mapped_column(Float)
    availability: Mapped[bool] = mapped_column(Boolean)
    derating_factor: Mapped[float] = mapped_column(Float)
    alarm_state: Mapped[bool] = mapped_column(Boolean)
    data_provenance: Mapped[str] = mapped_column(String(32))


class RackTelemetryModel(Base):
    __tablename__ = "rack_telemetry"
    __table_args__ = (Index("ix_rack_telemetry_rack_time", "rack_id", "timestamp_utc"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    rack_id: Mapped[str] = mapped_column(String(64), index=True)
    pcs_id: Mapped[str] = mapped_column(String(64), index=True)
    soc: Mapped[float] = mapped_column(Float)
    soh_proxy: Mapped[float] = mapped_column(Float)
    voltage_v: Mapped[float] = mapped_column(Float)
    current_a: Mapped[float] = mapped_column(Float)
    temperature_mean_c: Mapped[float] = mapped_column(Float)
    temperature_min_c: Mapped[float] = mapped_column(Float)
    temperature_max_c: Mapped[float] = mapped_column(Float)
    temperature_spread_c: Mapped[float] = mapped_column(Float)
    voltage_spread_v: Mapped[float] = mapped_column(Float)
    requested_power_mw: Mapped[float] = mapped_column(Float)
    actual_power_mw: Mapped[float] = mapped_column(Float)
    charge_energy_mwh: Mapped[float] = mapped_column(Float)
    discharge_energy_mwh: Mapped[float] = mapped_column(Float)
    cumulative_throughput_mwh: Mapped[float] = mapped_column(Float)
    equivalent_full_cycles: Mapped[float] = mapped_column(Float)
    rte: Mapped[float] = mapped_column(Float)
    availability: Mapped[bool] = mapped_column(Boolean)
    alarm_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operating_state: Mapped[str] = mapped_column(String(32))
    data_provenance: Mapped[str] = mapped_column(String(32))


class FaultGroundTruthModel(Base):
    __tablename__ = "fault_ground_truth"
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    fault_id: Mapped[str] = mapped_column(String(64), index=True)
    fault_type: Mapped[str] = mapped_column(String(64))
    component_id: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[float] = mapped_column(Float)
    ground_truth_label: Mapped[str] = mapped_column(String(128))


class MarketDataModel(Base):
    """Long-form ENTSO-E observation with raw-source lineage."""

    __tablename__ = "market_data"
    __table_args__ = (
        UniqueConstraint(
            "market_region",
            "metric",
            "timestamp_utc",
            "series_type",
            "timeseries_mrid",
            "raw_content_hash",
            name="uq_market_observation_source",
        ),
        Index("ix_market_region_metric_timestamp", "market_region", "metric", "timestamp_utc"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_region: Mapped[str] = mapped_column(String(32), index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))
    series_type: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(64))
    data_provenance: Mapped[str] = mapped_column(String(32))
    retrieved_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_content_hash: Mapped[str] = mapped_column(String(64), index=True)
    document_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    timeseries_mrid: Mapped[str] = mapped_column(String(128), default="")
    business_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    process_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    psr_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resolution: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ModelPredictionModel(Base):
    """Expected-behavior prediction and optional observed residual."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint(
            "timestamp_utc",
            "entity_id",
            "model_version",
            "prediction_name",
            name="uq_model_prediction_identity",
        ),
        Index(
            "ix_model_prediction_entity_time",
            "entity_id",
            "timestamp_utc",
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    entity_id: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    prediction_name: Mapped[str] = mapped_column(String(64))
    prediction_value: Mapped[float] = mapped_column(Float)
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    residual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_horizon_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)
    decision_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_class: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    prediction_provenance: Mapped[str] = mapped_column(String(32), default="MODEL_PREDICTION")
    residual_provenance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feature_set_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class PricePredictionModel(Base):
    """Timestamp-aligned market price forecast with explicit forecast origin."""

    __tablename__ = "price_predictions"
    __table_args__ = (
        UniqueConstraint(
            "forecast_origin_utc",
            "target_timestamp_utc",
            "market_region",
            "model_version",
            "quantile",
            name="uq_price_prediction_identity",
        ),
        Index(
            "ix_price_prediction_region_target",
            "market_region",
            "target_timestamp_utc",
        ),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    forecast_origin_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    target_timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    market_region: Mapped[str] = mapped_column(String(32), index=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), index=True)
    quantile: Mapped[float] = mapped_column(Float, default=-1.0)
    predicted_price_eur_per_mwh: Mapped[float] = mapped_column(Float)
    horizon_minutes: Mapped[int] = mapped_column(Integer)
    feature_set_version: Mapped[str] = mapped_column(String(32))
    data_provenance: Mapped[str] = mapped_column(String(32), default="MODEL_PREDICTION")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AnomalyEventModel(Base):
    """Component anomaly event; no commercial priority semantics."""

    __tablename__ = "anomaly_events"
    __table_args__ = (
        UniqueConstraint("anomaly_event_id", name="uq_anomaly_event_id"),
        Index("ix_anomaly_component_start", "component_id", "start_timestamp"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    anomaly_event_id: Mapped[str] = mapped_column(String(64), index=True)
    component_id: Mapped[str] = mapped_column(String(128), index=True)
    component_type: Mapped[str] = mapped_column(String(32))
    detector_name: Mapped[str] = mapped_column(String(64))
    detector_version: Mapped[str] = mapped_column(String(64))
    start_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    end_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    peak_score: Mapped[float] = mapped_column(Float)
    mean_score: Mapped[float] = mapped_column(Float)
    supporting_signals_json: Mapped[str] = mapped_column(String)
    feature_set_version: Mapped[str] = mapped_column(String(32))
    data_provenance: Mapped[str] = mapped_column(String(64), default="DERIVED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AvailabilitySnapshotModel(Base):
    """Deterministic site capability snapshot with directional quantities."""

    __tablename__ = "availability_snapshots"
    __table_args__ = (
        UniqueConstraint("timestamp_utc", "asset_id", name="uq_availability_snapshot_identity"),
        Index("ix_availability_asset_time", "asset_id", "timestamp_utc"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    technical_availability: Mapped[float] = mapped_column(Float)
    rack_technical_availability: Mapped[float] = mapped_column(Float)
    pcs_technical_availability: Mapped[float] = mapped_column(Float)
    known_component_fraction: Mapped[float] = mapped_column(Float)
    available_discharge_power_mw: Mapped[float] = mapped_column(Float)
    available_charge_power_mw: Mapped[float] = mapped_column(Float)
    available_discharge_energy_mwh: Mapped[float] = mapped_column(Float)
    available_charge_energy_mwh: Mapped[float] = mapped_column(Float)
    discharge_power_availability: Mapped[float] = mapped_column(Float)
    charge_power_availability: Mapped[float] = mapped_column(Float)
    discharge_energy_availability: Mapped[float] = mapped_column(Float)
    charge_energy_availability: Mapped[float] = mapped_column(Float)
    requested_power_mw: Mapped[float] = mapped_column(Float)
    requested_power_availability: Mapped[float | None] = mapped_column(Float, nullable=True)
    requested_energy_availability: Mapped[float | None] = mapped_column(Float, nullable=True)
    sustainable_request_duration_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    available_racks: Mapped[int] = mapped_column(Integer)
    unknown_racks: Mapped[int] = mapped_column(Integer)
    available_pcs: Mapped[int] = mapped_column(Integer)
    unknown_pcs: Mapped[int] = mapped_column(Integer)
    limiting_factor: Mapped[str] = mapped_column(String(64))
    limiting_factors_json: Mapped[str] = mapped_column(String)
    limiting_component_ids_json: Mapped[str] = mapped_column(String)
    data_provenance: Mapped[str] = mapped_column(String(64), default="DERIVED ENGINEERING ANALYTIC")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class RevenueAtRiskModel(Base):
    """Reproducible counterfactual commercial benchmark summary."""

    __tablename__ = "revenue_at_risk"
    __table_args__ = (
        UniqueConstraint("calculation_hash", name="uq_revenue_at_risk_calculation"),
        Index("ix_revenue_at_risk_asset_interval", "asset_id", "start_timestamp_utc"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    start_timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    market_mode: Mapped[str] = mapped_column(String(32))
    price_source: Mapped[str] = mapped_column(String(64))
    healthy_gross_revenue_eur: Mapped[float] = mapped_column(Float)
    healthy_degradation_cost_eur: Mapped[float] = mapped_column(Float)
    healthy_net_revenue_eur: Mapped[float] = mapped_column(Float)
    current_gross_revenue_eur: Mapped[float] = mapped_column(Float)
    current_degradation_cost_eur: Mapped[float] = mapped_column(Float)
    current_net_revenue_eur: Mapped[float] = mapped_column(Float)
    revenue_at_risk_eur: Mapped[float] = mapped_column(Float)
    revenue_at_risk_fraction: Mapped[float | None] = mapped_column(Float, nullable=True)
    power_derating_impact_eur: Mapped[float] = mapped_column(Float)
    energy_capacity_impact_eur: Mapped[float] = mapped_column(Float)
    efficiency_impact_eur: Mapped[float] = mapped_column(Float)
    availability_impact_eur: Mapped[float] = mapped_column(Float)
    interaction_unattributed_eur: Mapped[float] = mapped_column(Float)
    model_or_price_version: Mapped[str] = mapped_column(String(128))
    capability_version: Mapped[str] = mapped_column(String(64))
    price_dataset_hash: Mapped[str] = mapped_column(String(64))
    capability_dataset_hash: Mapped[str] = mapped_column(String(64))
    optimization_config_hash: Mapped[str] = mapped_column(String(64))
    calculation_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    attribution_json: Mapped[str] = mapped_column(String)
    data_provenance: Mapped[str] = mapped_column(String(64), default="COUNTERFACTUAL")
    disclaimer: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AlertModel(Base):
    """Auditable lifecycle record for transparent decision-support alerts."""

    __tablename__ = "alerts"
    __table_args__ = (
        UniqueConstraint("alert_key", name="uq_alert_key"),
        Index("ix_alert_asset_status", "asset_id", "status"),
        Index("ix_alert_status_priority", "status", "priority_score"),
    )
    id: Mapped[int] = mapped_column(primary_key=True)
    alert_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    component_id: Mapped[str] = mapped_column(String(128), index=True)
    component_type: Mapped[str] = mapped_column(String(32))
    alert_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    opened_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    resolved_at_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    priority_score: Mapped[float] = mapped_column(Float)
    priority_level: Mapped[str] = mapped_column(String(32))
    failure_probability: Mapped[float] = mapped_column(Float)
    risk_horizon_hours: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_confidence: Mapped[float] = mapped_column(Float)
    technical_severity: Mapped[float] = mapped_column(Float)
    commercial_severity: Mapped[float] = mapped_column(Float)
    affected_power_mw: Mapped[float] = mapped_column(Float)
    affected_energy_mwh: Mapped[float] = mapped_column(Float)
    revenue_at_risk_eur: Mapped[float] = mapped_column(Float)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    limiting_factor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    supporting_signals_json: Mapped[str] = mapped_column(String)
    source_versions_json: Mapped[str] = mapped_column(String)
    data_provenance: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


def create_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Build a session factory for SQLite locally or PostgreSQL later."""

    url: str = database_url or os.getenv("DATABASE_URL") or "sqlite:///data/besspulse.db"
    engine = create_engine(url)
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_schema(database_url: str | None = None) -> None:
    url: str = database_url or os.getenv("DATABASE_URL") or "sqlite:///data/besspulse.db"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
