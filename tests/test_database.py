from sqlalchemy import inspect

from besspulse.database import Base, create_engine


def test_database_foundation_defines_required_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert set(inspect(engine).get_table_names()) == {
        "anomaly_events",
        "alerts",
        "assets",
        "availability_snapshots",
        "battery_telemetry",
        "fault_ground_truth",
        "market_data",
        "model_predictions",
        "monitoring_metrics",
        "pcs",
        "pcs_telemetry",
        "price_predictions",
        "rack_telemetry",
        "racks",
        "revenue_at_risk",
    }
