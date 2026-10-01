from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from availability.engine import calculate_availability_bundle
from availability.storage import persist_availability_snapshots
from besspulse import BatterySimulator
from besspulse.database import AvailabilitySnapshotModel, Base, create_engine


def test_availability_persistence_is_idempotent_and_directional() -> None:
    run = BatterySimulator().simulate(datetime(2026, 1, 1, tzinfo=UTC), [2.0, -2.0])
    bundle = calculate_availability_bundle(
        pd.DataFrame(row.model_dump() for row in run.site_telemetry),
        pd.DataFrame(row.model_dump() for row in run.pcs_telemetry),
        pd.DataFrame(row.model_dump() for row in run.rack_telemetry),
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert persist_availability_snapshots(session, bundle.snapshots) == 2
        assert persist_availability_snapshots(session, bundle.snapshots) == 0
        stored = session.scalars(select(AvailabilitySnapshotModel)).all()
    assert len(stored) == 2
    assert stored[0].available_discharge_power_mw >= 0
    assert stored[0].available_charge_power_mw >= 0
    assert stored[0].data_provenance == "DERIVED ENGINEERING ANALYTIC"
    assert stored[0].limiting_factor
