from datetime import UTC

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from besspulse.database import Base, ModelPredictionModel
from models.experiment import run_demo_experiment
from models.storage import persist_predictions


def test_end_to_end_expected_behavior_experiment(tmp_path) -> None:
    summary = run_demo_experiment(tmp_path, intervals=120, fast=True)
    assert summary["simulation"]["fault_events"] == 2
    assert summary["power"]["selected_model"] in {
        "naive",
        "ridge_linear",
        "random_forest",
        "lightgbm",
    }
    assert summary["temperature"]["selected_model"] in {
        "naive",
        "ridge_linear",
        "random_forest",
        "lightgbm",
    }
    assert summary["power"]["test_metrics"]["rmse"] >= 0
    assert summary["temperature"]["test_metrics"]["rmse"] >= 0
    assert summary["temperature"]["unseen_rack_diagnostic"]["rmse"] >= 0
    assert (tmp_path / "expected_power" / "model.joblib").exists()
    assert (tmp_path / "expected_temperature" / "model.joblib").exists()
    assert (tmp_path / "experiment_summary.json").exists()


def test_prediction_persistence_preserves_prediction_and_residual_provenance() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    frame = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01", tz=UTC)],
            "asset_id": ["BESS-001"],
            "expected_actual_power_mw": [9.0],
            "actual_power_mw": [8.0],
            "power_residual_mw": [-1.0],
        }
    )
    with Session(engine) as session:
        assert (
            persist_predictions(
                session,
                frame,
                entity_column="asset_id",
                prediction_column="expected_actual_power_mw",
                model_name="ridge_linear",
                model_version="expected_power_v1",
                actual_column="actual_power_mw",
                residual_column="power_residual_mw",
            )
            == 1
        )
        stored = session.scalar(select(ModelPredictionModel))
    assert stored is not None
    assert stored.prediction_provenance == "MODEL_PREDICTION"
    assert stored.residual_provenance == "DERIVED"
    assert stored.actual_value == 8.0
