from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from besspulse.database import Base, PricePredictionModel
from models.artifact import load_model_artifact, save_model_artifact
from models.price.dataset import TARGET_COLUMN, build_price_dataset
from models.price.predict import predict_prices
from models.price.storage import store_price_predictions
from models.price.train import train_price_model


def fixture(periods: int = 1300) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    x = np.arange(periods)
    return pd.DataFrame(
        {"timestamp_utc": time, TARGET_COLUMN: 45 + 15 * np.sin(2 * np.pi * x / 24)}
    )


def test_end_to_end_forecast_reload_and_idempotent_storage(tmp_path: Path) -> None:
    source = fixture()
    dataset = build_price_dataset(source)
    result = train_price_model(dataset)
    model_path, _ = save_model_artifact(result.artifact, tmp_path / "artifact")
    loaded = load_model_artifact(model_path)
    np.testing.assert_allclose(
        result.artifact.estimator.predict(result.split.test.predictors),
        loaded.estimator.predict(result.split.test.predictors),
    )
    origin = source["timestamp_utc"].max()
    targets = pd.date_range(origin + pd.Timedelta("1h"), periods=24, freq="h", tz="UTC")
    forecast = predict_prices(loaded, source, targets, origin)
    assert len(forecast) == 24
    assert set(forecast["data_provenance"]) == {"MODEL_PREDICTION"}
    assert TARGET_COLUMN not in loaded.feature_names
    assert not any("battery" in column or "soc" in column for column in loaded.feature_names)

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assert store_price_predictions(session, forecast) == 24
        assert store_price_predictions(session, forecast) == 0
        stored = session.scalars(select(PricePredictionModel)).all()
    assert len(stored) == 24
    assert stored[0].data_provenance == "MODEL_PREDICTION"
    assert stored[0].model_version == "price_forecast_v1"
