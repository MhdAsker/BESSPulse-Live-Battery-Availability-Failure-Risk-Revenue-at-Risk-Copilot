"""Transparent price baselines fitted only on the supplied training partition."""

from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class LagBaseline:
    def __init__(self, column: str) -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LagBaseline":
        if self.column not in X:
            raise ValueError(f"Missing baseline feature: {self.column}")
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return pd.to_numeric(X[self.column], errors="raise").to_numpy(dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"column": self.column}


class HourOfWeekBaseline:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HourOfWeekBaseline":
        if "hour_of_week" not in X:
            raise ValueError("hour_of_week is required")
        frame = pd.DataFrame({"hour_of_week": X["hour_of_week"], "target": y.to_numpy()})
        self.mapping_ = frame.groupby("hour_of_week")["target"].median().to_dict()
        self.fallback_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return X["hour_of_week"].map(self.mapping_).fillna(self.fallback_).to_numpy(dtype=float)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}


class BlendBaseline:
    def __init__(self) -> None:
        self.hour_of_week = HourOfWeekBaseline()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BlendBaseline":
        self.hour_of_week.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> NDArray[np.float64]:
        values = np.column_stack(
            [
                X["price_lag_24h"].to_numpy(dtype=float),
                X["price_lag_168h"].to_numpy(dtype=float),
                self.hour_of_week.predict(X),
            ]
        )
        return cast(NDArray[np.float64], np.nanmean(values, axis=1))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}
