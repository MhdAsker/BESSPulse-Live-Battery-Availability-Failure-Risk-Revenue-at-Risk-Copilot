"""Commercial hashes and deterministic summary helpers."""

import hashlib
import json
from typing import Any

import pandas as pd
from pydantic import BaseModel


def frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    schema = "|".join(f"{column}:{normalized[column].dtype}" for column in normalized)
    values = pd.util.hash_pandas_object(normalized, index=False).to_numpy().tobytes()
    return hashlib.sha256(schema.encode() + values).hexdigest()


def config_hash(config: BaseModel) -> str:
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def calculation_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()
