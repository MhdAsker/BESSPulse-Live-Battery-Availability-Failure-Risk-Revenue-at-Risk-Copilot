"""Non-executing model artifact metadata health checks."""

import hashlib
import json
from pathlib import Path
from typing import Any


def artifact_health(path: Path, expected_feature_version: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "CRITICAL", "reason": "artifact_missing"}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = path.with_name("metadata.json")
    if not metadata_path.is_file():
        return {"status": "WARNING", "sha256": digest, "reason": "metadata_missing"}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "CRITICAL", "sha256": digest, "reason": "metadata_invalid"}
    recorded = metadata.get("artifact_sha256") or metadata.get("artifact_hash")
    if recorded and recorded != digest:
        return {"status": "CRITICAL", "sha256": digest, "reason": "hash_mismatch"}
    actual_version = metadata.get("feature_set_version")
    if expected_feature_version and actual_version != expected_feature_version:
        return {"status": "WARNING", "sha256": digest, "reason": "feature_version_mismatch"}
    return {"status": "OK", "sha256": digest, "feature_set_version": actual_version}
