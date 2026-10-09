"""Safe system metadata endpoint."""

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from sqlalchemy.engine import make_url

from api import __version__
from api.config import APISettings
from api.dependencies import get_api_settings
from api.schemas.system import SystemInfoResponse
from features.registry import FEATURE_SET_VERSION

router = APIRouter(prefix="/system", tags=["system"])
ROOT = Path(__file__).resolve().parents[2]


def _model_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    artifacts = ROOT / "artifacts" / "models"
    if not artifacts.exists():
        return versions
    for path in sorted(artifacts.rglob("metadata.json")):
        try:
            metadata: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict):
            name = str(metadata.get("model_name") or path.parent.name)
            version = str(
                metadata.get("model_version") or metadata.get("artifact_version") or "unknown"
            )
            versions[name] = version
    return versions


@router.get("/info", response_model=SystemInfoResponse, summary="Safe build metadata")
def system_info(
    settings: Annotated[APISettings, Depends(get_api_settings)],
) -> SystemInfoResponse:
    backend = make_url(settings.effective_database_url).get_backend_name()
    return SystemInfoResponse(
        application_version=__version__,
        api_version="v1",
        environment=settings.app_env,
        build_commit=settings.build_commit,
        feature_set_version=FEATURE_SET_VERSION,
        database_backend=backend,
        rag_backend=settings.rag_backend,
        copilot_status="rag_enabled" if settings.rag_enabled else "optional_unavailable",
        model_versions=_model_versions(),
    )
