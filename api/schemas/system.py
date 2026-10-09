"""Non-secret build and runtime metadata."""

from api.schemas.common import APIModel


class SystemInfoResponse(APIModel):
    application_version: str
    api_version: str
    environment: str
    build_commit: str
    feature_set_version: str
    database_backend: str
    rag_backend: str
    copilot_status: str
    model_versions: dict[str, str]
