"""Production configuration, metadata, packaging, and security contracts."""

from pathlib import Path

import pytest
import yaml
from api.config import APISettings
from api.main import create_app
from fastapi.testclient import TestClient
from pydantic import ValidationError
from scripts.check_secrets import scan
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from besspulse.database import Base
from rag.index import LocalVectorIndex
from rag.service import KnowledgeService

ROOT = Path(__file__).parents[1]


def test_production_configuration_is_strict() -> None:
    with pytest.raises(ValidationError, match="PostgreSQL"):
        APISettings(app_env="production", database_url="sqlite:///production.db")
    with pytest.raises(ValidationError, match="wildcard"):
        APISettings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@db.example/bess",
            api_cors_origins=("*",),
        )
    with pytest.raises(ValidationError, match="Simulation controls"):
        APISettings(
            app_env="production",
            database_url="postgresql+psycopg://user:password@db.example/bess",
            api_cors_origins=("https://dashboard.example",),
            enable_simulation_control_api=True,
        )


def test_production_omits_mutation_routes_and_system_info_is_safe() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    settings = APISettings(
        app_env="production",
        database_url="postgresql+psycopg://user:password@db.example/bess",
        api_cors_origins=("https://dashboard.example",),
        build_commit="abc123",
    )
    app = create_app(settings, factory)
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/v1/simulation/fault" not in paths
        response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["environment"] == "production"
    assert payload["database_backend"] == "postgresql"
    assert "password" not in response.text
    assert "db.example" not in response.text
    engine.dispose()


def test_liveness_does_not_require_database() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with TestClient(create_app(APISettings(), factory)) as client:
        response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["database_status"] == "not_checked"
    engine.dispose()


def test_runtime_rag_does_not_build_missing_index(tmp_path: Path) -> None:
    index = LocalVectorIndex(tmp_path / "missing" / "index.json")
    result = KnowledgeService(index=index).search_knowledge_base("availability")
    assert result.citations == ()
    assert not index.path.exists()


def test_container_and_deployment_manifests_are_secret_safe() -> None:
    for dockerfile in (ROOT / "Dockerfile", ROOT / "Dockerfile.dashboard"):
        content = dockerfile.read_text(encoding="utf-8")
        assert "USER besspulse" in content
        assert "COPY .env" not in content
        assert "--reload" not in content
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert set(compose["services"]) == {"postgres", "migrate", "api", "dashboard"}
    render = yaml.safe_load((ROOT / "render.yaml").read_text(encoding="utf-8"))
    assert {service["name"] for service in render["services"]} == {
        "besspulse-api",
        "besspulse-dashboard",
    }


def test_streamlit_builtin_navigation_is_disabled() -> None:
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert "showSidebarNavigation = false" in config


def test_repository_secret_scanner_is_clean() -> None:
    assert scan() == []
