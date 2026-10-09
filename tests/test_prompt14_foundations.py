"""Prompt 14 database, retrieval, tracking, and monitoring contracts."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from besspulse.agents.knowledge import QueryRoute, route_query
from besspulse.database import (
    Base,
    MonitoringMetricModel,
    assert_test_database,
    create_database_engine,
    safe_database_label,
)
from monitoring.artifacts import artifact_health
from monitoring.freshness import FreshnessStatus, classify_freshness
from monitoring.metrics import (
    brier_score,
    freshness_seconds,
    matured_labels,
    missingness,
    population_stability_index,
    rolling_brier_24h,
    wasserstein_distance,
)
from monitoring.predictions import classification_performance
from monitoring.schemas import HealthStatus, MonitoringSnapshot
from monitoring.service import MonitoringService, run_monitoring
from rag.citations import validate_citations
from rag.documents import chunk_document, load_approved_document
from rag.embeddings import DeterministicEmbeddingProvider
from rag.index import LocalVectorIndex
from rag.service import KnowledgeService
from tracking.service import ExperimentTracker, TrackingConfig, sanitize_mapping


def test_database_engine_dialects_and_safe_labels() -> None:
    sqlite = create_database_engine("sqlite:///:memory:")
    assert sqlite.dialect.name == "sqlite"
    postgres = create_database_engine(
        "postgresql+psycopg://user:secret@localhost/test_bess"
    )  # secret-scan: allow-test
    assert postgres.pool is not None
    assert "secret" not in safe_database_label(
        "postgresql+psycopg://user:secret@db.example/test_bess"  # secret-scan: allow-test
    )
    assert_test_database(
        "postgresql+psycopg://user:secret@localhost/bess"
    )  # secret-scan: allow-test
    with pytest.raises(ValueError, match="test database"):
        assert_test_database(
            "postgresql+psycopg://user:secret@db.example/production"
        )  # secret-scan: allow-test
    sqlite.dispose()
    postgres.dispose()


def test_deterministic_embeddings_and_incremental_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "guide.md"
    document.write_text("# Availability\n\nBattery availability is bounded by rack and PCS health.")
    import rag.documents as documents

    monkeypatch.setattr(documents, "APPROVED_FILES", (document,))
    monkeypatch.setattr(documents, "APPROVED_DIRS", ())
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    provider = DeterministicEmbeddingProvider()
    assert provider.embed(["same"])[0] == provider.embed(["same"])[0]
    index = LocalVectorIndex(tmp_path / "index.json")
    first = index.build(provider)
    second = index.build(provider)
    assert first == second
    assert first[0].source_path == "guide.md"
    result = KnowledgeService(index, provider).search_knowledge_base("battery availability", 1)
    assert len(result.citations) == 1
    assert validate_citations(result.citations)
    assert result.citations[0].heading == "Availability"
    with pytest.raises(ValueError, match="allowlist"):
        load_approved_document(tmp_path / ".env")


def test_rag_changed_content_reindexed_and_query_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = tmp_path / "runbook.md"
    document.write_text("# Runbook\n\nFirst procedure.")
    import rag.documents as documents

    monkeypatch.setattr(documents, "APPROVED_FILES", (document,))
    monkeypatch.setattr(documents, "APPROVED_DIRS", ())
    monkeypatch.setattr(documents, "ROOT", tmp_path)
    index = LocalVectorIndex(tmp_path / "index.json")
    original = index.build()
    assert chunk_document(document)[0].content_hash == original[0].content_hash
    document.write_text("# Runbook\n\nUpdated procedure.")
    changed = index.build()
    assert changed[0].content_hash != original[0].content_hash
    assert route_query("What is technical availability?") == QueryRoute.KNOWLEDGE_RAG
    assert route_query("What is the current SOC?") == QueryRoute.MIXED
    assert route_query("current SOC") == QueryRoute.OPERATIONAL_TOOL


def test_monitoring_metrics_maturity_and_edge_cases() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    assert freshness_seconds(now - timedelta(seconds=30), now) == 30
    assert missingness(np.array([1.0, np.nan])) == 0.5
    assert population_stability_index(np.arange(20), np.arange(20)) == pytest.approx(0)
    assert (wasserstein_distance(np.arange(3), np.arange(3) + 2) or 0) > 0
    assert population_stability_index(np.array([]), np.array([1])) is None
    times = [now - timedelta(hours=30), now - timedelta(hours=1)]
    indices, labels = matured_labels(times, [1.0, 0.0], 24, now)
    assert indices.tolist() == [0]
    assert labels.tolist() == [1.0]
    assert brier_score(np.array([0.8]), labels) == pytest.approx(0.04)
    rolling = rolling_brier_24h(np.array([0.2, 0.8]), np.array([0.0, 1.0]), times, now)
    assert rolling == pytest.approx(0.04)
    assert classify_freshness(now, now, timedelta(minutes=5)) == FreshnessStatus.FRESH
    assert (
        classify_freshness(now - timedelta(hours=1), now, timedelta(minutes=5))
        == FreshnessStatus.STALE
    )
    assert classify_freshness(None, now, timedelta(minutes=5)) == FreshnessStatus.MISSING
    one_class = classification_performance(np.array([0.1, 0.2]), np.array([0.0, 0.0]))
    assert one_class["pr_auc"] is None
    shifted = population_stability_index(np.arange(100.0), np.arange(100.0) + 100)
    assert shifted is not None and shifted > 0


def test_monitoring_orchestrator_detects_and_labels_drift() -> None:
    snapshots = run_monitoring(
        {"soc": np.arange(100.0)},
        {"soc": np.arange(100.0) + 100},
        asset_id="BESS-001",
        model_name="delivery_risk_12h",
    )
    drift = next(item for item in snapshots if item.metric_name == "psi")
    assert drift.status in {HealthStatus.WARNING, HealthStatus.CRITICAL}
    assert "not a battery fault" in drift.details["interpretation"].lower()


def test_monitoring_persistence_and_artifact_health(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    service = MonitoringService(session)
    snapshot = MonitoringSnapshot(
        observed_at_utc=datetime.now(UTC),
        metric_scope="input",
        metric_name="psi",
        metric_value=0.1,
        status=HealthStatus.OK,
        model_name="risk",
        asset_id="BESS-001",
    )
    service.record(snapshot)
    assert session.query(MonitoringMetricModel).count() == 1
    assert service.list(model_name="risk")[0].metric_name == "psi"
    artifact = tmp_path / "model.joblib"
    assert artifact_health(artifact)["reason"] == "artifact_missing"
    artifact.write_bytes(b"trusted-test-content")
    assert artifact_health(artifact)["reason"] == "metadata_missing"
    digest = __import__("hashlib").sha256(artifact.read_bytes()).hexdigest()
    artifact.with_name("metadata.json").write_text(
        '{"artifact_sha256":"' + digest + '","feature_set_version":"v1"}'
    )
    assert artifact_health(artifact, "v1")["status"] == "OK"
    assert artifact_health(artifact, "v2")["reason"] == "feature_version_mismatch"


def test_tracking_secret_filter_and_local_run(tmp_path: Path) -> None:
    assert sanitize_mapping({"api_key": "secret", "depth": 3}) == {"depth": "3"}
    tracker = ExperimentTracker(
        TrackingConfig(
            tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
            enabled=True,
            fail_open=False,
        )
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")
    with tracker.run(
        "delivery_risk_24h",
        parameters={
            "depth": 2,
            "token": "hidden",
            "dataset_hash": "sha256:test",
            "feature_set_version": "v1",
        },
    ) as run:
        assert run is not None
        run_id = run.info.run_id
        tracker.log_metrics({"brier": 0.2})
        tracker.log_artifact(metadata)
    assert any(tmp_path.iterdir())
    import mlflow

    client = mlflow.tracking.MlflowClient(tracking_uri=tracker.config.tracking_uri)
    recorded = client.get_run(run_id)
    assert recorded.data.metrics["brier"] == 0.2
    assert recorded.data.params["dataset_hash"] == "sha256:test"
    assert "token" not in recorded.data.params
    assert recorded.data.tags["calibration_status"] == "limited"


def test_tracking_remote_failure_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tracking.service as tracking_service

    class BrokenMLflow:
        def set_tracking_uri(self, uri: str) -> None:
            del uri
            raise ConnectionError("offline")

    monkeypatch.setattr(tracking_service, "_mlflow", lambda: BrokenMLflow())
    tracker = ExperimentTracker(
        TrackingConfig(tracking_uri="http://127.0.0.1:1", enabled=True, fail_open=True)
    )
    with tracker.run("expected_power", parameters={"dataset_hash": "abc"}) as run:
        assert run is None


@pytest.mark.postgres
@pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1" or not os.getenv("DATABASE_URL_TEST"),
    reason="opt-in PostgreSQL test",
)
def test_postgres_connectivity_opt_in() -> None:
    url = os.environ["DATABASE_URL_TEST"]
    assert_test_database(url)
    engine = create_database_engine(url)
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT 1").scalar_one() == 1
