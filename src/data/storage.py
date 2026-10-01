"""Immutable, content-addressed raw ENTSO-E response storage."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

from data.schemas import Dataset, RawArtifact


class RawEntsoeStore:
    """Deduplicate identical XML by SHA-256 without overwriting source artifacts."""

    def __init__(self, root: str | Path = "data/raw/entsoe") -> None:
        self.root = Path(root)

    def save(
        self,
        content: bytes,
        *,
        dataset: Dataset,
        region: str,
        requested_start: datetime,
        requested_end: datetime,
        retrieved_at_utc: datetime,
        http_status: int,
    ) -> RawArtifact:
        digest = hashlib.sha256(content).hexdigest()
        safe_region = region.replace("/", "-").replace(" ", "-")
        interval = f"{requested_start:%Y%m%dT%H%MZ}_{requested_end:%Y%m%dT%H%MZ}"
        stem = f"{dataset.value}_{safe_region}_{interval}_{digest[:16]}"
        xml_path = self.root / f"{stem}.xml"
        metadata_path = self.root / f"{stem}.json"
        self.root.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with xml_path.open("xb") as stream:
                stream.write(content)
            created = True
        except FileExistsError:
            if hashlib.sha256(xml_path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Raw artifact hash collision") from None
        metadata = {
            "source": "ENTSO-E Transparency Platform",
            "query_type": dataset.value,
            "market_region": region,
            "requested_start": requested_start.isoformat(),
            "requested_end": requested_end.isoformat(),
            "retrieved_at_utc": retrieved_at_utc.isoformat(),
            "http_status": http_status,
            "content_hash_sha256": digest,
            "data_provenance": "REAL",
        }
        if not metadata_path.exists():
            with metadata_path.open("x", encoding="utf-8") as stream:
                json.dump(metadata, stream, indent=2, sort_keys=True)
                stream.write("\n")
        return RawArtifact(
            xml_path=xml_path,
            metadata_path=metadata_path,
            content_hash=digest,
            created=created,
        )
