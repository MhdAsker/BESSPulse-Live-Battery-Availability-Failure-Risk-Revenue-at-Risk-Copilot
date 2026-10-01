"""Future Copilot route contract; no LLM implementation is present."""

from typing import Any

from pydantic import Field

from api.schemas.common import APIModel, Provenance


class CopilotQueryRequest(APIModel):
    asset_id: str
    query: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=128)


class CopilotQueryResponse(APIModel):
    status: str
    answer: str | None
    citations: tuple[str, ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    data_provenance: tuple[Provenance, ...] = ()
