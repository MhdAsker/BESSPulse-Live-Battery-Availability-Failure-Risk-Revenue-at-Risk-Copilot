"""Future Copilot contract with honest unavailable behavior."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_copilot_service
from api.schemas.copilot import CopilotQueryRequest, CopilotQueryResponse
from api.services import CopilotService

router = APIRouter(prefix="/copilot", tags=["copilot"])


@router.post(
    "/query",
    response_model=CopilotQueryResponse,
    summary="Query the optional grounded Copilot",
)
def query_copilot(
    request: CopilotQueryRequest,
    service: Annotated[CopilotService, Depends(get_copilot_service)],
) -> CopilotQueryResponse:
    return service.query(request)
