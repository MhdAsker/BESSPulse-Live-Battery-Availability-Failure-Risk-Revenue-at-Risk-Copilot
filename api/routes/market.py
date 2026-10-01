"""Latest observed and forecast market context endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_market_service
from api.schemas.market import MarketLatestResponse
from api.services import MarketService

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/latest", response_model=MarketLatestResponse, summary="Latest DE-LU market context")
def latest_market(
    service: Annotated[MarketService, Depends(get_market_service)],
) -> MarketLatestResponse:
    return service.latest()
