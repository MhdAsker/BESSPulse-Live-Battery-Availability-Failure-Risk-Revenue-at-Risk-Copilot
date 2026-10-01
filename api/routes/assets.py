"""Asset overview and persisted analytical outputs."""

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import (
    get_asset_service,
    get_availability_service,
    get_commercial_service,
    get_risk_service,
)
from api.schemas.assets import AssetStatus, AssetSummary
from api.schemas.availability import AvailabilityResponse
from api.schemas.commercial import RevenueRiskResponse
from api.schemas.risk import DeliveryRiskResponse
from api.services import AssetService, AvailabilityService, CommercialService, RiskService

router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("", response_model=tuple[AssetSummary, ...], summary="List configured assets")
def list_assets(
    service: Annotated[AssetService, Depends(get_asset_service)],
) -> tuple[AssetSummary, ...]:
    return service.list_assets()


@router.get(
    "/{asset_id}/status", response_model=AssetStatus, summary="Aggregate latest asset status"
)
def asset_status(
    asset_id: str, service: Annotated[AssetService, Depends(get_asset_service)]
) -> AssetStatus:
    return service.status(asset_id)


@router.get(
    "/{asset_id}/availability",
    response_model=AvailabilityResponse,
    summary="Latest directional availability",
)
def availability(
    asset_id: str, service: Annotated[AvailabilityService, Depends(get_availability_service)]
) -> AvailabilityResponse:
    return service.latest(asset_id)


@router.get(
    "/{asset_id}/delivery-risk",
    response_model=DeliveryRiskResponse,
    summary="Latest persisted delivery-risk predictions",
)
def delivery_risk(
    asset_id: str, service: Annotated[RiskService, Depends(get_risk_service)]
) -> DeliveryRiskResponse:
    return service.latest(asset_id)


@router.get(
    "/{asset_id}/revenue-risk",
    response_model=RevenueRiskResponse,
    summary="Latest persisted counterfactual Revenue-at-Risk",
)
def revenue_risk(
    asset_id: str, service: Annotated[CommercialService, Depends(get_commercial_service)]
) -> RevenueRiskResponse:
    return service.latest(asset_id)
