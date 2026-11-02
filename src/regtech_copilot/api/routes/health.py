"""Health check endpoints."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: Literal["ok"]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
async def health_check() -> HealthResponse:
    """Return the API health status."""

    return HealthResponse(status="ok")
