"""FastAPI dependencies for shared application services."""

from typing import Annotated, cast

from fastapi import Depends, Request

from regtech_copilot.generation.service import GenerationService
from regtech_copilot.ingestion.service import IngestionService


def get_ingestion_service(request: Request) -> IngestionService:
    """Retrieve the shared ingestion service from application state."""

    return cast(
        IngestionService,
        request.app.state.ingestion_service,
    )


def get_generation_service(request: Request) -> GenerationService:
    """Retrieve the shared generation service from application state."""

    return cast(
        GenerationService,
        request.app.state.generation_service,
    )


IngestionServiceDependency = Annotated[
    IngestionService,
    Depends(get_ingestion_service),
]

GenerationServiceDependency = Annotated[
    GenerationService,
    Depends(get_generation_service),
]
