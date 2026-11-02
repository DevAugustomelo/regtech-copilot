"""FastAPI application entry point."""

from fastapi import FastAPI

from regtech_copilot.api.lifespan import lifespan
from regtech_copilot.api.routes.health import router as health_router
from regtech_copilot.core.config import get_settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description="RegTech Copilot RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health_router)

    return app


app = create_app()
