"""Tests for the document upload endpoint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regtech_copilot.api.dependencies import get_ingestion_service
from regtech_copilot.api.routes.documents import router as documents_router
from regtech_copilot.ingestion.service import IngestionResult


class FakeIngestionService:
    """Fake ingestion service for API tests."""

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        return IngestionResult(
            document_id="test-document-id",
            filename=Path(file_path).name,
            page_count=10,
            chunks_created=15,
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create a test client with a fake ingestion service."""

    app = FastAPI()
    app.include_router(documents_router)

    app.state.settings = SimpleNamespace(
        documents_path=str(tmp_path),
        max_upload_size_mb=1,
    )

    app.dependency_overrides[get_ingestion_service] = lambda: FakeIngestionService()

    return TestClient(app)


def test_upload_pdf_returns_ingestion_result(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={
            "file": (
                "lgpd.pdf",
                b"%PDF-1.4\nFake PDF content",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["document_id"] == "test-document-id"
    assert data["filename"] == "lgpd.pdf"
    assert data["page_count"] == 10
    assert data["chunks_created"] == 15
    assert data["embedding_model"] == "nomic-embed-text"
    assert data["embedding_dimensions"] == 768


def test_upload_rejects_non_pdf_extension(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("document.txt", b"Some text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf_signature(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={
            "file": (
                "document.pdf",
                b"This is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400


def test_upload_rejects_file_above_size_limit(client: TestClient) -> None:
    content = b"%PDF-1.4\n" + (b"x" * (1024 * 1024))

    response = client.post(
        "/documents",
        files={"file": ("large.pdf", content, "application/pdf")},
    )

    assert response.status_code == 413
