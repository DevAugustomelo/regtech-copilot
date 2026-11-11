"""Tests for document management endpoints."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regtech_copilot.api.dependencies import get_ingestion_service
from regtech_copilot.api.routes.documents import router as documents_router
from regtech_copilot.documents.catalog import DocumentCatalog
from regtech_copilot.ingestion.service import IngestionResult

DOCUMENT_ID = "035a58cb-536b-40b0-b01b-facba17d0144"
PDF_CONTENT = b"%PDF-1.4\nFake PDF content"


class FakeIngestionService:
    """Fake ingestion service for API tests."""

    def __init__(self) -> None:
        self.calls = 0

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        self.calls += 1

        return IngestionResult(
            document_id=DOCUMENT_ID,
            filename=Path(file_path).name,
            page_count=10,
            chunks_created=15,
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
        )


class FakeIndexStore:
    """Fake vector or lexical index for deletion tests."""

    def __init__(self) -> None:
        self.deleted: list[UUID] = []

    def delete(self, document_id: UUID) -> None:
        self.deleted.append(document_id)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Create an isolated API with fake ingestion and index services."""

    app = FastAPI()
    app.include_router(documents_router)

    app.state.settings = SimpleNamespace(
        documents_path=str(tmp_path),
        max_upload_size_mb=1,
    )
    app.state.document_catalog = DocumentCatalog(tmp_path / "documents.sqlite3")
    app.state.document_ingestion_lock = asyncio.Lock()
    app.state.vector_store = FakeIndexStore()
    app.state.lexical_store = FakeIndexStore()
    app.state.fake_ingestion_service = FakeIngestionService()

    app.dependency_overrides[get_ingestion_service] = lambda: (
        app.state.fake_ingestion_service
    )

    return TestClient(app)


def upload_pdf(client: TestClient, content: bytes = PDF_CONTENT):
    """Upload a PDF using the API test client."""

    return client.post(
        "/documents",
        files={"file": ("lgpd.pdf", content, "application/pdf")},
    )


def test_upload_pdf_returns_ingestion_result(client: TestClient) -> None:
    response = upload_pdf(client)

    assert response.status_code == 201

    data = response.json()

    assert data["document_id"] == DOCUMENT_ID
    assert data["filename"] == "lgpd.pdf"
    assert data["page_count"] == 10
    assert data["chunks_created"] == 15
    assert data["embedding_model"] == "nomic-embed-text"
    assert data["embedding_dimensions"] == 768

    record = client.app.state.document_catalog.get(DOCUMENT_ID)
    assert record is not None
    assert record.sha256 is not None
    assert record.chunks_created == 15


def test_upload_rejects_duplicate_pdf(client: TestClient) -> None:
    first_response = upload_pdf(client)
    second_response = upload_pdf(client)

    assert first_response.status_code == 201
    assert second_response.status_code == 409
    assert client.app.state.fake_ingestion_service.calls == 1
    assert client.app.state.document_catalog.count() == 1


def test_upload_rejects_non_pdf_extension(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("document.txt", b"Some text", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_invalid_pdf_signature(client: TestClient) -> None:
    response = client.post(
        "/documents",
        files={"file": ("document.pdf", b"This is not a PDF", "application/pdf")},
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

    response = upload_pdf(client, content)

    assert response.status_code == 413


def test_list_documents_returns_registered_documents(
    client: TestClient,
) -> None:
    upload_pdf(client)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["documents"][0]["document_id"] == DOCUMENT_ID
    assert response.json()["documents"][0]["file_exists"] is True


def test_get_document_returns_details(client: TestClient) -> None:
    upload_pdf(client)

    response = client.get(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json()["document_id"] == DOCUMENT_ID
    assert response.json()["filename"] == "lgpd.pdf"


def test_get_document_returns_404_for_unknown_id(
    client: TestClient,
) -> None:
    response = client.get("/documents/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    assert response.status_code == 404


def test_delete_document_removes_file_and_indexes(
    client: TestClient,
) -> None:
    upload_pdf(client)

    record = client.app.state.document_catalog.get(DOCUMENT_ID)
    assert record is not None

    source_path = Path(record.source_path)
    assert source_path.is_file()

    response = client.delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": DOCUMENT_ID,
        "status": "deleted",
    }
    assert not source_path.exists()
    assert client.app.state.document_catalog.get(DOCUMENT_ID) is None
    assert client.app.state.vector_store.deleted == [UUID(DOCUMENT_ID)]
    assert client.app.state.lexical_store.deleted == [UUID(DOCUMENT_ID)]


def test_delete_document_returns_404_for_unknown_id(
    client: TestClient,
) -> None:
    response = client.delete("/documents/aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")

    assert response.status_code == 404


def test_delete_rejects_file_outside_managed_directory(
    client: TestClient,
    tmp_path: Path,
) -> None:
    external_file = tmp_path.parent / "external.pdf"
    external_file.write_bytes(PDF_CONTENT)

    client.app.state.document_catalog.register(
        document_id=DOCUMENT_ID,
        filename="external.pdf",
        source_path=str(external_file),
        sha256="a" * 64,
        page_count=1,
        chunks_created=1,
    )

    response = client.delete(f"/documents/{DOCUMENT_ID}")

    assert response.status_code == 409
    assert external_file.is_file()
    assert client.app.state.vector_store.deleted == []
    assert client.app.state.lexical_store.deleted == []
