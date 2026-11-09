"""Tests for the RAG query endpoint."""

from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from regtech_copilot.api.dependencies import get_generation_service
from regtech_copilot.api.routes.query import router as query_router
from regtech_copilot.generation.service import GenerationResult
from regtech_copilot.retrieval.service import RetrievalResult
from regtech_copilot.schemas.documents import DocumentChunk


class FakeGenerationService:
    """Fake generation service for API tests."""

    async def generate(
        self,
        query: str,
        top_k: int | None = None,
    ) -> GenerationResult:
        chunk = DocumentChunk(
            document_id=uuid4(),
            chunk_id=uuid4(),
            chunk_index=0,
            page_number=5,
            content="A LGPD estabelece regras para o tratamento de dados pessoais.",
            metadata={"filename": "lgpd.pdf"},
        )

        return GenerationResult(
            answer=f"Resposta fundamentada para: {query}",
            sources=(
                RetrievalResult(
                    chunk=chunk,
                    rrf_score=0.025,
                ),
            ),
            provider="nvidia",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
        )


@pytest.fixture
def client() -> TestClient:
    """Create a test client with a fake generation service."""

    app = FastAPI()
    app.include_router(query_router)

    app.dependency_overrides[get_generation_service] = lambda: FakeGenerationService()

    return TestClient(app)


def test_query_returns_answer_and_sources(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"query": "O que é a LGPD?"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Resposta fundamentada para: O que é a LGPD?"
    assert data["provider"] == "nvidia"
    assert data["model"] == "test-model"
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50

    assert len(data["sources"]) == 1
    assert data["sources"][0]["reference"] == "F1"
    assert data["sources"][0]["filename"] == "lgpd.pdf"
    assert data["sources"][0]["page_number"] == 5
    assert data["sources"][0]["rrf_score"] == 0.025


def test_query_rejects_empty_question(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={"query": "   "},
    )

    assert response.status_code == 422


def test_query_rejects_top_k_above_limit(client: TestClient) -> None:
    response = client.post(
        "/query",
        json={
            "query": "O que é a LGPD?",
            "top_k": 21,
        },
    )

    assert response.status_code == 422
