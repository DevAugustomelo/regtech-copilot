"""Tests for embedding data models."""

from regtech_copilot.embeddings.base import EmbeddingResponse


def test_embedding_response_stores_vectors() -> None:
    response = EmbeddingResponse(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        model="test-embedding-model",
        dimensions=3,
    )

    assert len(response.embeddings) == 2
    assert response.dimensions == 3
    assert response.model == "test-embedding-model"


def test_embedding_response_supports_empty_vectors() -> None:
    response = EmbeddingResponse(
        embeddings=[],
        model="test-embedding-model",
        dimensions=3,
    )

    assert response.embeddings == []
