"""Tests for the retrieval service."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from regtech_copilot.embeddings.base import EmbeddingResponse
from regtech_copilot.retrieval.service import (
    RetrievalEmbeddingError,
    RetrievalService,
)
from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.base import VectorSearchResult, VectorStore


@pytest.fixture
def embedding_provider() -> AsyncMock:
    """Provide a mocked embedding provider."""

    provider = AsyncMock()
    provider.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        model="nomic-embed-text",
        dimensions=3,
    )
    return provider


@pytest.fixture
def vector_store() -> MagicMock:
    """Provide a mocked vector store."""

    store = MagicMock(spec=VectorStore)
    store.search.return_value = [
        VectorSearchResult(
            chunk=DocumentChunk(
                document_id=uuid4(),
                content="Regulatory requirement about data protection.",
                chunk_index=0,
                page_number=3,
            ),
            distance=0.05,
        )
    ]
    return store


@pytest.fixture
def service(
    embedding_provider: AsyncMock,
    vector_store: MagicMock,
) -> RetrievalService:
    """Create a retrieval service with mocked dependencies."""

    return RetrievalService(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
    )


@pytest.mark.asyncio
async def test_retrieve_returns_relevant_chunks(
    service: RetrievalService,
    embedding_provider: AsyncMock,
    vector_store: MagicMock,
) -> None:
    """Test successful semantic retrieval."""

    results = await service.retrieve("What are the data protection rules?")

    embedding_provider.embed.assert_awaited_once_with(
        ["What are the data protection rules?"]
    )

    vector_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        5,
    )

    assert len(results) == 1
    assert results[0].chunk.page_number == 3
    assert results[0].distance == 0.05


@pytest.mark.asyncio
async def test_retrieve_uses_custom_top_k(
    service: RetrievalService,
    vector_store: MagicMock,
) -> None:
    """Test custom result limits."""

    await service.retrieve("What are the rules?", top_k=3)

    vector_store.search.assert_called_once_with(
        [0.1, 0.2, 0.3],
        3,
    )


@pytest.mark.asyncio
async def test_retrieve_rejects_empty_query(
    service: RetrievalService,
    embedding_provider: AsyncMock,
) -> None:
    """Test that empty queries are rejected."""

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await service.retrieve("   ")

    embedding_provider.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_top_k(
    service: RetrievalService,
    vector_store: MagicMock,
) -> None:
    """Test that invalid result limits are rejected."""

    with pytest.raises(ValueError, match="top_k must be greater"):
        await service.retrieve("Test query", top_k=0)

    vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_rejects_multiple_query_embeddings(
    service: RetrievalService,
    embedding_provider: AsyncMock,
    vector_store: MagicMock,
) -> None:
    """Test that multiple embeddings for one query are rejected."""

    embedding_provider.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        model="nomic-embed-text",
        dimensions=3,
    )

    with pytest.raises(
        RetrievalEmbeddingError,
        match="exactly one embedding",
    ):
        await service.retrieve("Test query")

    vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_embedding_dimensions(
    service: RetrievalService,
    embedding_provider: AsyncMock,
    vector_store: MagicMock,
) -> None:
    """Test that inconsistent embedding dimensions are rejected."""

    embedding_provider.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        model="nomic-embed-text",
        dimensions=4,
    )

    with pytest.raises(
        RetrievalEmbeddingError,
        match="invalid dimensions",
    ):
        await service.retrieve("Test query")

    vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_retrieve_returns_empty_list_when_no_chunks_match(
    service: RetrievalService,
    vector_store: MagicMock,
) -> None:
    """Test retrieval when the vector store has no matching chunks."""

    vector_store.search.return_value = []

    results = await service.retrieve("Unknown regulatory subject")

    assert results == []
