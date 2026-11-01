"""Tests for the document ingestion service."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from regtech_copilot.embeddings.base import EmbeddingResponse
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.ingestion.pdf_loader import PDFLoader
from regtech_copilot.ingestion.service import (
    IngestionEmbeddingError,
    IngestionService,
    IngestionServiceError,
)
from regtech_copilot.schemas.documents import Document, DocumentChunk
from regtech_copilot.vector_store.base import VectorStore


@pytest.fixture
def document() -> Document:
    """Provide a sample document."""

    return Document(
        filename="regulation.pdf",
        content="Regulatory document content.",
        page_count=2,
        document_id=uuid4(),
    )


@pytest.fixture
def chunks(document: Document) -> list[DocumentChunk]:
    """Provide sample document chunks."""

    return [
        DocumentChunk(
            document_id=document.document_id,
            content="First chunk.",
            chunk_index=0,
            page_number=1,
        ),
        DocumentChunk(
            document_id=document.document_id,
            content="Second chunk.",
            chunk_index=1,
            page_number=2,
        ),
    ]


@pytest.fixture
def embedding_response() -> EmbeddingResponse:
    """Provide sample embedding results."""

    return EmbeddingResponse(
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        model="nomic-embed-text",
        dimensions=3,
    )


@pytest.fixture
def dependencies(
    document: Document,
    chunks: list[DocumentChunk],
    embedding_response: EmbeddingResponse,
) -> tuple[MagicMock, MagicMock, AsyncMock, MagicMock]:
    """Provide mocked ingestion dependencies."""

    pdf_loader = MagicMock(spec=PDFLoader)
    pdf_loader.load.return_value = document

    chunker = MagicMock(spec=TextChunker)
    chunker.chunk.return_value = chunks

    embedding_provider = AsyncMock()
    embedding_provider.embed.return_value = embedding_response

    vector_store = MagicMock(spec=VectorStore)

    return pdf_loader, chunker, embedding_provider, vector_store


@pytest.fixture
def service(
    dependencies: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
) -> IngestionService:
    """Create an ingestion service with mocked dependencies."""

    return IngestionService(*dependencies)


@pytest.mark.asyncio
async def test_ingest_processes_document_successfully(
    service: IngestionService,
    dependencies: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
    document: Document,
    chunks: list[DocumentChunk],
    tmp_path: Path,
) -> None:
    """Test the complete ingestion pipeline."""

    pdf_loader, chunker, embedding_provider, vector_store = dependencies
    file_path = tmp_path / "regulation.pdf"

    result = await service.ingest(file_path)

    pdf_loader.load.assert_called_once_with(file_path)
    chunker.chunk.assert_called_once_with(document)

    embedding_provider.embed.assert_awaited_once_with(
        [chunk.content for chunk in chunks]
    )

    vector_store.add.assert_called_once_with(
        chunks,
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    )

    assert result.document_id == str(document.document_id)
    assert result.filename == document.filename
    assert result.page_count == 2
    assert result.chunks_created == 2
    assert result.embedding_model == "nomic-embed-text"
    assert result.embedding_dimensions == 3


@pytest.mark.asyncio
async def test_ingest_raises_error_when_no_chunks_are_generated(
    service: IngestionService,
    dependencies: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
    tmp_path: Path,
) -> None:
    """Test that empty chunk results stop the pipeline."""

    _, chunker, embedding_provider, vector_store = dependencies
    chunker.chunk.return_value = []

    with pytest.raises(IngestionServiceError, match="No text chunks"):
        await service.ingest(tmp_path / "empty.pdf")

    embedding_provider.embed.assert_not_awaited()
    vector_store.add.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_rejects_mismatched_embedding_count(
    service: IngestionService,
    dependencies: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
    tmp_path: Path,
) -> None:
    """Test that mismatched embedding counts are rejected."""

    _, _, embedding_provider, vector_store = dependencies

    embedding_provider.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3]],
        model="nomic-embed-text",
        dimensions=3,
    )

    with pytest.raises(
        IngestionEmbeddingError,
        match="number of generated embeddings",
    ):
        await service.ingest(tmp_path / "regulation.pdf")

    vector_store.add.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_embedding_dimensions(
    service: IngestionService,
    dependencies: tuple[MagicMock, MagicMock, AsyncMock, MagicMock],
    tmp_path: Path,
) -> None:
    """Test that invalid embedding dimensions are rejected."""

    _, _, embedding_provider, vector_store = dependencies

    embedding_provider.embed.return_value = EmbeddingResponse(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        model="nomic-embed-text",
        dimensions=0,
    )

    with pytest.raises(
        IngestionEmbeddingError,
        match="invalid dimensions",
    ):
        await service.ingest(tmp_path / "regulation.pdf")

    vector_store.add.assert_not_called()
