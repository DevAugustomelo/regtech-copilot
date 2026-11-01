"""Orchestrate the complete document ingestion pipeline."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from regtech_copilot.embeddings.base import EmbeddingProvider
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.ingestion.pdf_loader import PDFLoader
from regtech_copilot.schemas.documents import Document
from regtech_copilot.vector_store.base import VectorStore


class IngestionServiceError(Exception):
    """Base exception for document ingestion errors."""


class IngestionEmbeddingError(IngestionServiceError):
    """Raised when embedding generation returns invalid results."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of a successfully ingested document."""

    document_id: str
    filename: str
    page_count: int
    chunks_created: int
    embedding_model: str
    embedding_dimensions: int


class IngestionService:
    """Coordinate PDF loading, chunking, embedding, and vector storage."""

    def __init__(
        self,
        pdf_loader: PDFLoader,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self.pdf_loader = pdf_loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        """Process a PDF and persist its chunks and embeddings."""

        document: Document = await asyncio.to_thread(
            self.pdf_loader.load,
            file_path,
        )

        chunks = await asyncio.to_thread(
            self.chunker.chunk,
            document,
        )

        if not chunks:
            raise IngestionServiceError(
                f"No text chunks were generated from '{document.filename}'."
            )

        embedding_response = await self.embedding_provider.embed(
            [chunk.content for chunk in chunks]
        )

        if len(embedding_response.embeddings) != len(chunks):
            raise IngestionEmbeddingError(
                "The number of generated embeddings does not match "
                "the number of document chunks."
            )

        if embedding_response.dimensions <= 0:
            raise IngestionEmbeddingError(
                "The embedding provider returned invalid dimensions."
            )

        await asyncio.to_thread(
            self.vector_store.add,
            chunks,
            embedding_response.embeddings,
        )

        return IngestionResult(
            document_id=str(document.document_id),
            filename=document.filename,
            page_count=document.page_count,
            chunks_created=len(chunks),
            embedding_model=embedding_response.model,
            embedding_dimensions=embedding_response.dimensions,
        )
