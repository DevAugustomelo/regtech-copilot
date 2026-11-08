"""Orchestrate the complete document ingestion pipeline."""

import asyncio
from dataclasses import dataclass
from pathlib import Path

from regtech_copilot.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResponse,
)
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.ingestion.pdf_loader import PDFLoader
from regtech_copilot.retrieval.lexical_store import LexicalStore
from regtech_copilot.schemas.documents import Document, DocumentChunk
from regtech_copilot.vector_store.base import VectorStore


class IngestionServiceError(Exception):
    """Base exception for document ingestion errors."""


class IngestionEmbeddingError(IngestionServiceError):
    """Raised when embedding generation fails or returns invalid results."""


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
    """Coordinate PDF loading, chunking, embedding, and storage."""

    def __init__(
        self,
        pdf_loader: PDFLoader,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        batch_size: int = 4,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch size must be greater than zero.")

        self.pdf_loader = pdf_loader
        self.chunker = chunker
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self.batch_size = batch_size

    async def _generate_embeddings(
        self,
        chunks: list[DocumentChunk],
    ) -> EmbeddingResponse:
        """Generate embeddings in batches while preserving chunk order."""

        all_embeddings: list[list[float]] = []
        embedding_model: str | None = None
        embedding_dimensions: int | None = None

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]

            try:
                response = await self.embedding_provider.embed(
                    [chunk.content for chunk in batch],
                    input_type="document",
                )
            except EmbeddingProviderError as exc:
                raise IngestionEmbeddingError(
                    "The embedding provider failed to process a batch."
                ) from exc

            if len(response.embeddings) != len(batch):
                raise IngestionEmbeddingError(
                    "The number of generated embeddings does not match "
                    "the number of document chunks in the batch."
                )

            if response.dimensions <= 0:
                raise IngestionEmbeddingError(
                    "The embedding provider returned invalid dimensions."
                )

            if embedding_model is None:
                embedding_model = response.model
                embedding_dimensions = response.dimensions
            elif (
                response.model != embedding_model
                or response.dimensions != embedding_dimensions
            ):
                raise IngestionEmbeddingError(
                    "Embedding model or dimensions changed between batches."
                )

            if any(
                len(embedding) != response.dimensions
                for embedding in response.embeddings
            ):
                raise IngestionEmbeddingError(
                    "An embedding vector has an unexpected number of dimensions."
                )

            all_embeddings.extend(response.embeddings)

        if embedding_model is None or embedding_dimensions is None:
            raise IngestionEmbeddingError("No embeddings were generated.")

        return EmbeddingResponse(
            embeddings=all_embeddings,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        """Process a PDF and persist its chunks in both indexes."""

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

        embedding_response = await self._generate_embeddings(chunks)

        if len(embedding_response.embeddings) != len(chunks):
            raise IngestionEmbeddingError(
                "The number of generated embeddings does not match "
                "the number of document chunks."
            )

        await asyncio.to_thread(
            self.vector_store.add,
            chunks,
            embedding_response.embeddings,
        )

        try:
            await asyncio.to_thread(
                self.lexical_store.add,
                chunks,
            )
        except Exception as exc:
            try:
                await asyncio.to_thread(
                    self.vector_store.delete,
                    document.document_id,
                )
            except Exception as rollback_error:
                raise IngestionServiceError(
                    "Lexical indexing failed and vector-store rollback "
                    "also failed. Rebuild the indexes before serving queries."
                ) from rollback_error

            raise IngestionServiceError(
                "Lexical indexing failed. The document was removed "
                "from the vector store."
            ) from exc

        return IngestionResult(
            document_id=str(document.document_id),
            filename=document.filename,
            page_count=document.page_count,
            chunks_created=len(chunks),
            embedding_model=embedding_response.model,
            embedding_dimensions=embedding_response.dimensions,
        )
