"""ChromaDB vector store implementation."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import chromadb
from chromadb.api.types import Embeddings, Where

from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.base import VectorSearchResult


class ChromaVectorStoreError(Exception):
    """Base exception for ChromaDB vector store errors."""


class ChromaVectorStore:
    """Persistent vector store implementation using ChromaDB."""

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "regtech_documents",
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Collection name cannot be empty.")

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to initialize ChromaDB: {exc}"
            ) from exc

    @staticmethod
    def _validate_embeddings(embeddings: list[list[float]]) -> None:
        """Validate embedding dimensions and numeric values."""

        if not embeddings:
            raise ValueError("Embeddings cannot be empty.")

        dimensions = {len(embedding) for embedding in embeddings}

        if 0 in dimensions:
            raise ValueError("Embeddings cannot be empty.")

        if len(dimensions) != 1:
            raise ValueError("All embeddings must have the same dimensions.")

        for embedding in embeddings:
            for value in embedding:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Embedding values must be finite numbers.")

    @staticmethod
    def _build_chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
        """Build ChromaDB metadata, exposing article reference for filtering."""

        metadata: dict[str, Any] = {
            "document_id": str(chunk.document_id),
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number or 0,
            "metadata": json.dumps(chunk.metadata),
        }

        article_reference = chunk.metadata.get("article_reference")

        if isinstance(article_reference, str) and article_reference:
            metadata["article_reference"] = article_reference

        return metadata

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their corresponding embeddings."""

        if not chunks:
            raise ValueError("Chunks cannot be empty.")

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings."
            )

        self._validate_embeddings(embeddings)

        ids = [str(chunk.chunk_id) for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("Chunk IDs must be unique.")

        try:
            self.collection.upsert(
                ids=ids,
                documents=[chunk.content for chunk in chunks],
                embeddings=cast(Embeddings, embeddings),
                metadatas=[self._build_chunk_metadata(chunk) for chunk in chunks],
            )
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to store document chunks: {exc}"
            ) from exc

    @staticmethod
    def _parse_chunk(
        chunk_id: Any,
        content: Any,
        metadata: Any,
    ) -> DocumentChunk:
        """Convert a stored ChromaDB record into a document chunk."""

        if not isinstance(chunk_id, str):
            raise ChromaVectorStoreError("Retrieved chunk ID is invalid.")

        if not isinstance(content, str):
            raise ChromaVectorStoreError("Retrieved document content is invalid.")

        if not isinstance(metadata, Mapping):
            raise ChromaVectorStoreError("Retrieved document metadata is invalid.")

        document_id = metadata.get("document_id")
        chunk_index = metadata.get("chunk_index")
        page_number = metadata.get("page_number")

        if not isinstance(document_id, str):
            raise ChromaVectorStoreError("Retrieved document ID is invalid.")

        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise ChromaVectorStoreError("Retrieved chunk index is invalid.")

        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise ChromaVectorStoreError("Retrieved page number is invalid.")

        stored_metadata = metadata.get("metadata", "{}")

        if not isinstance(stored_metadata, str):
            raise ChromaVectorStoreError("Retrieved additional metadata is invalid.")

        try:
            parsed_metadata = json.loads(stored_metadata)
        except json.JSONDecodeError as exc:
            raise ChromaVectorStoreError(
                "Failed to decode stored document metadata."
            ) from exc

        if not isinstance(parsed_metadata, dict) or not all(
            isinstance(key, str) for key in parsed_metadata
        ):
            raise ChromaVectorStoreError("Decoded document metadata is invalid.")

        try:
            return DocumentChunk(
                chunk_id=UUID(chunk_id),
                document_id=UUID(document_id),
                content=content,
                chunk_index=chunk_index,
                page_number=page_number or None,
                metadata=cast(dict[str, Any], parsed_metadata),
            )
        except ValueError as exc:
            raise ChromaVectorStoreError(
                "Retrieved document identifiers are invalid."
            ) from exc

    @staticmethod
    def _parse_search_results(result: Any) -> list[VectorSearchResult]:
        """Convert ChromaDB query results into application search results."""

        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]
        distances = result["distances"]

        if (
            not ids
            or not ids[0]
            or documents is None
            or metadatas is None
            or distances is None
            or documents[0] is None
            or metadatas[0] is None
            or distances[0] is None
        ):
            return []

        search_results: list[VectorSearchResult] = []

        for index, chunk_id in enumerate(ids[0]):
            distance = distances[0][index]

            if (
                not isinstance(distance, (int, float))
                or isinstance(distance, bool)
                or not math.isfinite(distance)
            ):
                raise ChromaVectorStoreError("Retrieved distance is invalid.")

            chunk = ChromaVectorStore._parse_chunk(
                chunk_id=chunk_id,
                content=documents[0][index],
                metadata=metadatas[0][index],
            )

            search_results.append(
                VectorSearchResult(
                    chunk=chunk,
                    distance=float(distance),
                )
            )

        return search_results

    def get_all_chunks(self) -> list[DocumentChunk]:
        """Retrieve all stored chunks without loading their embeddings."""

        try:
            result = self.collection.get(
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to retrieve stored document chunks: {exc}"
            ) from exc

        ids = result["ids"]
        documents = result["documents"]
        metadatas = result["metadatas"]

        if not ids:
            return []

        if (
            documents is None
            or metadatas is None
            or len(ids) != len(documents)
            or len(ids) != len(metadatas)
        ):
            raise ChromaVectorStoreError(
                "Stored document records have inconsistent data."
            )

        chunks = [
            self._parse_chunk(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
            )
            for chunk_id, content, metadata in zip(
                ids,
                documents,
                metadatas,
                strict=True,
            )
        ]

        return sorted(
            chunks,
            key=lambda chunk: (str(chunk.document_id), chunk.chunk_index),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Retrieve the most similar chunks."""

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self._validate_embeddings([query_embedding])

        try:
            collection_count = self.collection.count()

            if collection_count == 0:
                return []

            result = self.collection.query(
                query_embeddings=cast(Embeddings, [query_embedding]),
                n_results=min(top_k, collection_count),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to search vector store: {exc}"
            ) from exc

        return self._parse_search_results(result)

    def search_by_article(
        self,
        query_embedding: list[float],
        document_id: UUID,
        article_reference: str,
    ) -> list[VectorSearchResult]:
        """Retrieve all chunks of an article ranked by query similarity."""

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        if not article_reference.strip():
            raise ValueError("Article reference cannot be empty.")

        self._validate_embeddings([query_embedding])

        where: Where = cast(
            Where,
            {
                "$and": [
                    {"document_id": str(document_id)},
                    {"article_reference": article_reference},
                ]
            },
        )

        try:
            collection_count = self.collection.count()

            if collection_count == 0:
                return []

            result = self.collection.query(
                query_embeddings=cast(Embeddings, [query_embedding]),
                n_results=collection_count,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to search article '{article_reference}': {exc}"
            ) from exc

        return self._parse_search_results(result)

    def delete(self, document_id: UUID) -> None:
        """Delete all chunks belonging to a document."""

        try:
            self.collection.delete(where={"document_id": str(document_id)})
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to delete document '{document_id}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of stored chunks."""

        try:
            return self.collection.count()
        except Exception as exc:
            raise ChromaVectorStoreError(
                f"Failed to count stored chunks: {exc}"
            ) from exc
