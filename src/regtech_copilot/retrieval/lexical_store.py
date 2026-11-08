"""Persistent lexical search using SQLite FTS5 and BM25."""

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from regtech_copilot.schemas.documents import DocumentChunk


class LexicalStoreError(Exception):
    """Base exception for lexical storage errors."""


@dataclass(frozen=True, slots=True)
class LexicalSearchResult:
    """A chunk retrieved through lexical search."""

    chunk: DocumentChunk
    score: float


class LexicalStore(Protocol):
    """Contract for persistent lexical search providers."""

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks."""
        ...

    def replace_all(self, chunks: list[DocumentChunk]) -> None:
        """Replace the complete lexical index."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LexicalSearchResult]:
        """Retrieve chunks ranked by BM25."""
        ...

    def delete(self, document_id: UUID) -> None:
        """Delete all chunks belonging to a document."""
        ...

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        ...


class SQLiteLexicalStore:
    """Persistent full-text search backed by SQLite FTS5."""

    TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(
                        chunk_id UNINDEXED,
                        document_id UNINDEXED,
                        chunk_index UNINDEXED,
                        page_number UNINDEXED,
                        content,
                        metadata UNINDEXED,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError(
                f"Failed to initialize lexical index: {exc}"
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Create and safely close a short-lived SQLite connection."""

        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _serialize_chunk(chunk: DocumentChunk) -> tuple[str, ...]:
        """Serialize a chunk for SQLite storage."""

        return (
            str(chunk.chunk_id),
            str(chunk.document_id),
            str(chunk.chunk_index),
            str(chunk.page_number or 0),
            chunk.content,
            json.dumps(
                chunk.metadata,
                ensure_ascii=False,
            ),
        )

    @staticmethod
    def _deserialize_chunk(row: sqlite3.Row) -> DocumentChunk:
        """Restore a document chunk from a SQLite row."""

        try:
            metadata = json.loads(row["metadata"])

            if not isinstance(metadata, dict):
                raise ValueError("Chunk metadata must be a JSON object.")

            if not all(isinstance(key, str) for key in metadata):
                raise ValueError("Chunk metadata keys must be strings.")

            chunk = DocumentChunk(
                chunk_id=UUID(row["chunk_id"]),
                document_id=UUID(row["document_id"]),
                content=row["content"],
                chunk_index=int(row["chunk_index"]),
                page_number=int(row["page_number"]) or None,
                metadata=metadata,
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LexicalStoreError(
                "Failed to deserialize a stored document chunk."
            ) from exc

        return chunk

    @classmethod
    def _build_match_query(cls, query: str) -> str:
        """Build a safe FTS5 expression from user-provided text."""

        tokens = cls.TOKEN_PATTERN.findall(query)

        if not tokens:
            return ""

        return " OR ".join(f'"{token}"' for token in tokens)

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update document chunks."""

        if not chunks:
            return

        serialized_chunks = [self._serialize_chunk(chunk) for chunk in chunks]

        chunk_ids = [item[0] for item in serialized_chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk IDs must be unique.")

        try:
            with self._connect() as connection:
                connection.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id = ?",
                    [(chunk_id,) for chunk_id in chunk_ids],
                )
                connection.executemany(
                    """
                    INSERT INTO chunks_fts (
                        chunk_id,
                        document_id,
                        chunk_index,
                        page_number,
                        content,
                        metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    serialized_chunks,
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError(f"Failed to insert lexical chunks: {exc}") from exc

    def replace_all(self, chunks: list[DocumentChunk]) -> None:
        """Atomically replace the complete lexical index."""

        serialized_chunks = [self._serialize_chunk(chunk) for chunk in chunks]

        chunk_ids = [item[0] for item in serialized_chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Chunk IDs must be unique.")

        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM chunks_fts")

                if serialized_chunks:
                    connection.executemany(
                        """
                        INSERT INTO chunks_fts (
                            chunk_id,
                            document_id,
                            chunk_index,
                            page_number,
                            content,
                            metadata
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        serialized_chunks,
                    )
        except sqlite3.Error as exc:
            raise LexicalStoreError(f"Failed to rebuild lexical index: {exc}") from exc

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[LexicalSearchResult]:
        """Retrieve chunks ranked by BM25."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        match_query = self._build_match_query(query)

        if not match_query:
            return []

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT
                        chunk_id,
                        document_id,
                        chunk_index,
                        page_number,
                        content,
                        metadata,
                        bm25(chunks_fts) AS score
                    FROM chunks_fts
                    WHERE chunks_fts MATCH ?
                    ORDER BY score ASC
                    LIMIT ?
                    """,
                    (match_query, top_k),
                ).fetchall()
        except sqlite3.Error as exc:
            raise LexicalStoreError(f"Failed to search lexical index: {exc}") from exc

        return [
            LexicalSearchResult(
                chunk=self._deserialize_chunk(row),
                score=float(row["score"]),
            )
            for row in rows
        ]

    def delete(self, document_id: UUID) -> None:
        """Delete all chunks belonging to a document."""

        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM chunks_fts WHERE document_id = ?",
                    (str(document_id),),
                )
        except sqlite3.Error as exc:
            raise LexicalStoreError(
                f"Failed to delete lexical document '{document_id}': {exc}"
            ) from exc

    def count(self) -> int:
        """Return the total number of indexed chunks."""

        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM chunks_fts"
                ).fetchone()
        except sqlite3.Error as exc:
            raise LexicalStoreError(f"Failed to count lexical chunks: {exc}") from exc

        return int(row["total"])
