"""Persistent catalog for ingested regulatory documents."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


class DocumentCatalogError(Exception):
    """Base exception for document catalog errors."""


class DuplicateDocumentError(DocumentCatalogError):
    """Raised when a document with the same SHA-256 already exists."""


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Metadata associated with an indexed document."""

    document_id: str
    filename: str
    source_path: str
    sha256: str | None
    page_count: int | None
    chunks_created: int
    embedding_model: str | None
    embedding_dimensions: int | None
    created_at: str


class DocumentCatalog:
    """SQLite-backed catalog for indexed PDF documents."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        sha256 TEXT UNIQUE,
                        page_count INTEGER,
                        chunks_created INTEGER NOT NULL,
                        embedding_model TEXT,
                        embedding_dimensions INTEGER,
                        created_at TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise DocumentCatalogError(
                f"Failed to initialize document catalog: {exc}"
            ) from exc

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Create and safely close a SQLite connection."""
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
    def _deserialize(row: sqlite3.Row) -> DocumentRecord:
        """Convert a database row into a document record."""
        return DocumentRecord(
            document_id=row["document_id"],
            filename=row["filename"],
            source_path=row["source_path"],
            sha256=row["sha256"],
            page_count=row["page_count"],
            chunks_created=row["chunks_created"],
            embedding_model=row["embedding_model"],
            embedding_dimensions=row["embedding_dimensions"],
            created_at=row["created_at"],
        )

    def register(
        self,
        *,
        document_id: str,
        filename: str,
        source_path: str,
        sha256: str | None,
        page_count: int | None,
        chunks_created: int,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> DocumentRecord:
        """Register a successfully indexed document."""
        if not document_id.strip():
            raise ValueError("document_id cannot be empty.")

        if not filename.strip():
            raise ValueError("filename cannot be empty.")

        if chunks_created < 0:
            raise ValueError("chunks_created cannot be negative.")

        if page_count is not None and page_count < 0:
            raise ValueError("page_count cannot be negative.")

        if sha256 is not None and len(sha256) != 64:
            raise ValueError("sha256 must contain exactly 64 hexadecimal characters.")

        created_at = datetime.now(UTC).isoformat()

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        filename,
                        source_path,
                        sha256,
                        page_count,
                        chunks_created,
                        embedding_model,
                        embedding_dimensions,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        filename,
                        source_path,
                        sha256,
                        page_count,
                        chunks_created,
                        embedding_model,
                        embedding_dimensions,
                        created_at,
                    ),
                )

                row = connection.execute(
                    """
                    SELECT * FROM documents
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()

        except sqlite3.IntegrityError as exc:
            if "sha256" in str(exc).lower():
                raise DuplicateDocumentError(
                    "A document with this SHA-256 is already registered."
                ) from exc

            raise DocumentCatalogError(
                f"Document registration conflict: {exc}"
            ) from exc

        except sqlite3.Error as exc:
            raise DocumentCatalogError(f"Failed to register document: {exc}") from exc

        if row is None:
            raise DocumentCatalogError(
                "Document registration completed without returning a record."
            )

        return self._deserialize(row)

    def get(self, document_id: str) -> DocumentRecord | None:
        """Retrieve a document by its identifier."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM documents
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise DocumentCatalogError(f"Failed to retrieve document: {exc}") from exc

        return self._deserialize(row) if row is not None else None

    def get_by_sha256(self, sha256: str) -> DocumentRecord | None:
        """Retrieve a document by its file hash."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT * FROM documents
                    WHERE sha256 = ?
                    """,
                    (sha256,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise DocumentCatalogError(
                f"Failed to retrieve document by hash: {exc}"
            ) from exc

        return self._deserialize(row) if row is not None else None

    def list_all(self) -> list[DocumentRecord]:
        """List registered documents, newest first."""
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM documents
                    ORDER BY created_at DESC, filename ASC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise DocumentCatalogError(f"Failed to list documents: {exc}") from exc

        return [self._deserialize(row) for row in rows]

    def delete(self, document_id: str) -> bool:
        """Delete a document record and return whether it existed."""
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM documents
                    WHERE document_id = ?
                    """,
                    (document_id,),
                )
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            raise DocumentCatalogError(
                f"Failed to delete document record: {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of registered documents."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM documents"
                ).fetchone()
        except sqlite3.Error as exc:
            raise DocumentCatalogError(f"Failed to count documents: {exc}") from exc

        return int(row["total"])
