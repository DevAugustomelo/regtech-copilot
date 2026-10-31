"""Domain models for documents and document chunks."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Document:
    """Represents a source document processed by the application."""

    filename: str
    content: str
    page_count: int
    document_id: UUID = field(default_factory=uuid4)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Represents a chunk extracted from a source document."""

    document_id: UUID
    content: str
    chunk_index: int
    chunk_id: UUID = field(default_factory=uuid4)
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
