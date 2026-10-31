"""Tests for document domain models."""

from uuid import UUID

from regtech_copilot.schemas.documents import Document, DocumentChunk


def test_document_generates_unique_id() -> None:
    document = Document(
        filename="regulation.pdf",
        content="Regulatory document content.",
        page_count=10,
    )

    assert isinstance(document.document_id, UUID)


def test_document_preserves_metadata() -> None:
    metadata = {"source": "official", "category": "regulation"}

    document = Document(
        filename="regulation.pdf",
        content="Content",
        page_count=5,
        metadata=metadata,
    )

    assert document.metadata == metadata


def test_document_chunk_references_source_document() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    chunk = DocumentChunk(
        document_id=document_id,
        content="First chunk.",
        chunk_index=0,
        page_number=1,
    )

    assert chunk.document_id == document_id
    assert chunk.chunk_index == 0
    assert chunk.page_number == 1


def test_document_chunk_generates_unique_id() -> None:
    document_id = UUID("12345678-1234-5678-1234-567812345678")

    first = DocumentChunk(
        document_id=document_id,
        content="First chunk.",
        chunk_index=0,
    )
    second = DocumentChunk(
        document_id=document_id,
        content="Second chunk.",
        chunk_index=1,
    )

    assert first.chunk_id != second.chunk_id


def test_document_metadata_defaults_to_empty_dictionary() -> None:
    document = Document(
        filename="regulation.pdf",
        content="Content",
        page_count=1,
    )

    assert document.metadata == {}
