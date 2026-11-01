"""Tests for vector store contracts."""

from uuid import uuid4

from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.base import VectorSearchResult


def test_vector_search_result_preserves_chunk_and_distance() -> None:
    document_id = uuid4()

    chunk = DocumentChunk(
        document_id=document_id,
        content="A organização deve manter registros de conformidade.",
        chunk_index=0,
        page_number=3,
    )

    result = VectorSearchResult(
        chunk=chunk,
        distance=0.15,
    )

    assert result.chunk == chunk
    assert result.chunk.document_id == document_id
    assert result.chunk.page_number == 3
    assert result.distance == 0.15
