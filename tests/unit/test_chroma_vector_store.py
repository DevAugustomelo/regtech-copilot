"""Tests for the ChromaDB vector store."""

from uuid import uuid4

import pytest

from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.chroma import (
    ChromaVectorStore,
)


@pytest.fixture
def vector_store(tmp_path: object) -> ChromaVectorStore:
    """Create an isolated ChromaDB instance for testing."""

    return ChromaVectorStore(
        persist_directory=str(tmp_path),
        collection_name="test_documents",
    )


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """Create a sample document chunk."""

    return DocumentChunk(
        document_id=uuid4(),
        content="The organization must maintain compliance records.",
        chunk_index=0,
        page_number=3,
        metadata={
            "filename": "regulation.pdf",
            "source": "official_document",
        },
    )


def test_store_initializes_collection(
    vector_store: ChromaVectorStore,
) -> None:
    assert vector_store.count() == 0


def test_add_stores_chunks_and_embeddings(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    vector_store.add(
        chunks=[sample_chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    assert vector_store.count() == 1


def test_search_returns_most_similar_chunks(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    second_chunk = DocumentChunk(
        document_id=sample_chunk.document_id,
        content="Unrelated financial information.",
        chunk_index=1,
        page_number=4,
    )

    vector_store.add(
        chunks=[sample_chunk, second_chunk],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == sample_chunk.chunk_id
    assert results[0].chunk.content == sample_chunk.content
    assert results[0].distance == pytest.approx(0.0)


def test_search_preserves_document_metadata(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    vector_store.add(
        chunks=[sample_chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    results = vector_store.search([1.0, 0.0, 0.0])

    retrieved_chunk = results[0].chunk

    assert retrieved_chunk.document_id == sample_chunk.document_id
    assert retrieved_chunk.chunk_id == sample_chunk.chunk_id
    assert retrieved_chunk.chunk_index == sample_chunk.chunk_index
    assert retrieved_chunk.page_number == sample_chunk.page_number
    assert retrieved_chunk.metadata == sample_chunk.metadata


def test_search_returns_empty_list_for_empty_collection(
    vector_store: ChromaVectorStore,
) -> None:
    results = vector_store.search([1.0, 0.0, 0.0])

    assert results == []


def test_delete_removes_all_chunks_from_document(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    second_chunk = DocumentChunk(
        document_id=sample_chunk.document_id,
        content="Additional compliance information.",
        chunk_index=1,
        page_number=4,
    )

    vector_store.add(
        chunks=[sample_chunk, second_chunk],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    assert vector_store.count() == 2

    vector_store.delete(sample_chunk.document_id)

    assert vector_store.count() == 0


def test_delete_preserves_other_documents(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    other_chunk = DocumentChunk(
        document_id=uuid4(),
        content="Another regulatory document.",
        chunk_index=0,
        page_number=1,
    )

    vector_store.add(
        chunks=[sample_chunk, other_chunk],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    vector_store.delete(sample_chunk.document_id)

    assert vector_store.count() == 1

    results = vector_store.search([0.0, 1.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk.document_id == other_chunk.document_id


def test_persistence_survives_reinitialization(
    tmp_path: object,
    sample_chunk: DocumentChunk,
) -> None:
    path = str(tmp_path)

    first_store = ChromaVectorStore(
        persist_directory=path,
        collection_name="persistence_test",
    )

    first_store.add(
        chunks=[sample_chunk],
        embeddings=[[1.0, 0.0, 0.0]],
    )

    second_store = ChromaVectorStore(
        persist_directory=path,
        collection_name="persistence_test",
    )

    assert second_store.count() == 1

    results = second_store.search([1.0, 0.0, 0.0])

    assert len(results) == 1
    assert results[0].chunk.chunk_id == sample_chunk.chunk_id


def test_add_rejects_different_number_of_chunks_and_embeddings(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    with pytest.raises(ValueError, match="must match"):
        vector_store.add(
            chunks=[sample_chunk],
            embeddings=[],
        )


def test_add_rejects_embeddings_with_different_dimensions(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    with pytest.raises(ValueError, match="same dimensions"):
        vector_store.add(
            chunks=[sample_chunk, sample_chunk],
            embeddings=[
                [1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )


def test_add_rejects_non_finite_embedding_values(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        vector_store.add(
            chunks=[sample_chunk],
            embeddings=[[1.0, float("nan")]],
        )


def test_search_rejects_invalid_top_k(
    vector_store: ChromaVectorStore,
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        vector_store.search(
            query_embedding=[1.0, 0.0],
            top_k=0,
        )


def test_search_rejects_non_finite_query_embedding(
    vector_store: ChromaVectorStore,
) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        vector_store.search(
            query_embedding=[1.0, float("inf")],
        )


def test_search_by_article_returns_only_matching_article(
    vector_store: ChromaVectorStore,
    sample_chunk: DocumentChunk,
) -> None:
    first_chunk = DocumentChunk(
        document_id=sample_chunk.document_id,
        content="Art. 6º - Princípios I a VI.",
        chunk_index=0,
        page_number=12,
        metadata={"article_reference": "6"},
    )

    continuation_chunk = DocumentChunk(
        document_id=sample_chunk.document_id,
        content="Art. 6º - Princípios VII a X.",
        chunk_index=1,
        page_number=13,
        metadata={
            "article_reference": "6",
            "article_continuation": True,
        },
    )

    other_article = DocumentChunk(
        document_id=sample_chunk.document_id,
        content="Art. 7º - Outra disposição.",
        chunk_index=2,
        page_number=14,
        metadata={"article_reference": "7"},
    )

    vector_store.add(
        chunks=[first_chunk, continuation_chunk, other_article],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
    )

    results = vector_store.search_by_article(
        query_embedding=[1.0, 0.0, 0.0],
        document_id=sample_chunk.document_id,
        article_reference="6",
    )

    assert len(results) == 2
    assert {result.chunk.chunk_id for result in results} == {
        first_chunk.chunk_id,
        continuation_chunk.chunk_id,
    }
    assert all(result.distance >= 0 for result in results)
