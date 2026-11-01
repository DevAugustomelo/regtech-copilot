"""Tests for token-based text chunking."""

import tiktoken

from regtech_copilot.core.config import Settings
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.schemas.documents import (
    Document,
    DocumentPage,
)


def make_settings(
    chunk_size: int = 10,
    chunk_overlap: int = 2,
) -> Settings:
    """Create settings for chunking tests."""

    return Settings(
        llm_model="test-model",
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
    )


def test_chunk_respects_maximum_token_size() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=1,
        pages=(
            DocumentPage(
                page_number=1,
                content=" ".join(["regulatory"] * 30),
            ),
        ),
    )

    chunker = TextChunker(settings=make_settings())

    chunks = chunker.chunk(document)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk.content)) <= 10 for chunk in chunks)


def test_chunk_preserves_document_id_and_page_number() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=2,
        pages=(
            DocumentPage(page_number=1, content="First page content."),
            DocumentPage(page_number=2, content="Second page content."),
        ),
    )

    chunker = TextChunker(settings=make_settings())
    chunks = chunker.chunk(document)

    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_chunk_preserves_metadata() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=1,
        metadata={"source": "official.pdf"},
        pages=(DocumentPage(page_number=1, content="Regulatory content."),),
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert chunks
    assert chunks[0].metadata["source"] == "official.pdf"
    assert chunks[0].metadata["filename"] == "regulation.pdf"


def test_chunk_ignores_empty_pages() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=2,
        pages=(
            DocumentPage(page_number=1, content=""),
            DocumentPage(page_number=2, content="Valid content."),
        ),
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_uses_document_content_when_pages_are_unavailable() -> None:
    document = Document(
        filename="regulation.pdf",
        content="Fallback document content.",
        page_count=1,
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1


def test_chunk_returns_empty_list_for_empty_document() -> None:
    document = Document(
        filename="empty.pdf",
        content="",
        page_count=0,
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert chunks == []


def test_chunk_overlap_reuses_text_between_chunks() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=1,
        pages=(
            DocumentPage(
                page_number=1,
                content=" ".join(["regulatory"] * 30),
            ),
        ),
    )

    chunker = TextChunker(settings=make_settings())
    chunks = chunker.chunk(document)

    assert len(chunks) > 1

    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert (
            previous.content.endswith(current.content.split(" ", 1)[0])
            or current.content in previous.content
        )


def test_chunk_indices_are_sequential() -> None:
    document = Document(
        filename="regulation.pdf",
        content="",
        page_count=2,
        pages=(
            DocumentPage(page_number=1, content="First page content."),
            DocumentPage(page_number=2, content="Second page content."),
        ),
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
