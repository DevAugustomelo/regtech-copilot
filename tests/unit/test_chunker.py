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


def test_chunk_splits_legal_articles_and_preserves_pages() -> None:
    document = Document(
        filename="lgpd.pdf",
        content="",
        page_count=2,
        pages=(
            DocumentPage(
                page_number=12,
                content=(
                    "Definição final do artigo anterior.\n"
                    "Art.\u20046o\u2002 Princípios I, II e VI."
                ),
            ),
            DocumentPage(
                page_number=13,
                content=(
                    "Princípios VII, VIII, IX e X.\n"
                    "Art.\u20047o\u2002 Regras do tratamento."
                ),
            ),
        ),
    )

    chunker = TextChunker(settings=make_settings(200, 20))
    chunks = chunker.chunk(document)

    article_6_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("article_reference") == "6"
    ]
    article_7_chunks = [
        chunk for chunk in chunks if chunk.metadata.get("article_reference") == "7"
    ]

    assert article_6_chunks
    assert article_7_chunks

    assert any(chunk.page_number == 12 for chunk in article_6_chunks)
    assert any(chunk.page_number == 13 for chunk in article_6_chunks)

    continuation = [
        chunk
        for chunk in article_6_chunks
        if chunk.metadata.get("article_continuation") is True
    ]

    assert continuation
    assert all(chunk.page_number == 13 for chunk in continuation)
    assert all("continuação do Art. 6º" in chunk.content for chunk in continuation)

    assert all(chunk.page_number == 13 for chunk in article_7_chunks)


def test_article_pattern_recognizes_ordinal_variations() -> None:
    chunker = TextChunker(settings=make_settings(200, 20))

    content = (
        "Art. 6º Primeiro artigo.\n"
        "Art. 7° Segundo artigo.\n"
        "Art.\u20048o\u2002 Terceiro artigo."
    )

    matches = list(chunker.ARTICLE_PATTERN.finditer(content))

    assert [match.group(1) for match in matches] == ["6", "7", "8"]


def test_chunk_falls_back_to_generic_content_without_articles() -> None:
    document = Document(
        filename="generic.pdf",
        content="",
        page_count=1,
        pages=(
            DocumentPage(
                page_number=1,
                content="Generic regulatory document without article headings.",
            ),
        ),
    )

    chunks = TextChunker(settings=make_settings()).chunk(document)

    assert chunks
    assert all("article_reference" not in chunk.metadata for chunk in chunks)
