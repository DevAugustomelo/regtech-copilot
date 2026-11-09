"""Tests for the prompt builder."""

from uuid import uuid4

import pytest

from regtech_copilot.generation.prompt_builder import PromptBuilder
from regtech_copilot.retrieval.service import RetrievalResult
from regtech_copilot.schemas.documents import DocumentChunk


@pytest.fixture
def search_results() -> list[RetrievalResult]:
    """Provide sample retrieved document chunks."""

    document_id = uuid4()

    return [
        RetrievalResult(
            chunk=DocumentChunk(
                document_id=document_id,
                content="Organizations must protect personal data.",
                chunk_index=0,
                page_number=3,
                metadata={"filename": "regulation.pdf"},
            ),
            rrf_score=0.05,
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                document_id=document_id,
                content="Organizations must report certain incidents.",
                chunk_index=1,
                page_number=4,
                metadata={"filename": "regulation.pdf"},
            ),
            rrf_score=0.12,
        ),
    ]


def test_build_creates_system_and_user_prompts(
    search_results: list[RetrievalResult],
) -> None:
    """Test that both prompts are generated."""

    builder = PromptBuilder()

    result = builder.build(
        "What are the data protection requirements?",
        search_results,
    )

    assert "RegTech Copilot" in result.system_prompt
    assert "QUESTION:" in result.user_prompt
    assert "What are the data protection requirements?" in result.user_prompt


def test_build_includes_retrieved_sources(
    search_results: list[RetrievalResult],
) -> None:
    """Test that retrieved chunks and their references are included."""

    builder = PromptBuilder()

    result = builder.build("What are the requirements?", search_results)

    assert '"reference": "F1"' in result.user_prompt
    assert '"reference": "F2"' in result.user_prompt
    assert "regulation.pdf" in result.user_prompt
    assert "Organizations must protect personal data." in result.user_prompt
    assert '"page_number": 3' in result.user_prompt


def test_build_instructs_model_to_avoid_unsupported_answers(
    search_results: list[RetrievalResult],
) -> None:
    """Test that grounding instructions are included."""

    builder = PromptBuilder()

    result = builder.build("What are the requirements?", search_results)

    assert "Do not invent laws" in result.system_prompt
    assert "evidence is insufficient" in result.system_prompt
    assert "untrusted data" in result.system_prompt


def test_build_handles_empty_search_results() -> None:
    """Test prompt generation without retrieved documents."""

    builder = PromptBuilder()

    result = builder.build("What does the regulation require?", [])

    assert "What does the regulation require?" in result.user_prompt
    assert "RETRIEVED DOCUMENT EXCERPTS:" in result.user_prompt
    assert "[]" in result.user_prompt


def test_build_rejects_empty_query() -> None:
    """Test that empty questions are rejected."""

    builder = PromptBuilder()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        builder.build("   ", [])


def test_builder_rejects_invalid_context_size() -> None:
    """Test that invalid context limits are rejected."""

    with pytest.raises(
        ValueError,
        match="Maximum context size must be greater than zero",
    ):
        PromptBuilder(max_context_chars=0)


def test_build_limits_context_size(
    search_results: list[RetrievalResult],
) -> None:
    """Test that oversized source content is not fully included."""

    builder = PromptBuilder(max_context_chars=300)

    result = builder.build("What are the requirements?", search_results)

    assert len(result.user_prompt) > 0
    assert (
        "Organizations must protect personal data." not in result.user_prompt
        or "Organizations must report certain incidents." not in result.user_prompt
    )
