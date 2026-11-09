"""Tests for the generation service."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from regtech_copilot.generation.prompt_builder import PromptBuilder
from regtech_copilot.generation.service import (
    GenerationResponseError,
    GenerationService,
)
from regtech_copilot.llm.base import GenerationOptions, LLMResponse
from regtech_copilot.retrieval.service import RetrievalResult, RetrievalService
from regtech_copilot.schemas.documents import DocumentChunk


@pytest.fixture
def search_results() -> list[RetrievalResult]:
    """Provide sample retrieved chunks."""

    return [
        RetrievalResult(
            chunk=DocumentChunk(
                document_id=uuid4(),
                content="Organizations must protect personal data.",
                chunk_index=0,
                page_number=3,
                metadata={"filename": "regulation.pdf"},
            ),
            rrf_score=0.05,
        )
    ]


@pytest.fixture
def retrieval_service(
    search_results: list[RetrievalResult],
) -> AsyncMock:
    """Provide a mocked retrieval service."""

    service = AsyncMock(spec=RetrievalService)
    service.retrieve.return_value = search_results
    return service


@pytest.fixture
def llm_provider() -> AsyncMock:
    """Provide a mocked LLM provider."""

    provider = AsyncMock()
    provider.generate.return_value = LLMResponse(
        content="Organizations must protect personal data [F1].",
        provider="ollama",
        model="llama3.2",
        input_tokens=500,
        output_tokens=30,
    )
    return provider


@pytest.fixture
def generation_options() -> GenerationOptions:
    """Provide generation parameters."""

    return GenerationOptions(
        temperature=0.1,
        top_p=0.9,
        max_tokens=1200,
    )


@pytest.fixture
def service(
    retrieval_service: AsyncMock,
    llm_provider: AsyncMock,
    generation_options: GenerationOptions,
) -> GenerationService:
    """Create a generation service with mocked dependencies."""

    return GenerationService(
        retrieval_service=retrieval_service,
        prompt_builder=PromptBuilder(),
        llm_provider=llm_provider,
        generation_options=generation_options,
    )


@pytest.mark.asyncio
async def test_generate_returns_grounded_response(
    service: GenerationService,
    retrieval_service: AsyncMock,
    llm_provider: AsyncMock,
    generation_options: GenerationOptions,
    search_results: list[RetrievalResult],
) -> None:
    """Test the complete RAG generation pipeline."""

    query = "What are the data protection requirements?"

    result = await service.generate(query)

    retrieval_service.retrieve.assert_awaited_once_with(
        query,
        top_k=None,
    )

    llm_provider.generate.assert_awaited_once()

    call_kwargs = llm_provider.generate.await_args.kwargs

    assert call_kwargs["options"] == generation_options
    assert "RegTech Copilot" in call_kwargs["system_prompt"]
    assert query in llm_provider.generate.await_args.args[0]

    assert result.answer == "Organizations must protect personal data [F1]."
    assert result.sources == tuple(search_results)
    assert result.provider == "ollama"
    assert result.model == "llama3.2"
    assert result.input_tokens == 500
    assert result.output_tokens == 30


@pytest.mark.asyncio
async def test_generate_supports_custom_top_k(
    service: GenerationService,
    retrieval_service: AsyncMock,
) -> None:
    """Test custom retrieval result limits."""

    await service.generate("What are the requirements?", top_k=3)

    retrieval_service.retrieve.assert_awaited_once_with(
        "What are the requirements?",
        top_k=3,
    )


@pytest.mark.asyncio
async def test_generate_rejects_empty_query(
    service: GenerationService,
    retrieval_service: AsyncMock,
    llm_provider: AsyncMock,
) -> None:
    """Test that empty questions are rejected."""

    with pytest.raises(ValueError, match="Query cannot be empty"):
        await service.generate("   ")

    retrieval_service.retrieve.assert_not_awaited()
    llm_provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_rejects_empty_llm_response(
    service: GenerationService,
    llm_provider: AsyncMock,
) -> None:
    """Test that empty LLM responses are rejected."""

    llm_provider.generate.return_value = LLMResponse(
        content="   ",
        provider="ollama",
        model="llama3.2",
    )

    with pytest.raises(
        GenerationResponseError,
        match="empty response",
    ):
        await service.generate("What are the requirements?")


@pytest.mark.asyncio
async def test_generate_handles_missing_sources(
    service: GenerationService,
    retrieval_service: AsyncMock,
    llm_provider: AsyncMock,
) -> None:
    """Test generation when retrieval returns no results."""

    retrieval_service.retrieve.return_value = []

    result = await service.generate("Unknown regulatory subject")

    assert result.sources == ()

    prompt = llm_provider.generate.await_args.args[0]

    assert "[]" in prompt
    assert "Unknown regulatory subject" in prompt
