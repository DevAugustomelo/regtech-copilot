"""Orchestrate retrieval, prompt construction, and LLM generation."""

from dataclasses import dataclass

from regtech_copilot.generation.prompt_builder import PromptBuilder
from regtech_copilot.llm.base import (
    GenerationOptions,
    LLMProvider,
)
from regtech_copilot.retrieval.service import (
    RetrievalResult,
    RetrievalService,
)


class GenerationServiceError(Exception):
    """Base exception for generation service errors."""


class GenerationResponseError(GenerationServiceError):
    """Raised when the LLM returns an empty response."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Complete response with its supporting sources and usage metadata."""

    answer: str
    sources: tuple[RetrievalResult, ...]
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class GenerationService:
    """Coordinate the complete RAG question-answering pipeline."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        generation_options: GenerationOptions,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.generation_options = generation_options

    async def generate(
        self,
        query: str,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Generate an answer grounded in retrieved document excerpts."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        retrieved_results = await self.retrieval_service.retrieve(
            query,
            top_k=top_k,
        )

        messages = self.prompt_builder.build(
            query,
            retrieved_results,
        )

        response = await self.llm_provider.generate(
            messages.user_prompt,
            options=self.generation_options,
            system_prompt=messages.system_prompt,
        )

        if not response.content.strip():
            raise GenerationResponseError("The LLM returned an empty response.")

        return GenerationResult(
            answer=response.content,
            sources=messages.sources,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
