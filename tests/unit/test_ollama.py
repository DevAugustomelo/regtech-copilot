"""Tests for the Ollama provider."""

import asyncio

import httpx
import pytest

from regtech_copilot.llm.base import GenerationOptions, LLMResponse
from regtech_copilot.llm.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaProviderError,
    OllamaResponseError,
)


def test_generate_returns_normalized_response() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat"

            payload = request.read().decode()
            assert '"model":"test-model"' in payload
            assert '"stream":false' in payload

            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "Resposta de teste.",
                    },
                    "prompt_eval_count": 20,
                    "eval_count": 10,
                },
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            result = await client.generate(
                "Explique RAG.",
                options=GenerationOptions(
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=500,
                ),
                system_prompt="Você é um assistente técnico.",
            )

            assert result == LLMResponse(
                content="Resposta de teste.",
                provider="ollama",
                model="test-model",
                input_tokens=20,
                output_tokens=10,
            )
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_generate_rejects_empty_prompt() -> None:
    async def run_test() -> None:
        client = OllamaClient(
            base_url="http://localhost:11434",
            model="test-model",
        )

        try:
            with pytest.raises(ValueError, match="Prompt cannot be empty"):
                await client.generate(
                    "   ",
                    options=GenerationOptions(0.1, 0.9, 500),
                )
        finally:
            await client.aclose()

    asyncio.run(run_test())


def test_generate_raises_provider_error_for_http_failure() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaProviderError, match="HTTP status 500"):
                await client.generate(
                    "Hello",
                    options=GenerationOptions(0.1, 0.9, 500),
                )
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_generate_raises_response_error_for_invalid_data() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"unexpected": "response"},
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaResponseError):
                await client.generate(
                    "Hello",
                    options=GenerationOptions(0.1, 0.9, 500),
                )
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_generate_raises_connection_error() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(
                "Connection refused",
                request=request,
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaConnectionError):
                await client.generate(
                    "Hello",
                    options=GenerationOptions(0.1, 0.9, 500),
                )
        finally:
            await http_client.aclose()

    asyncio.run(run_test())
