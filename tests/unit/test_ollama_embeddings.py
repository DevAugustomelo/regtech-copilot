"""Tests for the Ollama embedding provider."""

import asyncio

import httpx
import pytest

from regtech_copilot.embeddings.base import EmbeddingResponse
from regtech_copilot.embeddings.ollama import (
    OllamaEmbeddingClient,
    OllamaEmbeddingConnectionError,
    OllamaEmbeddingError,
    OllamaEmbeddingResponseError,
)


def test_embed_returns_normalized_response() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/embed"

            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        [0.1, 0.2, 0.3],
                        [0.4, 0.5, 0.6],
                    ],
                },
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-embedding-model",
            expected_dimensions=3,
            http_client=http_client,
        )

        try:
            result = await client.embed(
                ["Texto um", "Texto dois"],
            )

            assert result == EmbeddingResponse(
                embeddings=[
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ],
                model="test-embedding-model",
                dimensions=3,
            )
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_empty_input() -> None:
    async def run_test() -> None:
        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
        )

        try:
            with pytest.raises(ValueError, match="At least one text"):
                await client.embed([])
        finally:
            await client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_mismatched_vector_count() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2]]},
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaEmbeddingResponseError):
                await client.embed(["Texto um", "Texto dois"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_inconsistent_dimensions() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "embeddings": [
                        [0.1, 0.2],
                        [0.3, 0.4, 0.5],
                    ],
                },
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(
                OllamaEmbeddingResponseError,
                match="invalid embedding response",
            ):
                await client.embed(["Texto um", "Texto dois"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_unexpected_dimensions() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, 0.2, 0.3]]},
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            expected_dimensions=768,
            http_client=http_client,
        )

        try:
            with pytest.raises(
                OllamaEmbeddingResponseError,
                match="Expected 768 dimensions",
            ):
                await client.embed(["Texto"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_raises_connection_error() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(
                "Connection refused",
                request=request,
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaEmbeddingConnectionError):
                await client.embed(["Texto"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_non_numeric_values() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"embeddings": [[0.1, "invalid"]]},
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaEmbeddingResponseError):
                await client.embed(["Texto"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


def test_embed_rejects_http_errors() -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaEmbeddingError, match="HTTP status 500"):
                await client.embed(["Texto"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())


@pytest.mark.parametrize(
    "invalid_value",
    ["NaN", "Infinity", "-Infinity"],
)
def test_embed_rejects_non_finite_values(invalid_value: str) -> None:
    async def run_test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            response_body = f'{{"embeddings": [[{invalid_value}]]}}'

            return httpx.Response(
                200,
                content=response_body,
                headers={"Content-Type": "application/json"},
            )

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
        )

        client = OllamaEmbeddingClient(
            base_url="http://localhost:11434",
            model="test-model",
            http_client=http_client,
        )

        try:
            with pytest.raises(OllamaEmbeddingResponseError):
                await client.embed(["Texto"])
        finally:
            await http_client.aclose()

    asyncio.run(run_test())
