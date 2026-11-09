"""Tests for the NVIDIA NIM provider."""

import httpx
import pytest

from regtech_copilot.llm.base import GenerationOptions
from regtech_copilot.llm.nvidia import (
    NvidiaAuthenticationError,
    NvidiaClient,
    NvidiaConnectionError,
    NvidiaProviderError,
    NvidiaResponseError,
)


@pytest.fixture
def options() -> GenerationOptions:
    """Provide generation options for tests."""

    return GenerationOptions(
        temperature=0.1,
        top_p=0.9,
        max_tokens=500,
    )


def create_client(
    handler: httpx.MockTransport,
) -> tuple[NvidiaClient, httpx.AsyncClient]:
    """Create a NVIDIA client with a mocked HTTP transport."""

    http_client = httpx.AsyncClient(transport=handler)

    client = NvidiaClient(
        api_key="test-api-key",
        base_url="https://integrate.api.nvidia.com/v1",
        model="test-model",
        http_client=http_client,
    )

    return client, http_client


@pytest.mark.asyncio
async def test_generate_returns_normalized_response(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-api-key"

        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Resposta baseada nos documentos.",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
        )

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        result = await client.generate(
            "Explique o artigo 6 da LGPD.",
            options=options,
            system_prompt="Responda somente com base nas fontes.",
        )

        assert result.content == "Resposta baseada nos documentos."
        assert result.provider == "nvidia"
        assert result.model == "test-model"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_sends_correct_payload(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        payload = json.loads(request.content)

        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.1
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 500
        assert payload["chat_template_kwargs"] == {
            "enable_thinking": False,
        }
        assert payload["stream"] is False
        assert payload["messages"] == [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User prompt"},
        ]

        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "OK"}}]},
        )

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        await client.generate(
            "User prompt",
            options=options,
            system_prompt="System prompt",
        )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt(
    options: GenerationOptions,
) -> None:
    client = NvidiaClient(
        api_key="test-api-key",
        base_url="https://integrate.api.nvidia.com/v1",
        model="test-model",
    )

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        await client.generate(" ", options=options)

    await client.aclose()


@pytest.mark.asyncio
async def test_generate_handles_authentication_error(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401)

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(NvidiaAuthenticationError):
            await client.generate("Test prompt", options=options)
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_handles_http_error(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(NvidiaProviderError, match="HTTP status 500"):
            await client.generate("Test prompt", options=options)
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_handles_connection_error(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(NvidiaConnectionError):
            await client.generate("Test prompt", options=options)
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_handles_invalid_response(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": []},
        )

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(NvidiaResponseError):
            await client.generate("Test prompt", options=options)
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_handles_token_limit_without_final_content(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                        },
                        "finish_reason": "length",
                    }
                ]
            },
        )

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(
            NvidiaResponseError,
            match="token limit",
        ):
            await client.generate(
                "User prompt",
                options=options,
            )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
async def test_generate_rejects_truncated_response(
    options: GenerationOptions,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Resposta incompleta...",
                        },
                        "finish_reason": "length",
                    }
                ]
            },
        )

    client, http_client = create_client(httpx.MockTransport(handler))

    try:
        with pytest.raises(
            NvidiaResponseError,
            match="token limit",
        ):
            await client.generate("User prompt", options=options)
    finally:
        await http_client.aclose()


def test_client_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError, match="API key cannot be empty"):
        NvidiaClient(
            api_key=" ",
            base_url="https://integrate.api.nvidia.com/v1",
            model="test-model",
        )


def test_client_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model cannot be empty"):
        NvidiaClient(
            api_key="test-api-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model=" ",
        )


def test_client_rejects_invalid_timeout() -> None:
    with pytest.raises(ValueError, match="Timeout must be greater than zero"):
        NvidiaClient(
            api_key="test-api-key",
            base_url="https://integrate.api.nvidia.com/v1",
            model="test-model",
            timeout=0,
        )
