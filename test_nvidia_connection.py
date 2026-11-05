
import asyncio

from regtech_copilot.core.config import get_settings
from regtech_copilot.llm.base import GenerationOptions
from regtech_copilot.llm.factory import LLMFactory


async def main() -> None:
    settings = get_settings()

    if settings.nvidia_api_key is None:
        raise RuntimeError("NVIDIA_API_KEY não configurada.")

    if not settings.nvidia_api_key.get_secret_value().strip():
        raise RuntimeError("NVIDIA_API_KEY está vazia.")

    provider = LLMFactory.create(settings)

    try:
        response = await provider.generate(
            prompt=(
                "Explique em português, em até três frases, "
                "o que é a LGPD e qual sua importância."
            ),
            options=GenerationOptions(
                temperature=0.1,
                top_p=0.9,
                max_tokens=1200,
            ),
            system_prompt=(
                "Você é um assistente especializado em proteção "
                "de dados pessoais e legislação brasileira."
            ),
        )

        print("\n========== TESTE NVIDIA ==========")
        print(f"Provider: {response.provider}")
        print(f"Model: {response.model}")
        print(f"Input tokens: {response.input_tokens}")
        print(f"Output tokens: {response.output_tokens}")
        print(f"\nResposta:\n{response.content}")
        print("==================================")

    finally:
        await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
