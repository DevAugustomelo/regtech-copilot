"""Manage application resources during the FastAPI lifecycle."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from regtech_copilot.core.config import get_settings
from regtech_copilot.embeddings.factory import EmbeddingFactory
from regtech_copilot.generation.prompt_builder import PromptBuilder
from regtech_copilot.generation.service import GenerationService
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.ingestion.pdf_loader import PDFLoader
from regtech_copilot.ingestion.service import IngestionService
from regtech_copilot.llm.base import GenerationOptions
from regtech_copilot.llm.factory import LLMFactory
from regtech_copilot.retrieval.lexical_store import SQLiteLexicalStore
from regtech_copilot.retrieval.service import RetrievalService
from regtech_copilot.vector_store.chroma import ChromaVectorStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared services and release resources on shutdown."""

    settings = get_settings()

    Path(settings.documents_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    async with AsyncExitStack() as stack:
        llm_provider = LLMFactory.create(settings)
        stack.push_async_callback(llm_provider.aclose)

        embedding_provider = EmbeddingFactory.create(settings)
        stack.push_async_callback(embedding_provider.aclose)

        vector_store = ChromaVectorStore(
            persist_directory=settings.vector_store_path,
        )

        lexical_store = SQLiteLexicalStore(
            Path(settings.vector_store_path) / "lexical_index.sqlite3"
        )

        stored_chunks = await asyncio.to_thread(vector_store.get_all_chunks)

        await asyncio.to_thread(
            lexical_store.replace_all,
            stored_chunks,
        )

        pdf_loader = PDFLoader()
        chunker = TextChunker(settings=settings)

        ingestion_service = IngestionService(
            pdf_loader=pdf_loader,
            chunker=chunker,
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
            batch_size=settings.embedding_batch_size,
        )

        retrieval_service = RetrievalService(
            embedding_provider=embedding_provider,
            vector_store=vector_store,
            lexical_store=lexical_store,
            top_k=settings.retrieval_top_k,
        )

        generation_service = GenerationService(
            retrieval_service=retrieval_service,
            prompt_builder=PromptBuilder(),
            llm_provider=llm_provider,
            generation_options=GenerationOptions(
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_tokens=settings.llm_max_tokens,
            ),
        )

        app.state.settings = settings
        app.state.llm_provider = llm_provider
        app.state.embedding_provider = embedding_provider
        app.state.vector_store = vector_store
        app.state.lexical_store = lexical_store
        app.state.ingestion_service = ingestion_service
        app.state.retrieval_service = retrieval_service
        app.state.generation_service = generation_service

        yield
