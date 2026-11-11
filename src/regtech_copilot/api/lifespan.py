"""Manage application resources during the FastAPI lifecycle."""

import asyncio
import hashlib
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from regtech_copilot.core.config import get_settings
from regtech_copilot.documents.catalog import DocumentCatalog
from regtech_copilot.embeddings.factory import EmbeddingFactory
from regtech_copilot.generation.prompt_builder import PromptBuilder
from regtech_copilot.generation.service import GenerationService
from regtech_copilot.ingestion.chunker import TextChunker
from regtech_copilot.ingestion.pdf_loader import PDFLoader
from regtech_copilot.ingestion.service import IngestionService
from regtech_copilot.llm.base import GenerationOptions
from regtech_copilot.llm.factory import LLMFactory
from regtech_copilot.retrieval.lexical_store import SQLiteLexicalStore
from regtech_copilot.retrieval.query_expander import QueryExpander
from regtech_copilot.retrieval.service import RetrievalService
from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.chroma import ChromaVectorStore


def _calculate_sha256(path: Path) -> str:
    """Calculate a file SHA-256 without loading it entirely into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _synchronize_document_catalog(
    catalog: DocumentCatalog,
    chunks: list[DocumentChunk],
) -> None:
    """Register indexed documents missing from the persistent catalog."""
    grouped_chunks: dict[str, list[DocumentChunk]] = defaultdict(list)

    for chunk in chunks:
        grouped_chunks[str(chunk.document_id)].append(chunk)

    for document_id, document_chunks in grouped_chunks.items():
        if catalog.get(document_id) is not None:
            continue

        metadata = document_chunks[0].metadata
        source = metadata.get("source")
        filename = metadata.get("filename")

        if not isinstance(source, str) or not source:
            raise RuntimeError(
                f"Indexed document {document_id} has no valid source path."
            )

        source_path = Path(source)

        if not isinstance(filename, str) or not filename:
            filename = source_path.name

        sha256 = _calculate_sha256(source_path) if source_path.is_file() else None

        page_numbers = [
            chunk.page_number
            for chunk in document_chunks
            if chunk.page_number is not None
        ]

        catalog.register(
            document_id=document_id,
            filename=filename,
            source_path=str(source_path),
            sha256=sha256,
            page_count=max(page_numbers) if page_numbers else None,
            chunks_created=len(document_chunks),
        )


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

        document_catalog = DocumentCatalog(
            Path(settings.vector_store_path) / "documents.sqlite3"
        )

        stored_chunks = await asyncio.to_thread(vector_store.get_all_chunks)

        await asyncio.to_thread(
            lexical_store.replace_all,
            stored_chunks,
        )

        await asyncio.to_thread(
            _synchronize_document_catalog,
            document_catalog,
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
            query_expander=QueryExpander(),
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
        app.state.document_catalog = document_catalog
        app.state.document_ingestion_lock = asyncio.Lock()
        app.state.document_ingestion_lock = asyncio.Lock()
        app.state.ingestion_service = ingestion_service
        app.state.retrieval_service = retrieval_service
        app.state.generation_service = generation_service

        yield
