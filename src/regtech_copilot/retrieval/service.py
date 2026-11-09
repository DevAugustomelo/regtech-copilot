"""Retrieve relevant document chunks using hybrid search and RRF."""

import asyncio
from dataclasses import dataclass
from uuid import UUID

from regtech_copilot.embeddings.base import EmbeddingProvider
from regtech_copilot.retrieval.lexical_store import LexicalStore
from regtech_copilot.retrieval.query_expander import QueryExpander
from regtech_copilot.schemas.documents import DocumentChunk
from regtech_copilot.vector_store.base import VectorStore


class RetrievalServiceError(Exception):
    """Base exception for retrieval errors."""


class RetrievalEmbeddingError(RetrievalServiceError):
    """Raised when query embedding generation returns invalid results."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A chunk retrieved and ranked through hybrid search."""

    chunk: DocumentChunk
    rrf_score: float
    vector_distance: float | None = None
    lexical_score: float | None = None


class RetrievalService:
    """Generate query embeddings and retrieve chunks using hybrid search."""

    CANDIDATE_MULTIPLIER = 3
    RRF_CONSTANT = 60
    LEXICAL_WEIGHT = 2.0

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        top_k: int = 5,
        query_expander: QueryExpander | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self.top_k = top_k
        self.query_expander = query_expander

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve and fuse semantic and lexical search results."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        result_limit = self.top_k if top_k is None else top_k

        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        search_query = (
            self.query_expander.expand(query)
            if self.query_expander is not None
            else query
        )

        embedding_response = await self.embedding_provider.embed(
            [search_query],
            input_type="query",
        )

        if len(embedding_response.embeddings) != 1:
            raise RetrievalEmbeddingError(
                "Expected exactly one embedding for the query."
            )

        query_embedding = embedding_response.embeddings[0]

        if (
            embedding_response.dimensions <= 0
            or len(query_embedding) != embedding_response.dimensions
        ):
            raise RetrievalEmbeddingError("The query embedding has invalid dimensions.")

        candidate_limit = result_limit * self.CANDIDATE_MULTIPLIER

        vector_results, lexical_results = await asyncio.gather(
            asyncio.to_thread(
                self.vector_store.search,
                query_embedding,
                candidate_limit,
            ),
            asyncio.to_thread(
                self.lexical_store.search,
                search_query,
                candidate_limit,
            ),
        )

        scores: dict[object, float] = {}
        chunks: dict[object, DocumentChunk] = {}
        vector_distances: dict[object, float] = {}
        lexical_scores: dict[object, float] = {}

        for rank, vector_result in enumerate(vector_results, start=1):
            chunk_id = vector_result.chunk.chunk_id

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (self.RRF_CONSTANT + rank)
            )
            chunks[chunk_id] = vector_result.chunk
            vector_distances[chunk_id] = vector_result.distance

        for rank, lexical_result in enumerate(lexical_results, start=1):
            chunk_id = lexical_result.chunk.chunk_id

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                self.LEXICAL_WEIGHT / (self.RRF_CONSTANT + rank)
            )
            chunks.setdefault(chunk_id, lexical_result.chunk)
            lexical_scores[chunk_id] = lexical_result.score

        ranked_results = [
            RetrievalResult(
                chunk=chunks[chunk_id],
                rrf_score=score,
                vector_distance=vector_distances.get(chunk_id),
                lexical_score=lexical_scores.get(chunk_id),
            )
            for chunk_id, score in scores.items()
        ]

        ranked_results = sorted(
            ranked_results,
            key=lambda result: result.rrf_score,
            reverse=True,
        )[:result_limit]

        # Expand retrieved articles with their neighboring chunks.
        article_keys: list[tuple[UUID, str]] = []
        for result in ranked_results:
            article_reference = result.chunk.metadata.get("article_reference")

            if not isinstance(article_reference, str) or not article_reference.strip():
                continue

            article_key = (result.chunk.document_id, article_reference)

            if article_key not in article_keys:
                article_keys.append(article_key)

        if not article_keys:
            return ranked_results

        article_results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    self.vector_store.search_by_article,
                    query_embedding,
                    document_id,
                    article_reference,
                )
                for document_id, article_reference in article_keys
            )
        )

        related_chunks = dict(zip(article_keys, article_results, strict=True))
        included_chunk_ids = {result.chunk.chunk_id for result in ranked_results}
        expanded_articles: set[tuple[UUID, str]] = set()
        expanded_results: list[RetrievalResult] = []

        for result in ranked_results:
            expanded_results.append(result)

            article_reference = result.chunk.metadata.get("article_reference")

            if not isinstance(article_reference, str) or not article_reference.strip():
                continue

            article_key = (result.chunk.document_id, article_reference)

            if article_key in expanded_articles:
                continue

            expanded_articles.add(article_key)
            additional_count = 0

            for related_result in related_chunks.get(article_key, []):
                chunk = related_result.chunk

                if chunk.chunk_id in included_chunk_ids:
                    continue

                expanded_results.append(
                    RetrievalResult(
                        chunk=chunk,
                        rrf_score=0.0,
                        vector_distance=related_result.distance,
                    )
                )
                included_chunk_ids.add(chunk.chunk_id)
                additional_count += 1

                if additional_count >= 2:
                    break

        return expanded_results
