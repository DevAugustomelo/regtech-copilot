"""Rerank retrieved document chunks using FlashRank."""

from flashrank import Ranker, RerankRequest  # type: ignore[import-untyped]

from regtech_copilot.retrieval.service import RetrievalResult


class FlashRankReranker:
    """Rerank retrieval candidates according to their relevance to a query."""

    def __init__(self) -> None:
        self._ranker = Ranker(
            model_name="ms-marco-MultiBERT-L-12",
            max_length=256,
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Return retrieval results ordered by relevance."""

        if not results:
            return []

        passages = [
            {
                "id": str(index),
                "text": result.chunk.content,
                "meta": {},
            }
            for index, result in enumerate(results)
        ]

        reranked_passages = self._ranker.rerank(
            RerankRequest(query=query, passages=passages)
        )

        return [results[int(passage["id"])] for passage in reranked_passages]
