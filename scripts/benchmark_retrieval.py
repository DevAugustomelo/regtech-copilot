"""Compare retrieval quality with and without FlashRank."""

import asyncio
import time
from dataclasses import dataclass
from statistics import mean

from regtech_copilot.core.config import get_settings
from regtech_copilot.embeddings.factory import EmbeddingFactory
from regtech_copilot.retrieval.flashrank_reranker import FlashRankReranker
from regtech_copilot.retrieval.lexical_store import SQLiteLexicalStore
from regtech_copilot.retrieval.query_expander import QueryExpander
from regtech_copilot.retrieval.service import RetrievalResult, RetrievalService
from regtech_copilot.vector_store.chroma import ChromaVectorStore

TOP_K = 5

BENCHMARK_CASES = [
    (
        "Princípios da LGPD",
        (
            "Quais são os princípios que devem ser observados no tratamento "
            "de dados pessoais segundo a LGPD?"
        ),
        {"6"},
    ),
    (
        "Dados sensíveis",
        (
            "O que são dados pessoais sensíveis e quais categorias são "
            "consideradas sensíveis pela LGPD?"
        ),
        {"5", "11"},
    ),
    (
        "Bases legais",
        "Quais são as hipóteses legais que autorizam o tratamento de dados pessoais?",
        {"7"},
    ),
    (
        "Direitos do titular",
        (
            "Quais direitos o titular de dados pessoais pode exercer "
            "perante o controlador?"
        ),
        {"17", "18", "19"},
    ),
    (
        "Incidente de segurança",
        (
            "O que o controlador deve fazer em caso de incidente de segurança "
            "que possa acarretar risco ou dano relevante aos titulares?"
        ),
        {"48"},
    ),
    (
        "Controlador e operador",
        (
            "Quais são as diferenças entre controlador e operador e quais "
            "responsabilidades cabem a cada um?"
        ),
        {"37", "39", "42", "46"},
    ),
    (
        "ANPD",
        (
            "Quais são as competências e atribuições da Autoridade Nacional "
            "de Proteção de Dados (ANPD)?"
        ),
        {"55-A", "55-J"},
    ),
    (
        "Crianças e adolescentes",
        (
            "Quais regras especiais a LGPD estabelece para o tratamento "
            "de dados pessoais de crianças e adolescentes?"
        ),
        {"14"},
    ),
    (
        "Término e conservação",
        (
            "Em quais situações o tratamento de dados pessoais deve ser "
            "encerrado e quando os dados podem ser conservados?"
        ),
        {"15", "16"},
    ),
    (
        "Transferência internacional",
        (
            "Quais requisitos a LGPD estabelece para a transferência "
            "internacional de dados pessoais?"
        ),
        {"33", "34", "35", "36"},
    ),
]


@dataclass
class BenchmarkMetrics:
    """Store retrieval quality and performance metrics."""

    precision: float
    recall: float
    mrr: float
    latency_ms: float


def evaluate(
    results: list[RetrievalResult],
    relevant_articles: set[str],
    latency_ms: float,
) -> BenchmarkMetrics:
    """Calculate retrieval metrics using the first TOP_K scored results."""

    top_results = [result for result in results if result.rrf_score > 0][:TOP_K]

    article_references = [
        str(result.chunk.metadata["article_reference"])
        if result.chunk.metadata.get("article_reference") is not None
        else None
        for result in top_results
    ]

    relevant_hits = sum(
        article in relevant_articles
        for article in article_references
        if article is not None
    )

    retrieved_relevant_articles = {
        article
        for article in article_references
        if article is not None and article in relevant_articles
    }

    precision = relevant_hits / TOP_K
    recall = len(retrieved_relevant_articles) / len(relevant_articles)

    reciprocal_rank = 0.0

    for rank, article in enumerate(article_references, start=1):
        if article in relevant_articles:
            reciprocal_rank = 1 / rank
            break

    return BenchmarkMetrics(
        precision=precision,
        recall=recall,
        mrr=reciprocal_rank,
        latency_ms=latency_ms,
    )


async def measure(
    service: RetrievalService,
    query: str,
    relevant_articles: set[str],
) -> BenchmarkMetrics:
    """Measure retrieval quality and execution time."""

    start = time.perf_counter()
    results = await service.retrieve(query, top_k=TOP_K)
    latency_ms = (time.perf_counter() - start) * 1000

    return evaluate(results, relevant_articles, latency_ms)


def print_summary(
    baseline: list[BenchmarkMetrics],
    reranked: list[BenchmarkMetrics],
) -> None:
    """Print per-query and aggregate benchmark results."""

    print("\nRESULTADOS POR PERGUNTA")
    print("-" * 94)
    print(
        f"{'Pergunta':<25}"
        f"{'P@5 base':>11}"
        f"{'P@5 Flash':>12}"
        f"{'R@5 base':>11}"
        f"{'R@5 Flash':>12}"
        f"{'MRR base':>11}"
        f"{'MRR Flash':>12}"
    )

    for case, base, flash in zip(
        BENCHMARK_CASES,
        baseline,
        reranked,
        strict=True,
    ):
        print(
            f"{case[0]:<25}"
            f"{base.precision:>11.3f}"
            f"{flash.precision:>12.3f}"
            f"{base.recall:>11.3f}"
            f"{flash.recall:>12.3f}"
            f"{base.mrr:>11.3f}"
            f"{flash.mrr:>12.3f}"
        )

    print("\nMÉDIAS GERAIS")
    print("-" * 50)
    print(f"{'Métrica':<20}{'Sem reranker':>15}{'Com FlashRank':>15}")

    metrics = [
        ("Precision@5", "precision"),
        ("Recall@5", "recall"),
        ("MRR", "mrr"),
        ("Latência (ms)", "latency_ms"),
    ]

    for label, attribute in metrics:
        base_average = mean(getattr(item, attribute) for item in baseline)
        flash_average = mean(getattr(item, attribute) for item in reranked)

        print(f"{label:<20}{base_average:>15.3f}{flash_average:>15.3f}")

    base_latency = mean(item.latency_ms for item in baseline)
    flash_latency = mean(item.latency_ms for item in reranked)

    print("\nDiferença de latência:")
    print(f"{flash_latency - base_latency:+.2f} ms")


async def main() -> None:
    """Run the retrieval benchmark."""

    settings = get_settings()

    embedding_provider = EmbeddingFactory.create(settings)

    vector_store = ChromaVectorStore(
        persist_directory=settings.vector_store_path,
    )

    lexical_store = SQLiteLexicalStore(
        database_path=(f"{settings.vector_store_path}/lexical_index.sqlite3"),
    )

    common_dependencies = {
        "embedding_provider": embedding_provider,
        "vector_store": vector_store,
        "lexical_store": lexical_store,
        "top_k": TOP_K,
        "query_expander": QueryExpander(),
    }

    baseline_service = RetrievalService(**common_dependencies)

    # Load FlashRank before starting the measurements.
    reranker = FlashRankReranker()

    reranked_service = RetrievalService(
        **common_dependencies,
        reranker=reranker,
    )

    baseline_metrics: list[BenchmarkMetrics] = []
    reranked_metrics: list[BenchmarkMetrics] = []

    try:
        # Warmup reduces the impact of the first request.
        warmup_query = BENCHMARK_CASES[0][1]

        await baseline_service.retrieve(warmup_query, top_k=TOP_K)
        await reranked_service.retrieve(warmup_query, top_k=TOP_K)

        print("Executando benchmark de recuperação...\n")

        for index, (name, query, relevant_articles) in enumerate(
            BENCHMARK_CASES,
            start=1,
        ):
            print(f"[{index}/{len(BENCHMARK_CASES)}] {name}")

            baseline = await measure(
                baseline_service,
                query,
                relevant_articles,
            )

            flashrank = await measure(
                reranked_service,
                query,
                relevant_articles,
            )

            baseline_metrics.append(baseline)
            reranked_metrics.append(flashrank)

        print_summary(baseline_metrics, reranked_metrics)

    finally:
        await embedding_provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())
