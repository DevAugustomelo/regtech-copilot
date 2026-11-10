from regtech_copilot.retrieval.query_expander import QueryExpander


def test_expand_adds_related_terms_to_lgpd_query() -> None:
    query = "Quais são os princípios que orientam o tratamento de dados pessoais?"

    expanded = QueryExpander().expand(query)

    assert expanded.startswith(query)
    assert "finalidade" in expanded
    assert "transparencia" in expanded


def test_expand_preserves_unrelated_query() -> None:
    query = "Quais cláusulas são comuns em contratos comerciais?"

    assert QueryExpander().expand(query) == query


def test_expand_variants_returns_separate_expansions_for_multiple_concepts() -> None:
    query = (
        "Em quais situações o tratamento de dados pessoais deve ser encerrado "
        "e quando os dados podem ser conservados?"
    )

    variants = QueryExpander().expand_variants(query)

    assert len(variants) == 2
    assert "finalidade alcancada" in variants[0]
    assert "cumprimento obrigacao legal" in variants[1]


def test_expand_variants_returns_empty_for_single_concept() -> None:
    query = "Quais princípios orientam o tratamento de dados pessoais?"

    assert QueryExpander().expand_variants(query) == []
