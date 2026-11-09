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
