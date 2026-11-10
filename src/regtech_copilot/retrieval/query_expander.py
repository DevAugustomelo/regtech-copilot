"""Expand regulatory queries with deterministic LGPD terminology."""

import re
import unicodedata


class QueryExpander:
    """Enrich queries with related legal terminology."""

    EXPANSIONS = {
        (
            "principios",
            "principio",
        ): "finalidade adequacao necessidade transparencia seguranca prevencao",
        (
            "dados sensiveis",
            "dado pessoal sensivel",
            "dados pessoais sensiveis",
        ): (
            "origem racial etnica conviccao religiosa opiniao politica "
            "saude vida sexual dado genetico biometrico"
        ),
        (
            "bases legais",
            "base legal",
            "hipoteses legais",
        ): "hipoteses consentimento obrigacao legal contrato",
        (
            "direitos do titular",
            "direitos dos titulares",
        ): (
            "confirmacao acesso correcao eliminacao portabilidade "
            "anonimizacao bloqueio revogacao consentimento"
        ),
        (
            "incidente de seguranca",
            "incidentes de seguranca",
        ): "comunicacao ANPD titulares risco dano relevante",
        (
            "controlador e operador",
            "controlador e do operador",
            "agentes de tratamento",
        ): ("agentes de tratamento responsabilidades instrucoes decisoes tratamento"),
        (
            "papel da ANPD",
            "poderes da ANPD",
            "competencias da ANPD",
            "autoridade nacional",
        ): "competencias fiscalizacao sancoes atribuicoes",
        (
            "criancas e adolescentes",
            "criancas",
            "adolescentes",
        ): "melhor interesse consentimento especifico pais responsaveis",
        (
            "termino",
            "encerramento do tratamento",
            "tratamento encerrado",
            "ser encerrado",
        ): (
            "finalidade alcancada dados desnecessarios impertinentes "
            "fim periodo tratamento comunicacao titular "
            "revogacao consentimento determinacao autoridade nacional"
        ),
        (
            "dados conservados",
            "dados podem ser conservados",
            "conservacao dos dados",
            "conservar os dados",
        ): (
            "eliminacao cumprimento obrigacao legal regulatoria "
            "estudo orgao pesquisa anonimização transferencia terceiro "
            "uso exclusivo controlador acesso terceiro"
        ),
        (
            "transferencia internacional",
            "transferencia de dados para o exterior",
        ): "pais estrangeiro organismo internacional garantias nivel de protecao",
    }

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for accent-insensitive matching."""

        normalized = unicodedata.normalize("NFKD", text.casefold())
        return "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
        )

    def expand(self, query: str) -> str:
        """Append related terminology without removing original terms."""

        normalized_query = self._normalize(query)
        additions: list[str] = []

        existing_terms = set(re.findall(r"[^\W_]+", normalized_query, flags=re.UNICODE))

        for triggers, expansion in self.EXPANSIONS.items():
            matched = any(
                re.search(
                    rf"(?<!\w){re.escape(self._normalize(trigger))}(?!\w)",
                    normalized_query,
                )
                for trigger in triggers
            )

            if not matched:
                continue

            for term in expansion.split():
                normalized_term = self._normalize(term)

                if normalized_term not in existing_terms:
                    additions.append(term)
                    existing_terms.add(normalized_term)

        if not additions:
            return query

        return f"{query} {' '.join(additions)}"


    def expand_variants(self, query: str) -> list[str]:
        """Return focused expansions when multiple concepts are detected."""

        normalized_query = self._normalize(query)
        matched_expansions: list[str] = []

        for triggers, expansion in self.EXPANSIONS.items():
            matched = any(
                re.search(
                    rf"(?<!\w){re.escape(self._normalize(trigger))}(?!\w)",
                    normalized_query,
                )
                for trigger in triggers
            )

            if matched:
                matched_expansions.append(expansion)

        if len(matched_expansions) <= 1:
            return []

        return matched_expansions
