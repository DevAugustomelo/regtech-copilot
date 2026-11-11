"""Streamlit user interface for the RegTech Copilot."""

import os
from typing import Any

import streamlit as st

from regtech_copilot.ui.api_client import APIClient, APIClientError

st.set_page_config(
    page_title="RegTech Copilot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_api_client(base_url: str) -> APIClient:
    """Reuse an HTTP client for the configured backend URL."""
    return APIClient(base_url=base_url)


def render_sidebar() -> tuple[APIClient, str]:
    """Render application navigation and backend configuration."""
    st.sidebar.title("⚖️ RegTech Copilot")
    st.sidebar.caption(
        "Assistente regulatório especializado em conformidade com a LGPD,"
        " baseado em RAG e LLMs"
    )

    default_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    api_url = st.sidebar.text_input(
        "Endereço da API",
        value=default_url,
        help="Endereço base do servidor FastAPI.",
    ).strip()

    if not api_url:
        st.sidebar.error("Informe o endereço da API.")
        st.stop()

    client = get_api_client(api_url)

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navegação",
        options=["Consultar legislação", "Gerenciar documentos"],
        index=0,
    )

    st.sidebar.divider()

    try:
        health = client.health_check()
        if health.get("status") == "ok":
            st.sidebar.success("API conectada")
        else:
            st.sidebar.warning("API respondeu com estado inesperado.")
    except APIClientError as exc:
        st.sidebar.error("API indisponível")
        st.sidebar.caption(str(exc))

    st.sidebar.caption("RegTech Copilot · Projeto RAG")

    return client, page


def render_upload_page(client: APIClient) -> None:
    """Render document upload and catalog management."""
    st.title("📄 Gerenciar documentos")
    st.write(
        "Envie documentos regulatórios em PDF, consulte os documentos "
        "indexados e gerencie sua base de conhecimento."
    )

    if message := st.session_state.pop("document_action_message", None):
        st.success(message)

    st.subheader("Enviar documento")

    uploaded_file = st.file_uploader(
        "Selecione um documento PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="O documento será enviado à API para extração, fragmentação "
        "e geração de embeddings.",
    )

    if uploaded_file is not None:
        st.write(f"**Arquivo selecionado:** {uploaded_file.name}")
        st.caption(f"Tamanho: {uploaded_file.size / (1024 * 1024):.2f} MB")

        if st.button(
            "Enviar e processar documento",
            type="primary",
            disabled=uploaded_file.size == 0,
        ):
            try:
                with st.spinner(
                    "Processando documento... Isso pode levar alguns minutos."
                ):
                    result = client.upload_document(
                        filename=uploaded_file.name,
                        file_content=uploaded_file.getvalue(),
                    )

                st.success("Documento processado e indexado com sucesso!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Páginas", result["page_count"])
                col2.metric("Chunks criados", result["chunks_created"])
                col3.metric(
                    "Dimensões dos embeddings",
                    result["embedding_dimensions"],
                )

                st.write(f"**Documento:** {result['filename']}")
                st.write(f"**ID:** `{result['document_id']}`")
                st.write(f"**Modelo de embedding:** `{result['embedding_model']}`")

            except APIClientError as exc:
                st.error("Não foi possível processar o documento.")
                st.warning(str(exc))

            except (KeyError, TypeError) as exc:
                st.error("A API retornou dados incompletos ou inesperados.")
                st.caption(str(exc))

    st.divider()
    st.subheader("Documentos indexados")

    if st.button("🔄 Atualizar lista", key="refresh_documents"):
        st.rerun()

    try:
        with st.spinner("Carregando documentos..."):
            catalog = client.list_documents()

        documents = catalog.get("documents", [])
        total = catalog.get("total", len(documents))

        st.metric("Total de documentos", total)

        if not documents:
            st.info("Nenhum documento indexado.")
            return

        for document in documents:
            document_id = document["document_id"]
            filename = document["filename"]

            with st.expander(f"📄 {filename}"):
                col1, col2 = st.columns(2)
                col1.metric(
                    "Páginas",
                    document.get("page_count") or "—",
                )
                col2.metric(
                    "Chunks",
                    document.get("chunks_created", 0),
                )

                st.caption(f"ID: `{document_id}`")
                st.caption(
                    f"Cadastrado em: {document.get('created_at', 'Não informado')}"
                )

                if document.get("file_exists", False):
                    st.success("Arquivo PDF disponível")
                else:
                    st.warning("Arquivo PDF não encontrado no armazenamento")

                if st.button(
                    "Ver detalhes completos",
                    key=f"details_{document_id}",
                ):
                    try:
                        details = client.get_document(document_id)
                        st.session_state[f"document_details_{document_id}"] = details
                    except APIClientError as exc:
                        st.error(f"Não foi possível carregar os detalhes: {exc}")

                cached_details: Any = st.session_state.get(
                    f"document_details_{document_id}"
                )

                if isinstance(cached_details, dict):
                    st.json(cached_details)

                confirm = st.checkbox(
                    "Confirmo que desejo excluir este documento e seus índices.",
                    key=f"confirm_delete_{document_id}",
                )

                if st.button(
                    "🗑️ Excluir documento",
                    key=f"delete_{document_id}",
                    disabled=not confirm,
                    type="secondary",
                ):
                    try:
                        with st.spinner("Excluindo documento e índices..."):
                            client.delete_document(document_id)

                        st.session_state["document_action_message"] = (
                            f"Documento '{filename}' excluído com sucesso."
                        )
                        st.session_state.pop(
                            f"confirm_delete_{document_id}",
                            None,
                        )
                        st.session_state.pop(
                            f"document_details_{document_id}",
                            None,
                        )
                        st.rerun()

                    except APIClientError as exc:
                        st.error("Não foi possível excluir o documento.")
                        st.warning(str(exc))

    except APIClientError as exc:
        st.error("Não foi possível carregar o catálogo de documentos.")
        st.warning(str(exc))


def render_source(source: dict[str, Any]) -> None:
    """Render one retrieved source and its metadata."""
    reference = source.get("reference", "Fonte")
    filename = source.get("filename", "Documento sem nome")
    page_number = source.get("page_number")
    score = source.get("rrf_score")
    excerpt = source.get("excerpt", "")

    page_label = f" · Página {page_number}" if page_number is not None else ""
    score_label = (
        f" · Score RRF: {score:.4f}" if isinstance(score, (int, float)) else ""
    )

    with st.expander(f"**{reference}** — {filename}{page_label}"):
        if score_label:
            st.caption(score_label.strip(" ·"))
        st.markdown(excerpt or "_Trecho não disponível._")


def render_query_result(result: dict[str, Any]) -> None:
    """Render the generated answer, sources and model information."""
    st.subheader("Resposta")
    answer = result.get("answer", "")

    if answer:
        st.markdown(answer)
    else:
        st.warning("A API não retornou uma resposta.")

    sources = result.get("sources", [])

    st.subheader(f"Fontes utilizadas ({len(sources)})")

    if sources:
        for source in sources:
            render_source(source)
    else:
        st.info("Nenhuma fonte foi retornada para esta consulta.")

    st.divider()
    st.caption(
        f"Provedor: {result.get('provider', 'Não informado')} · "
        f"Modelo: {result.get('model', 'Não informado')}"
    )

    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")

    if input_tokens is not None or output_tokens is not None:
        col1, col2 = st.columns(2)
        col1.metric(
            "Tokens de entrada",
            input_tokens if input_tokens is not None else "—",
        )
        col2.metric(
            "Tokens de saída",
            output_tokens if output_tokens is not None else "—",
        )


def render_query_page(client: APIClient) -> None:
    """Render the regulatory question and answer page."""
    st.title("⚖️ Consultar legislação")
    st.write(
        "Faça perguntas sobre os documentos regulatórios indexados. "
        "O assistente buscará evidências antes de gerar a resposta."
    )

    with st.form("query_form"):
        question = st.text_area(
            "Qual é a sua pergunta?",
            placeholder=(
                "Ex.: Quais são os princípios previstos no artigo 6º da LGPD?"
            ),
            max_chars=4_000,
            height=120,
        )

        top_k = st.slider(
            "Quantidade máxima de trechos para recuperação",
            min_value=1,
            max_value=20,
            value=5,
            help="Define quantos trechos serão considerados na recuperação.",
        )

        submitted = st.form_submit_button(
            "Consultar documentos",
            type="primary",
        )

    if submitted:
        if not question.strip():
            st.warning("Digite uma pergunta antes de consultar.")
            return

        try:
            with st.spinner("Consultando documentos e gerando resposta..."):
                result = client.query(
                    question=question.strip(),
                    top_k=top_k,
                )

            st.session_state["last_query_result"] = result
            st.session_state["last_question"] = question.strip()

        except APIClientError as exc:
            st.error("Não foi possível concluir a consulta.")
            st.warning(str(exc))

    saved_result = st.session_state.get("last_query_result")

    if isinstance(saved_result, dict):
        previous_question = st.session_state.get("last_question")

        if isinstance(previous_question, str) and previous_question:
            st.markdown("#### Pergunta")
            st.info(previous_question)

        render_query_result(saved_result)


def main() -> None:
    """Run the Streamlit application."""
    client, page = render_sidebar()

    if page == "Gerenciar documentos":
        render_upload_page(client)
    else:
        render_query_page(client)


if __name__ == "__main__":
    main()
