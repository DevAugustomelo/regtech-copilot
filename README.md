# RegTech Copilot RAG

Assistente inteligente para análise documental regulatória baseado em
Large Language Models (LLMs) e Retrieval-Augmented Generation (RAG).

> Projeto acadêmico e de portfólio em desenvolvimento.

## 1. Problema

A análise de documentos regulatórios e de políticas internas pode exigir
a leitura de grandes volumes de texto, a localização de requisitos e a
verificação de evidências em diferentes páginas.

O RegTech Copilot pretende facilitar esse processo permitindo que o
usuário consulte documentos em linguagem natural e receba respostas
fundamentadas em trechos recuperados das fontes.

A ferramenta será um apoio à análise documental. Não substitui avaliação
jurídica, auditoria formal ou decisão profissional de conformidade.

## 2. Objetivos

- Construir uma aplicação web utilizável, não apenas um notebook.
- Implementar um pipeline RAG rastreável.
- Demonstrar conceitos de tokenização, embeddings, Transformers,
  engenharia de prompt e recuperação de informação.
- Permitir o uso de provedores de LLM locais ou remotos.
- Medir qualidade da recuperação, latência e consumo de tokens.
- Aplicar testes, tipagem, documentação e boas práticas de segurança.
- Preparar a aplicação para execução local, Docker e AWS EC2.

## 3. Arquitetura planejada

1. Ingestão: upload e extração de texto de PDFs, preservando páginas.
2. Segmentação: divisão do texto em chunks com limites de tokens.
3. Embeddings: representação vetorial dos trechos.
4. Indexação: armazenamento persistente dos vetores e metadados.
5. Recuperação: busca dos trechos relevantes para uma pergunta.
6. Geração: envio das evidências e instruções ao modelo de linguagem.
7. Resposta: apresentação de conclusões, citações e limitações.
8. Avaliação: testes de recuperação, fundamentação e desempenho.

## 4. Tecnologias

- Python
- FastAPI
- Streamlit
- ChromaDB
- PyMuPDF
- tiktoken
- Pydantic Settings
- HTTPX
- pytest
- Ruff e mypy
- Docker
- GitHub Actions (etapa futura)

Os provedores de LLM e de embeddings serão configuráveis e poderão
utilizar serviços locais ou APIs externas compatíveis.

## 5. Estrutura do repositório

```text
regtech-copilot/
├── src/
│   └── regtech_copilot/
│       ├── api/
│       ├── core/
│       ├── ingestion/
│       ├── embeddings/
│       ├── retrieval/
│       ├── llm/
│       ├── rag/
│       ├── schemas/
│       ├── services/
│       └── ui/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
├── data/
│   ├── documents/
│   └── vector_store/
├── docs/
│   ├── architecture/
│   └── decisions/
├── scripts/
├── .env.example
├── .gitignore
├── .dockerignore
├── pyproject.toml
└── README.md

