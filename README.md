# Academic RAG Research Assistant

Local-first RAG app to chat with your research papers with grounded citations.

Upload PDFs → chunk + embed locally → store in Qdrant → retrieve top-k chunks → answer with a local LLM via LM Studio, with `[Source N]` citations and expandable source cards.

Built for PFE / research use where data stays on your machine.

## Stack

- **Frontend / App:** Streamlit (`app.py`, ~787 lines, custom CSS, 2 tabs + sidebar)
- **PDF parsing:** PyMuPDF (`fitz`)
- **Embeddings (local):** Ollama `qwen3-embedding:8b`, 4096-dim
- **Vector DB:** Qdrant, collection `research_papers`, cosine distance
- **Chat LLM (local):** LM Studio OpenAI-compatible server, default `openai/gpt-oss-20b`
- **Chunking:** 1200 chars with 200 overlap, sentence-boundary aware

## Features

- Upload multiple PDFs, extract text + heuristic title, chunk, embed, upsert with payload (`filename`, `chunk_text`, `chunk_index`, `upload_time`)
- Paper library in sidebar: paper count, chunk count, per-paper delete, clear chat, reset collection
- Research chat: semantic search (`limit` slider 1-10, default 5), system prompt forces answer ONLY from context, `[Source N]` citations, LaTeX preserved, multilingual answer
- Status badges for Ollama + LM Studio, streaming responses via `/chat/completions` SSE
- No cloud calls — all inference local

## Project structure

```
.
├── app.py            # full app: config, Qdrant, PDF, embeddings, search, chat, UI
└── requirements.txt  # streamlit, requests, PyMuPDF, qdrant-client, ollama
```

Key functions in `app.py`: `extract_pdf_text`, `extract_metadata`, `create_chunks`, `generate_embedding`, `semantic_search`, `build_context`, `stream_chat`, `get_qdrant_client`, `get_unique_papers`, `delete_paper`.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running locally
- [Qdrant](https://qdrant.tech) running on `localhost:6333`
- [LM Studio](https://lmstudio.ai) running OpenAI-compatible server on `http://127.0.0.1:4040/v1`

Pull models:

```bash
ollama pull qwen3-embedding:8b
# in LM Studio: download openai/gpt-oss-20b and start server on port 4040
```

Run Qdrant (Docker):

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. Open Upload Papers tab → select PDFs → Process and Index Papers
2. Switch to Research Chat tab → ask a question → check Sources panel on right for chunks + scores

## Configuration (`app.py` top)

```python
CHAT_MODEL = "openai/gpt-oss-20b"
EMBED_MODEL = "qwen3-embedding:8b"
VECTOR_SIZE = 4096
QDRANT_COLLECTION = "research_papers"
LMSTUDIO_BASE = "http://127.0.0.1:4040/v1"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
```

Change these if you use a different embedding dim or LM Studio port/model.

## Limitations / next steps

- Point IDs use `md5(filename_i_timestamp)` — re-upload duplicates instead of upserting; add deterministic hash for idempotency
- No eval (faithfulness, retrieval recall), no hybrid search / reranker
- Title extraction is heuristic (first 15 lines); use GROBID / PDF metadata for robustness
- `requirements.txt` unpinned; pin for reproducibility
- No Docker Compose, no auth, single collection for all users

Good next steps for PFE: add `docker-compose.yml` (streamlit + qdrant), eval set, hybrid BM25 + dense, reranking, per-user collections.

## Author

Nizar Sahl — https://github.com/nmetal05/rag-research-assistant
