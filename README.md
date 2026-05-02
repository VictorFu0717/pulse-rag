# Pluse RAG

> A configurable, production-ready **Agentic RAG** framework for enterprise customer service — built with LangGraph, FAISS, and BGE Reranker.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Pluse RAG?

Pluse RAG is an **agentic RAG chatbot framework** designed for enterprise customer support.  
It ships with a fully working example (Net-Chinese domain registrar) that you can replace with your own knowledge base in minutes.

**Key capabilities:**
- **Two-stage retrieval** — FAISS vector search → BGE LayerWise Reranker for high precision
- **Multi-language** — separate indices per language, each becomes its own agent tool
- **Streaming** — Server-Sent Events for real-time token output
- **Config-driven** — swap LLM, embeddings, add knowledge bases via `config/config.yaml`
- **Plugin tools** — web search (Tavily) and domain availability check included; add your own
- **Conversation memory** — LangGraph InMemorySaver with smart message trimming

---

## Architecture

```
User
 │
 ▼
FastAPI  (/query  /query-stream  /health)
 │
 ▼
LangGraph Agent  (create_agent + trim_messages middleware)
 │
 ├─► retrieve_tra_chi ──► FAISS (zh-TW)  ──► BGE Reranker ──► top-k docs
 ├─► retrieve_sim_chi ──► FAISS (zh-CN)  ──► BGE Reranker ──► top-k docs
 ├─► retrieve_en      ──► FAISS (en)     ──► BGE Reranker ──► top-k docs
 ├─► net_search       ──► Tavily Web Search
 └─► check_domain_*   ──► WhoisJSON API / local whois fallback
 │
 ▼
LLM  (Ollama Qwen / OpenAI / any OpenAI-compatible endpoint)
 │
 ▼
Response (streaming SSE or blocking JSON)
```

### Why these choices?

| Decision | Rationale |
|---|---|
| **LangGraph** over raw LCEL chain | Built-in checkpointing, tool-call loop, middleware hooks |
| **BGE LayerWise Reranker** | 5–15% precision gain over vector similarity alone; early-exit layers trade accuracy for speed |
| **FAISS** over hosted vector DB | Zero infra dependency; indices are files — easy to version and ship |
| **Streaming SSE** over WebSocket | Simpler client implementation; stateless per-request |
| **trim_messages middleware** | Prevents orphaned ToolMessages from triggering blank LLM responses on context overflow |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/VictorFu0717/pluse-rag.git
cd pluse-rag
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in TAVILY_API_KEY, HUGGINGFACEHUB_API_TOKEN, etc.
```

Edit `config/config.yaml` to set your LLM provider, knowledge bases, and tool settings.

### 3. Build FAISS indices

```bash
python scripts/build_index.py
```

> Skip this step if you already have `faiss_index_*/` directories.

### 4. Start the server

```bash
python main.py
# → http://localhost:8002
# → http://localhost:8002/docs  (Swagger UI)
```

### Docker (recommended for production)

```bash
docker compose up --build
```

---

## Configuration Guide

All configuration lives in **`config/config.yaml`**.  
Edit this file — no code changes required.

### Switch LLM provider

```yaml
# Ollama (default)
llm:
  provider: "ollama"
  model: "qwen3.5:latest"
  base_url: "http://localhost:11434"

# OpenAI
llm:
  provider: "openai"
  model: "gpt-4o-mini"

# Any OpenAI-compatible endpoint (vLLM, LM Studio, etc.)
llm:
  provider: "custom"
  model: "my-model"
  base_url: "http://localhost:8317/v1"
  api_key: "your-key"
```

### Add a knowledge base

```yaml
knowledge_bases:
  - name: "japanese"
    index_path: "faiss_index_ja"
    tool_name: "retrieve_ja"
    tool_description: "日本語で質問する顧客に使うツール"
```

Then rebuild: `python scripts/build_index.py --kb japanese`

### Enable / disable plugins

```yaml
tools:
  web_search_enabled: true      # Tavily
  domain_checker_enabled: true  # WhoisJSON
```

---

## Customising for Your Company

1. **Replace the system prompt** — edit `config/system_prompt.txt`
2. **Replace the FAQ data** — drop `.txt` files into `FAQ_data/` (see `data/README.md` for format)
3. **Update category metadata** — edit `data/faq_categories.yaml`
4. **Rebuild indices** — `python scripts/build_index.py`
5. **Add custom tools** — create a file in `plugins/`, follow the pattern in `plugins/domain_checker.py`

---

## API Reference

### `GET /health`

```json
{ "status": "ok", "models_loaded": true }
```

### `POST /query`

```json
// Request
{ "question": "How do I renew a domain?", "thread_id": "user-session-123" }

// Response
{ "response": "You can renew your domain by..." }
```

### `POST /query-stream`

Returns Server-Sent Events:

```
data: {"token": "You"}
data: {"token": " can"}
data: {"token": " renew..."}
data: [DONE]
```

---

## Project Structure

```
pluse-rag/
├── core/
│   ├── config.py       # Pydantic settings models + YAML loader
│   ├── agent.py        # LangGraph agent factory + trim_messages middleware
│   ├── retrieval.py    # KnowledgeBase class + retrieval tool factory
│   ├── reranker.py     # BGEReranker and BGEReranker_v2
│   └── server.py       # FastAPI app factory
│
├── plugins/
│   ├── domain_checker.py   # check_domain_available, check_domains_bulk
│   └── web_search.py       # net_search (Tavily)
│
├── config/
│   ├── config.yaml         # ← main configuration file
│   └── system_prompt.txt   # ← LLM system prompt
│
├── data/
│   ├── faq_categories.yaml # category metadata for index building
│   └── README.md           # data format guide
│
├── scripts/
│   └── build_index.py      # build FAISS indices from FAQ files
│
├── tests/
│   ├── test_retrieval.py   # unit tests (no GPU required)
│   └── test_api.py         # API integration tests (mocked)
│
├── FAQ_data/               # raw FAQ text files (zh-TW / zh-CN / en)
├── faiss_index_*/          # pre-built FAISS indices
│
├── main.py                 # entry point
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

Tests run without GPU — all heavy models are mocked.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph 1.x + LangChain 1.x |
| Vector store | FAISS (cosine similarity) |
| Embeddings | `BAAI/bge-m3` (multilingual) |
| Reranker | `BAAI/bge-reranker-v2-minicpm-layerwise` |
| LLM | Ollama (Qwen) / OpenAI compatible |
| API server | FastAPI + uvicorn |
| Web search | Tavily |
| Domain check | WhoisJSON API + python-whois fallback |

---

## License

MIT
