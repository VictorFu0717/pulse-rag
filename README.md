# Pulse RAG

> A configurable, production-ready **Agentic RAG** framework for enterprise customer service — built with LangGraph, FAISS, and BGE Reranker.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is Pulse RAG?

Pulse RAG is a **plug-and-play Agentic RAG chatbot framework** for enterprise customer service.  
Drop in your FAQ documents, set your company name and LLM, and have a production-ready chatbot running in minutes.

**Key capabilities:**
- **Two-stage retrieval** — FAISS vector search → BGE LayerWise Reranker for high-precision answers
- **Config-driven** — swap LLM provider, embedding model, add knowledge bases — no code changes needed
- **Multi-language** — one knowledge base per language, each becomes its own agent tool
- **Streaming** — Server-Sent Events for real-time token output
- **Plugin tools** — web search (Tavily) and domain availability check; add your own in `plugins/`
- **Conversation memory** — LangGraph InMemorySaver with smart message trimming

---

## Architecture

```
User Query
    │
    ▼
FastAPI  (/query · /query-stream · /health)
    │
    ▼
LangGraph Agent  ← trim_messages middleware (sliding window)
    │
    ├─► retrieve_en  ──► FAISS index ──► BGE Reranker ──► top-k docs
    ├─► retrieve_*   ──► (add more knowledge bases in config.yaml)
    ├─► net_search   ──► Tavily Web Search
    └─► (your custom plugins)
    │
    ▼
LLM  (Ollama · OpenAI · any OpenAI-compatible endpoint)
    │
    ▼
Response  (streaming SSE or blocking JSON)
```

### Why these design choices?

| Decision | Rationale |
|---|---|
| **LangGraph** agent | Built-in checkpointing, tool-call loop, middleware hooks for message trimming |
| **BGE LayerWise Reranker** | 5–15% precision gain over vector similarity alone; early-exit layers trade accuracy for speed |
| **FAISS** over hosted vector DB | Zero infrastructure dependency; indices are files — easy to version and ship |
| **Streaming SSE** | Simpler client than WebSocket; stateless per-request |
| **trim_messages middleware** | Prevents orphaned ToolMessages from causing blank responses on context overflow |
| **Config-driven KBs** | Each knowledge base is a YAML entry — no code change to add a language or domain |

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/VictorFu0717/pulse-rag.git
cd pulse-rag
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Fill in HUGGINGFACEHUB_API_TOKEN (needed to download BGE models)
# Add TAVILY_API_KEY if you want web search
```

### 3. Build the index

```bash
python scripts/build_index.py
# Builds FAISS index from data/example_faq/ — takes a few minutes on first run
# (downloads BAAI/bge-m3 embedding model ~1 GB)
```

### 4. Start the server

```bash
python main.py
# → API: http://localhost:8002
# → Docs: http://localhost:8002/docs
```

### Docker

```bash
# Build index first (needs HuggingFace models)
python scripts/build_index.py

# Then run everything with Docker
docker compose up --build
```

---

## Customise for Your Company

### Step 1 — Replace the example FAQ data

Put your `.txt` FAQ files under any directory (e.g. `data/my_faq/en/`).  
Organise them into sub-folders by topic — each folder becomes a category.

```
data/my_faq/
└── en/
    ├── products/
    │   └── catalog.txt
    ├── billing/
    │   └── payment_methods.txt
    └── support/
        └── troubleshooting.txt
```

### Step 2 — Update the config

Edit `config/config.yaml`:

```yaml
app:
  name: "My Company AI Assistant"

knowledge_bases:
  - name: "english"
    data_dir: "data/my_faq/en"          # ← point to your data
    index_path: "faiss_index_en"
    url_prefix: "https://help.mycompany.com/en"  # ← optional source links
    tool_name: "retrieve_en"
    tool_description: "Search the English knowledge base for customer questions"
```

### Step 3 — Customise the system prompt

Edit `config/system_prompt.txt`.  
Replace `[COMPANY_NAME]` with your company name and adjust the support contact details.

### Step 4 — Rebuild the index

```bash
python scripts/build_index.py --kb english
```

### Step 5 — Start the server

```bash
python main.py
```

---

## Configuration Reference

All settings live in **`config/config.yaml`**.

### LLM Provider

```yaml
# Ollama (default — free, local)
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

### Multiple Knowledge Bases (multi-language)

```yaml
knowledge_bases:
  - name: "english"
    data_dir: "data/faq/en"
    index_path: "faiss_index_en"
    tool_name: "retrieve_en"
    tool_description: "Search English knowledge base"

  - name: "traditional_chinese"
    data_dir: "data/faq/zh-TW"
    index_path: "faiss_index_zh_tw"
    tool_name: "retrieve_zh_tw"
    tool_description: "搜尋繁體中文知識庫"
```

### Adding a Custom Plugin Tool

1. Create `plugins/my_tool.py`:

```python
from langchain_core.tools import tool

def build_my_tool() -> list:
    @tool
    def my_tool(query: str) -> str:
        "Description of what this tool does"
        # your logic here
        return "result"
    return [my_tool]
```

2. Register it in `core/server.py` (or extend `ToolsConfig` in `core/config.py`).

---

## API Reference

### `GET /health`
```json
{ "status": "ok", "models_loaded": true }
```

### `POST /query`
```json
// Request
{ "question": "How do I reset my password?", "thread_id": "user-session-42" }

// Response
{ "response": "You can reset your password by..." }
```

### `POST /query-stream`
Returns Server-Sent Events:
```
data: {"token": "You"}
data: {"token": " can"}
data: [DONE]
```

---

## Project Structure

```
pulse-rag/
├── core/
│   ├── config.py        # Pydantic settings + YAML loader
│   ├── agent.py         # LangGraph agent factory + trim_messages middleware
│   ├── retrieval.py     # KnowledgeBase + retrieval tool factory
│   ├── reranker.py      # BGEReranker and BGEReranker_v2 (LayerWise)
│   └── server.py        # FastAPI app factory
│
├── plugins/
│   ├── domain_checker.py    # check_domain_available, check_domains_bulk
│   └── web_search.py        # net_search (Tavily)
│
├── config/
│   ├── config.yaml          # ← main configuration (edit this)
│   └── system_prompt.txt    # ← LLM system prompt (edit this)
│
├── data/
│   ├── faq_categories.yaml  # category metadata for index building
│   ├── README.md            # data format guide
│   └── example_faq/         # sample FAQ for AcmeTech (replace with your data)
│       └── en/
│           ├── products/
│           ├── billing/
│           ├── support/
│           └── account/
│
├── scripts/
│   └── build_index.py   # build FAISS indices from your FAQ files
│
├── tests/
│   ├── test_retrieval.py    # unit tests (no GPU required)
│   └── test_api.py          # API integration tests (mocked)
│
├── main.py              # entry point
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
| Reranker | `BAAI/bge-reranker-v2-minicpm-layerwise` (LayerWise) |
| LLM | Ollama / OpenAI / any OpenAI-compatible |
| API server | FastAPI + uvicorn |
| Web search | Tavily |

---

## License

MIT
