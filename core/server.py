from __future__ import annotations

import importlib
import json
import logging
import os
import pkgutil
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import plugins as _plugins_pkg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.config import Settings

logger = logging.getLogger(__name__)

_EXCLUDED_STREAM_NODES = {
    "SummarizationMiddleware",
    "SummarizationMiddleware.before_model",
    "SummarizationMiddleware.after_model",
}


def _setup_logging(settings: Settings) -> None:
    os.makedirs(settings.logging.dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = TimedRotatingFileHandler(
        filename=os.path.join(settings.logging.dir, "server.log"),
        when="midnight",
        interval=1,
        backupCount=settings.logging.retention_days,
        encoding="utf-8",
    )
    handler.setFormatter(fmt)
    handler.suffix = "%Y-%m-%d"
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))
    root.addHandler(handler)


def _build_llm(settings: Settings):
    if settings.llm.provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.llm.model,
            base_url=settings.llm.base_url,
            temperature=settings.llm.temperature,
        )
    if settings.llm.provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            streaming=True,
        )
    if settings.llm.provider == "custom":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key or "EMPTY",
            model=settings.llm.model,
            temperature=settings.llm.temperature,
        )
    raise ValueError(f"Unknown LLM provider: {settings.llm.provider!r}")


class QueryRequest(BaseModel):
    question: str
    thread_id: str


def create_app(settings: Settings) -> FastAPI:
    _setup_logging(settings)

    from langchain_huggingface import HuggingFaceEmbeddings
    from core.reranker import BGEReranker_v2
    from core.retrieval import build_retrieval_tools
    from core.agent import build_agent

    llm = _build_llm(settings)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embeddings.model)
    reranker = BGEReranker_v2(
        model_name=settings.reranker.model,
        use_fp16=settings.reranker.use_fp16,
        cutoff_layers=settings.reranker.cutoff_layers,
        batch_size=settings.reranker.batch_size,
    )

    tools = build_retrieval_tools(settings, embeddings, reranker)

    disabled = set(settings.tools.disabled_plugins)
    for _, name, _ in pkgutil.iter_modules(_plugins_pkg.__path__):
        if name in disabled:
            logger.info("Plugin %r disabled — skipping", name)
            continue
        try:
            module = importlib.import_module(f"plugins.{name}")
            for attr_name in dir(module):
                if not attr_name.startswith("build_"):
                    continue
                attr = getattr(module, attr_name)
                if callable(attr):
                    result = attr()
                    if isinstance(result, list):
                        tools.extend(result)
                    elif result is not None:
                        tools.append(result)
            logger.info("Plugin %r loaded", name)
        except Exception as e:
            logger.warning("Plugin %r skipped: %s", name, e)

    prompt_path = Path(settings.system_prompt_path)
    system_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""

    agent = build_agent(settings, llm, tools, system_prompt)

    app = FastAPI(
        title=settings.app.name,
        version="2.0",
        description="Agentic RAG customer service — powered by LangGraph + FAISS + BGE Reranker",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def chat_ui():
        ui_path = Path(__file__).parent.parent / "static" / "index.html"
        if not ui_path.exists():
            return HTMLResponse("<h1>Chat UI not found</h1>", status_code=404)
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health():
        return {"status": "ok", "models_loaded": True}

    @app.post("/query")
    async def query(request: QueryRequest):
        cfg = {"configurable": {"thread_id": request.thread_id}}
        t0 = time.time()
        try:
            logger.info("Request | thread=%s | q=%s", request.thread_id, request.question)
            response = None
            for step in agent.stream(
                {"messages": [{"role": "user", "content": request.question}]},
                stream_mode="values",
                config=cfg,
            ):
                response = step["messages"][-1].content
            logger.info(
                "Done | thread=%s | %.2fs | %s",
                request.thread_id, time.time() - t0, response,
            )
            return {"response": response}
        except Exception as e:
            logger.error("Error | thread=%s | %s", request.thread_id, e, exc_info=True)
            return {"response": "系統發生錯誤，請稍後再試"}

    @app.post("/query-stream")
    async def query_stream(request: QueryRequest):
        cfg = {"configurable": {"thread_id": request.thread_id}}

        async def event_generator():
            try:
                async for event in agent.astream_events(
                    {"messages": [{"role": "user", "content": request.question}]},
                    config=cfg,
                    version="v2",
                ):
                    if event["event"] != "on_chat_model_stream":
                        continue
                    if event.get("metadata", {}).get("langgraph_node", "") in _EXCLUDED_STREAM_NODES:
                        continue
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content and not getattr(chunk, "tool_call_chunks", None):
                        yield f"data: {json.dumps({'token': chunk.content}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Stream error | %s | %s", request.thread_id, e, exc_info=True)
                yield f"data: {json.dumps({'error': '系統發生錯誤，請稍後再試'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
