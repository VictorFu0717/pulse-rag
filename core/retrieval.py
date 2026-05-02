from __future__ import annotations

import logging
import time
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.tools import tool

from core.config import Settings, KnowledgeBaseConfig
from core.reranker import BGEReranker_v2

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Wraps a FAISS index + BGE reranker for one language / domain."""

    def __init__(
        self,
        cfg: KnowledgeBaseConfig,
        embeddings,
        reranker: BGEReranker_v2,
        faiss_k: int,
        rerank_top_k: int,
    ):
        self.cfg = cfg
        self.reranker = reranker
        self.faiss_k = faiss_k
        self.rerank_top_k = rerank_top_k

        self.store = FAISS.load_local(
            cfg.index_path,
            embeddings,
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.COSINE,
        )

        # Pre-load fallback docs (always appended regardless of query relevance)
        self.fallback_docs: List[Document] = []
        if cfg.fallback_group:
            self.fallback_docs = [
                d for d in self.store.docstore._dict.values()
                if d.metadata.get("group") == cfg.fallback_group
            ][: cfg.fallback_k]

    def retrieve(self, query: str) -> Tuple[List[Document], str]:
        t0 = time.time()
        candidates = self.store.similarity_search(query, k=self.faiss_k)
        logger.debug("FAISS %.3fs — %d candidates", time.time() - t0, len(candidates))

        main_docs = [
            d for d in candidates
            if d.metadata.get("group") != self.cfg.fallback_group
        ]

        t1 = time.time()
        reranked = self.reranker.rerank(query, main_docs, top_k=self.rerank_top_k)
        logger.debug("Rerank %.3fs", time.time() - t1)

        final = reranked + self.fallback_docs
        serialized = "\n\n".join(
            f"Source: {d.metadata}\nContent: {d.page_content}" for d in final
        )
        return final, serialized


def build_retrieval_tools(settings: Settings, embeddings, reranker: BGEReranker_v2) -> list:
    """Return one @tool per knowledge base defined in config."""
    tools = []

    for kb_cfg in settings.knowledge_bases:
        kb = KnowledgeBase(
            cfg=kb_cfg,
            embeddings=embeddings,
            reranker=reranker,
            faiss_k=settings.retrieval.faiss_k,
            rerank_top_k=settings.retrieval.rerank_top_k,
        )

        def _make_tool(kb=kb, cfg=kb_cfg):
            def retrieval_fn(query: str):
                logger.info("Tool: %s | query: %s", cfg.tool_name, query)
                return kb.retrieve(query)

            retrieval_fn.__name__ = cfg.tool_name
            retrieval_fn.__doc__ = cfg.tool_description
            return tool(cfg.tool_name, response_format="content_and_artifact")(retrieval_fn)

        tools.append(_make_tool())

    return tools
