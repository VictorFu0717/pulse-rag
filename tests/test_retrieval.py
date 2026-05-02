"""Unit tests for the retrieval pipeline (no GPU / real model required)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ── Config loading ────────────────────────────────────────────────────────────

def test_load_settings_minimal(tmp_path):
    from core.config import load_settings

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "app:\n  name: TestApp\n  port: 9000\n"
        "llm:\n  provider: openai\n  model: gpt-4o-mini\n"
    )
    s = load_settings(str(cfg))
    assert s.app.name == "TestApp"
    assert s.app.port == 9000
    assert s.llm.provider == "openai"


def test_load_settings_missing_file():
    from core.config import load_settings
    with pytest.raises(FileNotFoundError):
        load_settings("nonexistent_config.yaml")


def test_settings_defaults():
    from core.config import Settings
    s = Settings()
    assert s.retrieval.faiss_k == 8
    assert s.retrieval.rerank_top_k == 3
    assert s.memory.max_messages == 10


# ── KnowledgeBase ─────────────────────────────────────────────────────────────

def test_knowledge_base_includes_fallback_docs():
    """KnowledgeBase always appends fallback_group docs regardless of reranking."""
    from core.config import KnowledgeBaseConfig
    from core.retrieval import KnowledgeBase

    cfg = KnowledgeBaseConfig(
        name="test",
        index_path="test_index",
        tool_name="test_tool",
        tool_description="test",
        fallback_group="else",
        fallback_k=1,
    )

    else_doc = Document(page_content="else doc", metadata={"group": "else"})
    main_doc = Document(page_content="main doc", metadata={"group": "domain"})

    mock_store = MagicMock()
    mock_store.docstore._dict = {"a": else_doc, "b": main_doc}
    mock_store.similarity_search.return_value = [main_doc]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [main_doc]

    with patch("core.retrieval.FAISS.load_local", return_value=mock_store):
        kb = KnowledgeBase(cfg, MagicMock(), mock_reranker, faiss_k=8, rerank_top_k=3)
        docs, serialized = kb.retrieve("test query")

    assert else_doc in docs
    assert main_doc in docs
    assert "Source:" in serialized
    assert "Content:" in serialized


def test_knowledge_base_no_fallback():
    """KnowledgeBase without fallback_group returns only reranked docs."""
    from core.config import KnowledgeBaseConfig
    from core.retrieval import KnowledgeBase

    cfg = KnowledgeBaseConfig(
        name="test",
        index_path="test_index",
        tool_name="test_tool",
        tool_description="test",
    )
    main_doc = Document(page_content="main doc", metadata={"group": "domain"})
    mock_store = MagicMock()
    mock_store.docstore._dict = {"a": main_doc}
    mock_store.similarity_search.return_value = [main_doc]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [main_doc]

    with patch("core.retrieval.FAISS.load_local", return_value=mock_store):
        kb = KnowledgeBase(cfg, MagicMock(), mock_reranker, faiss_k=8, rerank_top_k=3)
        docs, _ = kb.retrieve("query")

    assert docs == [main_doc]


# ── Reranker (mock) ───────────────────────────────────────────────────────────

def test_bge_reranker_v2_empty_input():
    """rerank() returns [] for empty doc list without touching the model."""
    from core.reranker import BGEReranker_v2

    with patch("core.reranker.LayerWiseFlagLLMReranker"):
        r = BGEReranker_v2()
        assert r.rerank("query", []) == []
        assert r.rerank_with_scores("query", []) == []
        assert r.rerank_with_threshold("query", []) == []
