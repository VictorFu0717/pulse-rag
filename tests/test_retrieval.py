"""Unit tests for the retrieval pipeline (no GPU / real model required)."""
from __future__ import annotations

from pathlib import Path
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
        load_settings("nonexistent.yaml")


def test_settings_defaults():
    from core.config import Settings
    s = Settings()
    assert s.retrieval.faiss_k == 8
    assert s.retrieval.rerank_top_k == 3
    assert s.memory.max_messages == 10
    assert s.knowledge_bases == []


def test_knowledge_base_config_optional_fields():
    from core.config import KnowledgeBaseConfig
    kb = KnowledgeBaseConfig(
        name="test", index_path="idx", tool_name="t", tool_description="d"
    )
    assert kb.data_dir is None
    assert kb.url_prefix is None
    assert kb.fallback_group is None


# ── KnowledgeBase (mocked FAISS) ──────────────────────────────────────────────

def test_knowledge_base_includes_fallback_docs():
    """Fallback-group docs are always appended regardless of reranking results."""
    from core.config import KnowledgeBaseConfig
    from core.retrieval import KnowledgeBase

    cfg = KnowledgeBaseConfig(
        name="test", index_path="test_index",
        tool_name="test_tool", tool_description="test",
        fallback_group="else", fallback_k=1,
    )

    else_doc = Document(page_content="else doc", metadata={"group": "else"})
    main_doc = Document(page_content="main doc", metadata={"group": "products"})

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
    assert "Source:" in serialized and "Content:" in serialized


def test_knowledge_base_no_fallback():
    """Without fallback_group, only reranked docs are returned."""
    from core.config import KnowledgeBaseConfig
    from core.retrieval import KnowledgeBase

    cfg = KnowledgeBaseConfig(
        name="test", index_path="test_index",
        tool_name="test_tool", tool_description="test",
    )
    main_doc = Document(page_content="main doc", metadata={"group": "products"})

    mock_store = MagicMock()
    mock_store.docstore._dict = {"a": main_doc}
    mock_store.similarity_search.return_value = [main_doc]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [main_doc]

    with patch("core.retrieval.FAISS.load_local", return_value=mock_store):
        kb = KnowledgeBase(cfg, MagicMock(), mock_reranker, faiss_k=8, rerank_top_k=3)
        docs, _ = kb.retrieve("query")

    assert docs == [main_doc]


def test_build_retrieval_tools_skips_missing_index(tmp_path):
    """build_retrieval_tools warns and skips KBs whose index doesn't exist."""
    from core.config import Settings, KnowledgeBaseConfig
    from core.retrieval import build_retrieval_tools

    settings = Settings(
        knowledge_bases=[
            KnowledgeBaseConfig(
                name="missing",
                index_path=str(tmp_path / "nonexistent_index"),
                tool_name="retrieve_missing",
                tool_description="test",
            )
        ]
    )
    tools = build_retrieval_tools(settings, MagicMock(), MagicMock())
    assert tools == []


# ── Reranker (mocked) ─────────────────────────────────────────────────────────

def test_bge_reranker_v2_empty_input():
    from core.reranker import BGEReranker_v2
    with patch("core.reranker.LayerWiseFlagLLMReranker"):
        r = BGEReranker_v2()
        assert r.rerank("query", []) == []
        assert r.rerank_with_scores("query", []) == []
        assert r.rerank_with_threshold("query", []) == []


# ── build_index script ────────────────────────────────────────────────────────

def test_build_docs_from_txt_files(tmp_path):
    """build_docs correctly reads .txt files and attaches category metadata."""
    from core.config import KnowledgeBaseConfig
    from scripts.build_index import build_docs

    # Create minimal FAQ structure
    (tmp_path / "products").mkdir()
    (tmp_path / "products" / "overview.txt").write_text("We offer cloud hosting.")

    cfg = KnowledgeBaseConfig(
        name="test", data_dir=str(tmp_path),
        index_path="idx", tool_name="t", tool_description="d",
        url_prefix="https://example.com",
    )
    categories = {
        "products": {"group": "products", "label": "Products", "description": "Our services"}
    }
    docs = build_docs(cfg, categories)

    assert len(docs) == 1
    assert "cloud hosting" in docs[0].page_content
    assert docs[0].metadata["group"] == "products"
    assert docs[0].metadata["source"] == "https://example.com/products/overview"


def test_build_docs_no_url_prefix(tmp_path):
    """Without url_prefix, source is None."""
    from core.config import KnowledgeBaseConfig
    from scripts.build_index import build_docs

    (tmp_path / "billing").mkdir()
    (tmp_path / "billing" / "faq.txt").write_text("Payment FAQ content.")

    cfg = KnowledgeBaseConfig(
        name="test", data_dir=str(tmp_path),
        index_path="idx", tool_name="t", tool_description="d",
    )
    docs = build_docs(cfg, {})
    assert docs[0].metadata["source"] is None


def test_build_docs_unknown_category_falls_back(tmp_path):
    """Categories not in faq_categories.yaml use the folder name as label."""
    from core.config import KnowledgeBaseConfig
    from scripts.build_index import build_docs

    (tmp_path / "my_custom_topic").mkdir()
    (tmp_path / "my_custom_topic" / "info.txt").write_text("Custom content here.")

    cfg = KnowledgeBaseConfig(
        name="test", data_dir=str(tmp_path),
        index_path="idx", tool_name="t", tool_description="d",
    )
    docs = build_docs(cfg, {})  # empty categories → fallback
    assert docs[0].metadata["label"] == "my_custom_topic"
