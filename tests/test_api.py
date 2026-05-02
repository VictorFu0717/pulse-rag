"""Integration tests for the FastAPI endpoints (mocked — no real models needed)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.config import Settings, ToolsConfig


def _minimal_settings() -> Settings:
    return Settings(
        tools=ToolsConfig(web_search_enabled=False, domain_checker_enabled=False),
        knowledge_bases=[],
    )


def _mock_agent(response_text: str = "test response") -> MagicMock:
    agent = MagicMock()
    agent.stream.return_value = iter([
        {"messages": [MagicMock(content=response_text)]}
    ])
    return agent


@pytest.fixture
def client():
    settings = _minimal_settings()
    with (
        patch("core.server.HuggingFaceEmbeddings", return_value=MagicMock()),
        patch("core.server.BGEReranker_v2", return_value=MagicMock()),
        patch("core.server.build_retrieval_tools", return_value=[]),
        patch("core.server._build_llm", return_value=MagicMock()),
        patch("core.server.build_agent", return_value=_mock_agent()),
    ):
        from core.server import create_app
        yield TestClient(create_app(settings))


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "models_loaded": True}


# ── /query ────────────────────────────────────────────────────────────────────

def test_query_returns_response(client):
    r = client.post("/query", json={"question": "How do I register a domain?", "thread_id": "t1"})
    assert r.status_code == 200
    assert "response" in r.json()
    assert r.json()["response"] == "test response"


def test_query_missing_thread_id(client):
    r = client.post("/query", json={"question": "hello"})
    assert r.status_code == 422   # Pydantic validation error


def test_query_missing_question(client):
    r = client.post("/query", json={"thread_id": "t1"})
    assert r.status_code == 422


def test_query_empty_payload(client):
    r = client.post("/query", json={})
    assert r.status_code == 422
