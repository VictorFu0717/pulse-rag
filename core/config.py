from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str = "Pluse RAG"
    host: str = "0.0.0.0"
    port: int = 8002
    cors_origins: List[str] = ["*"]


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai", "custom"] = "ollama"
    model: str = "qwen3.5:latest"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0


class EmbeddingsConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    device: str = "auto"


class RetrievalConfig(BaseModel):
    faiss_k: int = 8
    rerank_top_k: int = 3


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-v2-minicpm-layerwise"
    cutoff_layers: List[int] = [28]
    use_fp16: bool = True
    batch_size: int = 16


class KnowledgeBaseConfig(BaseModel):
    name: str
    index_path: str
    tool_name: str
    tool_description: str
    fallback_group: Optional[str] = None
    fallback_k: int = 1


class ToolsConfig(BaseModel):
    web_search_enabled: bool = True
    domain_checker_enabled: bool = False


class MemoryConfig(BaseModel):
    max_messages: int = 10


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    retention_days: int = 7


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    knowledge_bases: List[KnowledgeBaseConfig] = []
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    system_prompt_path: str = "config/system_prompt.txt"


def load_settings(config_path: str = "config/config.yaml") -> Settings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(**raw)
