"""Build FAISS vector indices from your FAQ text files.

How it works
------------
For each knowledge base defined in config/config.yaml:
  1. Reads all .txt files under data_dir recursively
  2. Uses the immediate parent folder name as the document category
  3. Looks up category metadata in data/faq_categories.yaml (optional)
  4. Builds a FAISS index and saves it to index_path

Data directory layout (any depth works):
    data/my_faq/
    ├── products/
    │   ├── service_overview.txt
    │   └── pricing.txt
    ├── billing/
    │   └── payment_methods.txt
    └── support/
        └── troubleshooting.txt

Usage
-----
    python scripts/build_index.py                    # build all KBs
    python scripts/build_index.py --kb english       # build one KB by name
    python scripts/build_index.py --config custom.yaml
    python scripts/build_index.py --categories data/my_categories.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import KnowledgeBaseConfig, load_settings


def load_categories(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("categories", {})


def _source_url(cfg: KnowledgeBaseConfig, category: str, stem: str) -> str | None:
    """Build a source URL from url_prefix + category + filename (all optional)."""
    if not cfg.url_prefix:
        return None
    prefix = cfg.url_prefix.rstrip("/")
    return f"{prefix}/{category}/{stem}"


def build_docs(cfg: KnowledgeBaseConfig, categories: dict) -> list[Document]:
    data_dir = Path(cfg.data_dir)
    if not data_dir.exists():
        print(f"  ⚠ data_dir not found: {data_dir}")
        return []

    docs = []
    txt_files = sorted(data_dir.rglob("*.txt"))
    if not txt_files:
        print(f"  ⚠ No .txt files found in {data_dir}")
        return []

    for filepath in txt_files:
        category = filepath.parent.name
        stem = filepath.stem

        with open(filepath, encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        # Metadata: prefer categories.yaml entry; fall back to folder name
        if category in categories:
            meta = dict(categories[category])
        else:
            meta = {"group": category, "label": category, "description": ""}

        meta["source"] = _source_url(cfg, category, stem)
        meta["file"] = str(filepath.relative_to(Path(".")))

        # Prepend category context so the embedding captures the topic
        label = meta.get("label", category)
        description = meta.get("description", "")
        header = f"[Category: {label}]"
        if description:
            header += f"\n[Description: {description}]"

        full_text = f"{header}\n\n{content}"
        docs.append(Document(page_content=full_text, metadata=meta))

    return docs


_PROJECT_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for Pulse RAG")
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "config/config.yaml"),
                        help="Path to config.yaml")
    parser.add_argument("--categories", default=str(_PROJECT_ROOT / "data/faq_categories.yaml"),
                        help="Path to faq_categories.yaml (optional)")
    parser.add_argument("--kb", default=None,
                        help="Build only this knowledge base (by name). Builds all if omitted.")
    args = parser.parse_args()

    settings = load_settings(args.config)
    categories = load_categories(args.categories)

    kbs_to_build = [
        kb for kb in settings.knowledge_bases
        if args.kb is None or kb.name == args.kb
    ]

    if not kbs_to_build:
        print(f"No matching knowledge bases found. Available: "
              f"{[kb.name for kb in settings.knowledge_bases]}")
        sys.exit(1)

    print("Loading embedding model…")
    embedding = HuggingFaceEmbeddings(model_name=settings.embeddings.model)

    for kb in kbs_to_build:
        print(f"\n{'='*60}")
        print(f"Building: {kb.name}")

        if not kb.data_dir:
            print("  ⚠ No data_dir set in config — skipping")
            continue

        docs = build_docs(kb, categories)
        if not docs:
            print("  ⚠ No documents found — skipping")
            continue

        print(f"  Documents: {len(docs)}")
        print(f"  Building FAISS index…")
        vectorstore = FAISS.from_documents(
            docs, embedding, distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(kb.index_path)
        print(f"  ✅ Saved → {kb.index_path}/")


if __name__ == "__main__":
    main()
