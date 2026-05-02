"""Build FAISS vector indices from FAQ text files.

Usage
-----
    python scripts/build_index.py                      # build all knowledge bases
    python scripts/build_index.py --kb traditional_chinese  # build one
    python scripts/build_index.py --skip-services      # skip web-page scraping
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import yaml
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.config import load_settings

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9",
    "Referer": "https://www.google.com/",
}

# Maps knowledge-base index_path suffix → (faq_dir, url_prefix, service_lang_prefix)
_LANG_MAP = {
    "tra_chi": ("FAQ_data/url_txt_tra_chi", "https://www.net-chinese.com.tw/faq", ""),
    "sim_chi": ("FAQ_data/url_txt_sim_chi", "https://www.net-chinese.com.tw/cn/faq", "cn"),
    "en":      ("FAQ_data/url_txt_en",      "https://www.net-chinese.com.tw/en/faq", "en"),
}


def load_categories(path: str = "data/faq_categories.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_faq_docs(data_dir: str, categories: dict, url_prefix: str) -> list[Document]:
    docs = []
    for filepath in glob.glob(f"{data_dir}/*/*/*.txt"):
        label_name = Path(filepath).parent.name
        filename = Path(filepath).stem

        if label_name not in categories.get("categories", {}):
            print(f"  ⚠ Unknown category '{label_name}', skipping {filepath}")
            continue

        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        anchor = None
        m = re.search(r"__anchor_(.+)$", filename)
        if m:
            anchor = m.group(1)
        else:
            m2 = re.search(r"ANCHOR:(\w+)", content)
            if m2:
                anchor = m2.group(1)

        clean_content = re.sub(r"^ANCHOR:.*\n?", "", content)
        meta = categories["categories"][label_name].copy()
        clean_fn = filename.replace(f"__anchor_{anchor or ''}", "").replace("index", "")

        if meta.get("group") == "else":
            source = None
        else:
            source = f"{url_prefix}/{meta['group']}/{label_name}/{clean_fn}"
            if anchor:
                source += f"#{anchor}"
        meta["source"] = source

        full_text = (
            f"[主題標籤: {meta.get('label')}]\n"
            f"[說明: {meta.get('description')}]\n\n"
            f"{clean_content}"
        )
        docs.append(Document(page_content=full_text, metadata=meta))
    return docs


def _build_service_docs(services: dict, lang_prefix: str, base_domain: str) -> list[Document]:
    docs = []
    for key, item in services.items():
        url = item["url"]
        if base_domain in url and "HtmlSiteInfoList" not in url and lang_prefix:
            path_part = url.split(base_domain)[-1]
            url = f"{base_domain}/{lang_prefix}{path_part}"
        print(f"  Scraping: {url}")
        try:
            content = WebBaseLoader(url, header_template=_HEADERS).load()
            meta = {**item, "source": url}
            full_text = (
                f"[主題標籤: {meta.get('label')}]\n"
                f"[說明: {meta.get('description')}]\n\n"
                f"{content}"
            )
            docs.append(Document(page_content=full_text, metadata=meta))
        except Exception as e:
            print(f"  ⚠ Failed to load {url}: {e}")
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS indices for Pluse RAG")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--categories", default="data/faq_categories.yaml")
    parser.add_argument("--kb", default=None, help="Build only this knowledge base (by name)")
    parser.add_argument("--skip-services", action="store_true", help="Skip web-page scraping")
    args = parser.parse_args()

    settings = load_settings(args.config)
    categories = load_categories(args.categories)

    print("Loading embedding model…")
    embedding = HuggingFaceEmbeddings(model_name=settings.embeddings.model)

    for kb in settings.knowledge_bases:
        if args.kb and kb.name != args.kb:
            continue

        # Resolve lang dir from index_path suffix
        lang_key = next(
            (k for k in _LANG_MAP if kb.index_path.endswith(k)), None
        )
        if not lang_key:
            print(f"\n⚠ Cannot map '{kb.index_path}' to a language dir — skipping")
            continue

        data_dir, url_prefix, svc_lang = _LANG_MAP[lang_key]
        print(f"\n{'='*60}")
        print(f"Building: {kb.name}  →  {kb.index_path}")

        docs = _build_faq_docs(data_dir, categories, url_prefix)
        print(f"  FAQ docs: {len(docs)}")

        if not args.skip_services and "services" in categories:
            svc_docs = _build_service_docs(
                categories["services"], svc_lang, "https://www.net-chinese.com.tw"
            )
            docs.extend(svc_docs)
            print(f"  Service docs: {len(svc_docs)}")

        print(f"  Building FAISS index ({len(docs)} total docs)…")
        vectorstore = FAISS.from_documents(
            docs, embedding, distance_strategy=DistanceStrategy.COSINE
        )
        vectorstore.save_local(kb.index_path)
        print(f"  ✅ Saved → {kb.index_path}")


if __name__ == "__main__":
    main()
