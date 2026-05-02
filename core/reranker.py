from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from FlagEmbedding import LayerWiseFlagLLMReranker
from torch.amp import autocast
import math


class BGEReranker:
    """Standard cross-encoder reranker (lighter, faster)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    def rerank(self, query: str, docs: list, top_k: int = 2) -> list:
        pairs = [(query, doc.page_content) for doc in docs]
        inputs = self.tokenizer(
            pairs, padding="longest", truncation=True, return_tensors="pt", max_length=512
        ).to(self.device)
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.no_grad(), autocast(device_type=device_type):
            scores = self.model(**inputs).logits.squeeze(-1)
        sorted_indices = torch.argsort(scores, descending=True)
        return [docs[i] for i in sorted_indices[:top_k].cpu().numpy()]


class BGEReranker_v2:
    """LayerWise LLM-based reranker — higher accuracy, GPU recommended."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-minicpm-layerwise",
        use_fp16: bool = True,
        cutoff_layers: list = None,
        batch_size: int = 16,
    ):
        self.cutoff_layers = cutoff_layers or [28]
        self.batch_size = batch_size
        self.reranker = LayerWiseFlagLLMReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, docs: list, top_k: int = 3) -> list:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(
            pairs, cutoff_layers=self.cutoff_layers, batch_size=self.batch_size
        )
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [docs[i] for i in sorted_indices[:top_k]]

    def rerank_with_scores(self, query: str, docs: list, top_k: int = 3) -> list:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.compute_score(
            pairs, cutoff_layers=self.cutoff_layers, batch_size=self.batch_size
        )
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [{"doc": docs[i], "score": scores[i]} for i in sorted_indices[:top_k]]

    def rerank_with_threshold(
        self, query: str, docs: list, top_k: int = 3, threshold: float = 0.3
    ) -> list:
        if not docs:
            return []
        pairs = [[query, doc.page_content] for doc in docs]
        raw_scores = self.reranker.compute_score(
            pairs, cutoff_layers=self.cutoff_layers, batch_size=self.batch_size
        )

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        scored = [(docs[i], sigmoid(raw_scores[i])) for i in range(len(docs))]
        filtered = [(doc, s) for doc, s in scored if s >= threshold]
        if not filtered:
            return []
        filtered.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in filtered[:top_k]]
