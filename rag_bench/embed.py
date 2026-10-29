"""Embeddings from any OpenAI-compatible /embeddings endpoint: OpenAI, Ollama, vLLM, LM Studio, and others."""

from __future__ import annotations

from dataclasses import dataclass

from .http import bearer, request


@dataclass
class Embedder:
    model: str
    base_url: str = "https://api.openai.com/v1"
    key: str = ""
    batch_size: int = 64
    tokens: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            data = request(self.base_url.rstrip("/") + "/embeddings",
                           {"model": self.model, "input": texts[i:i + self.batch_size]}, bearer(self.key))
            self.tokens += (data.get("usage") or {}).get("total_tokens", 0)
            vectors += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return vectors
