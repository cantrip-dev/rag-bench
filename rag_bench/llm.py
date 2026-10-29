"""A chat LLM behind an API: any OpenAI-compatible /chat/completions endpoint, or Anthropic's /v1/messages."""

from __future__ import annotations

from dataclasses import dataclass

from .http import bearer, request


@dataclass
class LLM:
    model: str
    base_url: str = "https://api.openai.com/v1"
    key: str = ""
    provider: str = "openai"  # "openai" covers every OpenAI-compatible API; or "anthropic"
    max_tokens: int = 512
    tokens_in: int = 0
    tokens_out: int = 0

    def complete(self, prompt: str) -> str:
        base = self.base_url.rstrip("/")
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        if self.provider == "anthropic":
            data = request(base + "/v1/messages", body, {"x-api-key": self.key, "anthropic-version": "2023-06-01"})
            usage = data.get("usage") or {}
            self.tokens_in += usage.get("input_tokens", 0)
            self.tokens_out += usage.get("output_tokens", 0)
            return "".join(b.get("text", "") for b in data["content"])
        data = request(base + "/chat/completions", body, bearer(self.key))
        usage = data.get("usage") or {}
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"] or ""
