"""One JSON-over-HTTP helper, shared by the embedder, the LLM and the Qdrant store."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class HTTPError(Exception):
    pass


def request(url: str, body: dict | None = None, headers: dict | None = None,
            method: str = "POST", timeout: float = 120, tries: int = 3) -> dict:
    """Send JSON, return JSON. Retries rate limits, server errors and dropped connections."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "User-Agent": "rag-bench",
                                          **(headers or {})})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == tries - 1:
                raise HTTPError(f"{method} {url} -> {e.code}: {e.read()[:300].decode(errors='replace')}") from None
        except urllib.error.URLError as e:
            if attempt == tries - 1:
                raise HTTPError(f"{method} {url} failed: {e.reason}") from None
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}
