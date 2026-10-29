"""Vector stores, opened from a URL. Each run gets its own throwaway table or collection, dropped at the end,
so rag-bench never reads or changes anything else in your database.

    memory://                              exact search in this process (also the baseline for ANN recall)
    postgresql://user:pass@host:5432/db    Postgres with pgvector, e.g. Neon   (pip install "rag-bench[postgres]")
    qdrant://host:6333                     Qdrant over HTTP; qdrants:// for HTTPS, an API key as the password
"""

from __future__ import annotations

import heapq
import math
import threading
from urllib.parse import urlsplit

from .http import request


def open_store(url: str, name: str):
    scheme = urlsplit(url).scheme
    if scheme == "memory":
        return MemoryStore()
    if scheme in ("postgres", "postgresql"):
        return PgVectorStore(url, name)
    if scheme in ("qdrant", "qdrants"):
        return QdrantStore(url, name)
    raise ValueError(f"Unknown database URL scheme '{scheme}://': use memory://, postgresql:// or qdrant://")


def redact(url: str) -> str:
    """The URL without its password, safe to print and save."""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.netloc.rpartition("@")[2]
    user = f"{parts.username}:***@" if parts.username else "***@"
    return parts._replace(netloc=user + host).geturl()


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class MemoryStore:
    index = "exact"

    def create(self, dim: int) -> None:
        self.ids: list[str] = []
        self.vectors: list[list[float]] = []

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        self.ids += ids
        self.vectors += [_unit(v) for v in vectors]

    def search(self, vector: list[float], k: int) -> list[str]:
        q = _unit(vector)
        scores = ((sum(map(float.__mul__, q, v)), i) for i, v in enumerate(self.vectors))
        return [self.ids[i] for _, i in heapq.nlargest(k, scores)]

    def drop(self) -> None:
        self.ids, self.vectors = [], []


class PgVectorStore:
    MAX_INDEX_DIM = 2000  # pgvector's HNSW index takes vectors up to 2000 dimensions

    def __init__(self, url: str, name: str):
        try:
            import psycopg
        except ImportError:
            raise SystemExit('Postgres needs its driver: pip install "rag-bench[postgres]"') from None
        self.psycopg, self.url, self.table = psycopg, url, name
        self.local, self.conns = threading.local(), []

    def _conn(self):
        if not hasattr(self.local, "conn"):  # one connection per thread, for the concurrency test
            # No server-side prepared statements, so pooled endpoints (Neon's -pooler, PgBouncer) work too.
            self.local.conn = self.psycopg.connect(self.url, autocommit=True, prepare_threshold=None)
            self.conns.append(self.local.conn)
        return self.local.conn

    def create(self, dim: int) -> None:
        c = self._conn()
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute(f"CREATE TABLE {self.table} (id text PRIMARY KEY, embedding vector({dim}))")
        if dim <= self.MAX_INDEX_DIM:
            c.execute(f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_cosine_ops)")
            self.index = "hnsw"
        else:
            self.index = f"none (pgvector indexes up to {self.MAX_INDEX_DIM} dims, so this is exact search)"

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        with self._conn().cursor() as cur:
            cur.executemany(f"INSERT INTO {self.table} (id, embedding) VALUES (%s, %s::vector)",
                            [(i, str(v)) for i, v in zip(ids, vectors)])

    def search(self, vector: list[float], k: int) -> list[str]:
        rows = self._conn().execute(f"SELECT id FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s",
                                    (str(vector), k)).fetchall()
        return [r[0] for r in rows]

    def drop(self) -> None:
        self._conn().execute(f"DROP TABLE IF EXISTS {self.table}")
        for c in self.conns:
            c.close()


class QdrantStore:
    index = "hnsw"

    def __init__(self, url: str, name: str):
        parts = urlsplit(url)
        scheme = "https" if parts.scheme == "qdrants" else "http"
        self.base = f"{scheme}://{parts.hostname}:{parts.port or 6333}/collections/{name}"
        self.headers = {"api-key": parts.password} if parts.password else {}

    def _call(self, path: str, body: dict | None = None, method: str = "POST") -> dict:
        return request(self.base + path, body, self.headers, method=method)

    def create(self, dim: int) -> None:
        self._call("", {"vectors": {"size": dim, "distance": "Cosine"}}, "PUT")
        self.ids: list[str] = []

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        for start in range(0, len(ids), 256):
            points = [{"id": len(self.ids) + j, "vector": v}
                      for j, v in enumerate(vectors[start:start + 256])]
            self._call("/points?wait=true", {"points": points}, "PUT")
            self.ids += ids[start:start + 256]

    def search(self, vector: list[float], k: int) -> list[str]:
        hits = self._call("/points/search", {"vector": vector, "limit": k})["result"]
        return [self.ids[h["id"]] for h in hits]

    def drop(self) -> None:
        self._call("", method="DELETE")
