"""Vector stores, opened from a URL. A database server gets its own throwaway table each run, dropped at the end,
so rag-bench never reads or changes anything else in your database.

    faiss://                               FAISS HNSW index in this process; faiss://flat for exact search
                                           (pip install "rag-bench[faiss]")
    postgresql://user:pass@host:5432/db    Postgres with pgvector, e.g. Neon   (pip install "rag-bench[postgres]")
    memory://                              exact search in plain Python, no installs (also the baseline for ANN recall)
"""

from __future__ import annotations

import heapq
import math
import threading
from urllib.parse import urlsplit


def open_store(url: str, name: str):
    scheme = urlsplit(url).scheme
    if scheme == "memory":
        return MemoryStore()
    if scheme in ("postgres", "postgresql"):
        return PgVectorStore(url, name)
    if scheme == "faiss":
        return FaissStore(urlsplit(url).netloc or "hnsw")
    raise ValueError(f"Unknown database URL scheme '{scheme}://': use faiss://, postgresql:// or memory://")


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


class FaissStore:
    def __init__(self, kind: str):
        try:
            import faiss
            import numpy
        except ImportError:
            raise SystemExit('FAISS needs its library: pip install "rag-bench[faiss]"') from None
        if kind not in ("hnsw", "flat"):
            raise ValueError(f"Unknown FAISS index '{kind}': use faiss:// (HNSW) or faiss://flat (exact)")
        self.faiss, self.numpy, self.kind = faiss, numpy, kind
        self.index = "hnsw" if kind == "hnsw" else "exact (flat)"

    def _unit(self, vectors: list[list[float]]):
        a = self.numpy.asarray(vectors, dtype="float32")
        self.faiss.normalize_L2(a)  # unit vectors, so inner product = cosine similarity
        return a

    def create(self, dim: int) -> None:
        ip = self.faiss.METRIC_INNER_PRODUCT
        self.ix = self.faiss.IndexHNSWFlat(dim, 32, ip) if self.kind == "hnsw" else self.faiss.IndexFlatIP(dim)
        self.ids: list[str] = []

    def add(self, ids: list[str], vectors: list[list[float]]) -> None:
        self.ix.add(self._unit(vectors))
        self.ids += ids

    def search(self, vector: list[float], k: int) -> list[str]:
        _, found = self.ix.search(self._unit([vector]), k)
        return [self.ids[i] for i in found[0] if i >= 0]

    def drop(self) -> None:
        self.ix, self.ids = None, []


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
