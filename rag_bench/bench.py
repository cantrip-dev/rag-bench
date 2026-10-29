"""One benchmark run: a dataset, an embedder, a vector DB and (optionally) an LLM, all read from a config."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import struct
import time
import uuid
from array import array
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, datasets, metrics, rag
from .embed import Embedder
from .llm import LLM
from .stores import MemoryStore, open_store, redact


def run(cfg: dict, cache: Path = Path(".rag-bench")) -> dict:
    k = cfg.get("k", 10)
    size, overlap = cfg.get("chunk_size", 1000), cfg.get("chunk_overlap", 100)
    data = datasets.load(cfg.get("dataset", "tiny"), cache / "datasets")
    db_url = _db_url(cfg.get("db", {}))
    embedder = Embedder(**_api(cfg["embedder"]), batch_size=cfg["embedder"].get("batch_size", 64))
    llm = LLM(**_api(cfg["llm"]), provider=cfg["llm"].get("provider", "openai")) if "llm" in cfg else None
    judge = (LLM(**_api(cfg["judge"]), provider=cfg["judge"].get("provider", "openai"))
             if "judge" in cfg else llm)

    qids = sorted(data.questions)
    if 0 < cfg.get("questions", 100) < len(qids):
        qids = sorted(random.Random(0).sample(qids, cfg["questions"]))

    chunk_ids, texts = [], {}
    for doc_id, text in data.docs.items():
        for j, piece in enumerate(rag.chunk(text, size, overlap)):
            if piece.strip():
                chunk_ids.append(f"{doc_id}#{j}")
                texts[chunk_ids[-1]] = piece

    key = f"{embedder.base_url}|{embedder.model}|{data.name}|{size}|{overlap}"
    cache_file = cache / "embeddings" / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".f32")
    t = time.perf_counter()
    vectors, cached = _embed_corpus(embedder, [texts[c] for c in chunk_ids], cache_file)
    embed_s, corpus_tokens = time.perf_counter() - t, embedder.tokens
    dim = len(vectors[0])

    exact = MemoryStore()  # brute-force search over the same vectors: the truth the DB's index is checked against
    exact.create(dim)
    exact.add(chunk_ids, vectors)

    store = open_store(db_url, "rag_bench_" + uuid.uuid4().hex[:8])
    rows, query_vectors = [], []
    try:
        print(f"Loading into {redact(db_url)}...")
        t = time.perf_counter()
        store.create(dim)
        store.add(chunk_ids, vectors)
        load_s = time.perf_counter() - t

        print(f"Asking {len(qids)} questions{' and judging the answers' if llm else ''}...")
        for qid in qids:
            question, relevant = data.questions[qid], data.relevant[qid]
            t = time.perf_counter()
            qv = embedder.embed([question])[0]
            t2 = time.perf_counter()
            got = store.search(qv, k)
            t3 = time.perf_counter()
            truth = exact.search(qv, k)
            docs = rag.docs_in_order(got)
            row = {"id": qid, "question": question, "retrieved": docs[:k],
                   "embed_ms": (t2 - t) * 1000, "search_ms": (t3 - t2) * 1000,
                   "ann_recall": len(set(got) & set(truth)) / max(1, len(truth)),
                   **{name: fn(docs, relevant, k) for name, fn in
                      (("recall", metrics.recall), ("ndcg", metrics.ndcg), ("mrr", metrics.mrr), ("hit", metrics.hit))}}
            if llm:
                contexts = [texts[c] for c in got]
                t = time.perf_counter()
                row["answer"] = rag.answer(llm, question, contexts)
                row["llm_ms"] = (time.perf_counter() - t) * 1000
                row["faithful"] = rag.is_faithful(judge, row["answer"], contexts)
                if qid in data.answers:
                    row["reference"] = data.answers[qid]
                    row["correct"] = rag.is_correct(judge, question, row["answer"], data.answers[qid])
            rows.append(row)
            query_vectors.append(qv)

        throughput = {}
        searches = query_vectors * max(1, 200 // len(query_vectors))  # at least ~200 searches per level
        for workers in cfg.get("concurrency", [1, 8]):
            print(f"Measuring throughput with {workers} parallel searches...")
            with ThreadPoolExecutor(workers) as pool:
                list(pool.map(lambda v: store.search(v, k), searches[:workers]))  # untimed: opens each connection
                t = time.perf_counter()
                list(pool.map(lambda v: store.search(v, k), searches))
                throughput[str(workers)] = round(len(searches) / (time.perf_counter() - t), 1)
    finally:
        store.drop()

    col = lambda name: [r[name] for r in rows if name in r]
    result = {
        "run": {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rag_bench": __version__,
                "dataset": data.name, "docs": len(data.docs), "chunks": len(chunk_ids), "questions": len(rows),
                "k": k, "chunk_size": size, "chunk_overlap": overlap,
                "db": redact(db_url), "db_index": getattr(store, "index", "unknown"),
                "embedder": embedder.model, "llm": llm and llm.model, "judge": judge and judge.model,
                "python": platform.python_version(), "machine": platform.platform()},
        "retrieval": {f"{name}@{k}": metrics.mean(col(name)) for name in ("recall", "ndcg", "mrr", "hit")},
        "db": {f"ann_recall@{k}": metrics.mean(col("ann_recall")), "search_ms": metrics.percentiles(col("search_ms")),
               "load_s": round(load_s, 2), "queries_per_s": throughput},
        "embedder": {"dim": dim, "corpus_embed_s": round(embed_s, 2), "corpus_cached": cached,
                     "corpus_tokens": corpus_tokens, "query_tokens": embedder.tokens - corpus_tokens,
                     "query_embed_ms": metrics.percentiles(col("embed_ms"))},
        "answers": None,
        "per_question": rows,
    }
    if llm:
        result["answers"] = {"faithfulness": metrics.mean(col("faithful")),
                             "correctness": metrics.mean(col("correct")) if col("correct") else None,
                             "llm_ms": metrics.percentiles(col("llm_ms")),
                             "llm_tokens": {"in": llm.tokens_in, "out": llm.tokens_out}}
        if judge is not llm:
            result["answers"]["judge_tokens"] = {"in": judge.tokens_in, "out": judge.tokens_out}
    return result


def _api(section: dict) -> dict:
    """model / base_url from the config; the key only ever from the environment variable it names."""
    out = {"model": section["model"], "key": ""}
    if "base_url" in section:
        out["base_url"] = section["base_url"]
    if env := section.get("key_env"):
        if not os.environ.get(env):
            raise SystemExit(f"Set the environment variable {env} to your API key.")
        out["key"] = os.environ[env]
    return out


def _db_url(section: dict) -> str:
    if env := section.get("url_env"):
        if not os.environ.get(env):
            raise SystemExit(f"Set the environment variable {env} to your database URL.")
        return os.environ[env]
    return section.get("url", "memory://")


def _embed_corpus(embedder: Embedder, texts: list[str], path: Path) -> tuple[list[list[float]], bool]:
    """Embed once per model + dataset + chunking, then reuse: comparing DBs shouldn't re-pay for embeddings."""
    if path.exists():
        raw = path.read_bytes()
        (dim,) = struct.unpack("<i", raw[:4])
        flat = array("f")
        flat.frombytes(raw[4:])
        if len(flat) == dim * len(texts):
            return [flat[i:i + dim].tolist() for i in range(0, len(flat), dim)], True
    print(f"Embedding {len(texts)} chunks with {embedder.model} (saved for later runs)...")
    vectors = embedder.embed(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<i", len(vectors[0])) + array("f", (x for v in vectors for x in v)).tobytes())
    return vectors, False
