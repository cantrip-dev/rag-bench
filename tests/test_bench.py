import json
import os

import pytest

from rag_bench import cli, metrics, rag
from rag_bench.bench import run
from rag_bench.stores import MemoryStore, open_store, redact


def config(api, db="memory://", **extra):
    return {"dataset": "tiny", "k": 3, "concurrency": [1, 4], "db": {"url": db},
            "embedder": {"model": "fake-embed", "base_url": api},
            "llm": {"model": "fake-chat", "base_url": api}, **extra}


def test_metrics():
    ranked, relevant = ["a", "b", "c"], {"b", "z"}
    assert metrics.recall(ranked, relevant, 3) == 0.5
    assert metrics.hit(ranked, relevant, 1) == 0.0
    assert metrics.mrr(ranked, relevant, 3) == 0.5
    assert metrics.ndcg(["b", "z"], relevant, 2) == 1.0
    assert 0 < metrics.ndcg(ranked, relevant, 3) < 1
    assert metrics.percentiles([float(i) for i in range(1, 101)])["p50"] == 51.0


def test_chunk_covers_the_text_with_overlap():
    text = "".join(chr(97 + i % 26) for i in range(2500))
    pieces = rag.chunk(text, 1000, 100)
    assert [len(p) for p in pieces] == [1000, 1000, 700]
    assert pieces[0][-100:] == pieces[1][:100]
    assert rag.chunk("short", 1000, 100) == ["short"]


def test_docs_in_order_dedupes_chunks():
    assert rag.docs_in_order(["a#0", "b#2", "a#1", "c#0"]) == ["a", "b", "c"]


def test_memory_store_finds_nearest():
    s = MemoryStore()
    s.create(2)
    s.add(["x", "y", "z"], [[1, 0], [0, 1], [1, 1]])
    assert s.search([1, 0.1], 2) == ["x", "z"]


def test_redact_hides_passwords_only():
    assert redact("postgresql://bob:s3cret@db:5432/app") == "postgresql://bob:***@db:5432/app"
    assert redact("postgresql://:pw@db/app") == "postgresql://***@db/app"
    assert redact("faiss://") == "faiss://"
    assert redact("postgresql://bob:pw@/db?host=/run/pg") == "postgresql://bob:***@/db?host=/run/pg"


def test_unknown_db_scheme():
    with pytest.raises(ValueError, match="mysql"):
        open_store("mysql://x", "t")


def test_full_run_on_tiny(api, tmp_path):
    r = run(config(api), cache=tmp_path)
    assert r["run"]["questions"] == 8 and r["run"]["db"] == "memory://"
    assert r["retrieval"]["hit@3"] >= 0.75  # word-overlap embeddings still find most answers
    assert r["db"]["ann_recall@3"] == 1.0  # memory:// is exact search
    assert set(r["db"]["queries_per_s"]) == {"1", "4"}
    assert r["answers"]["faithfulness"] == 1.0 and r["answers"]["correctness"] == 1.0
    assert all(q["answer"] for q in r["per_question"])
    assert r["embedder"]["corpus_cached"] is False

    again = run(config(api), cache=tmp_path)
    assert again["embedder"]["corpus_cached"] is True and again["embedder"]["corpus_tokens"] == 0
    assert again["retrieval"] == r["retrieval"]


def test_retrieval_only_without_llm(api, tmp_path):
    cfg = config(api)
    del cfg["llm"]
    r = run(cfg, cache=tmp_path)
    assert r["answers"] is None and "answer" not in r["per_question"][0]


def test_separate_judge_on_anthropic_api(api, tmp_path):
    r = run(config(api, judge={"model": "fake-judge", "base_url": api, "provider": "anthropic"}), cache=tmp_path)
    assert r["run"]["judge"] == "fake-judge"
    assert r["answers"]["judge_tokens"]["in"] > 0


def test_keys_come_from_the_environment(api, tmp_path, monkeypatch):
    cfg = config(api)
    cfg["embedder"]["key_env"] = "RAG_BENCH_TEST_MISSING_KEY"
    monkeypatch.delenv("RAG_BENCH_TEST_MISSING_KEY", raising=False)
    with pytest.raises(SystemExit, match="RAG_BENCH_TEST_MISSING_KEY"):
        run(cfg, cache=tmp_path)


def test_cli_writes_results(api, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.toml").write_text(
        f'dataset = "tiny"\nk = 3\n[embedder]\nmodel = "e"\nbase_url = "{api}"\n'
        f'[llm]\nmodel = "m"\nbase_url = "{api}"\n')
    assert cli.main(["run", "c.toml", "--out", "out.json"]) == 0
    assert json.loads((tmp_path / "out.json").read_text())["run"]["dataset"] == "tiny"
    assert "faithfulness 1.000" in capsys.readouterr().out


@pytest.mark.parametrize("url, index", [("faiss://", "hnsw"), ("faiss://flat", "exact (flat)")])
def test_faiss(api, tmp_path, url, index):
    pytest.importorskip("faiss")
    r = run(config(api, db=url), cache=tmp_path)
    assert r["run"]["db_index"] == index
    assert r["db"]["ann_recall@3"] >= 0.9 and r["retrieval"]["hit@3"] >= 0.75


def test_unknown_faiss_index():
    pytest.importorskip("faiss")
    with pytest.raises(ValueError, match="ivf"):
        open_store("faiss://ivf", "t")


# A real Postgres with pgvector: set this to run against it (CI starts one as a service).
def test_pgvector(api, tmp_path):
    url = os.environ.get("RAG_BENCH_TEST_PG_URL") or pytest.skip("RAG_BENCH_TEST_PG_URL not set")
    r = run(config(api, db=url), cache=tmp_path)
    assert r["db"]["ann_recall@3"] >= 0.9 and r["retrieval"]["hit@3"] >= 0.75
    assert r["run"]["db_index"] == "hnsw"
    store = open_store(url, "rag_bench_leftover_check")  # the run's table is gone
    rows = store._conn().execute("SELECT count(*) FROM pg_tables WHERE tablename LIKE 'rag_bench_%'").fetchone()
    assert rows[0] == 0
