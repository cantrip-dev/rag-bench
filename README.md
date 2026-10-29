# rag-bench

Benchmark a plain RAG pipeline with **your own vector database, embedding model and LLM**, on demo data
where the right answers are known. No framework: the whole pipeline is a few hundred lines you can read.

```
split docs -> embed -> store in your DB -> embed the question -> search top k -> prompt your LLM -> judge the answer
```

Swap one piece per run and keep the rest fixed, and you get a clean comparison: same data, same DB, same
LLM, two embedding models.

## Quick start

```sh
pip install -e .                       # no dependencies; add [postgres] for pgvector
export OPENAI_API_KEY=...              # or point everything at a local Ollama: see examples/
rag-bench run examples/tiny-memory.toml
```

```
tiny: 8 questions, 10 chunks, k=3
Retrieval   recall@3 ...  ndcg@3 ...  mrr@3 ...  hit@3 ...
DB          ann_recall@3 ...  search p50 ... ms  p95 ... ms  load ... s
Embedder    query p50 ... ms  corpus ... s  tokens ...
Answers     faithfulness ...  correctness ...  llm p50 ... ms  tokens ... in / ... out
```

The full results, with every question, what was retrieved and the answer, are saved to `results/`.

## What it measures

The embedding model decides **what counts as relevant**; the database decides **how fast and how faithfully
it finds it**. rag-bench scores them separately, so you can tell whose fault a bad number is.

| | Metric | What it tells you |
|---|---|---|
| Retrieval | `recall@k` | Share of the right docs that made the top k. The headline number for RAG. |
| | `ndcg@k`, `mrr@k` | Whether the right docs are near the top, not just somewhere in it. |
| | `hit@k` | Share of questions with at least one right doc in the top k. |
| Database | `ann_recall@k` | How many of the true nearest neighbours (found by brute force on the same vectors) the DB's index returned. 1.0 = no loss from approximate search. |
| | search p50 / p95 / p99 | Search latency, without the embedding call. |
| | queries/s | Throughput at each level of parallel searches you set. |
| | load time | Time to insert every chunk, index included. |
| Embedder | query p50 / p95 / p99 | Time to embed one question. |
| | tokens, dimensions | What the corpus and questions cost to embed. |
| Answers | faithfulness | Share of answers fully supported by the retrieved context (no made-up claims). |
| | correctness | Share of answers that agree with the reference answer, on datasets that have one. |
| | LLM latency, tokens | Per answer, and in total. |

Answers are graded by an LLM judge. It defaults to your LLM; give it its own `[judge]` section so that
testing a weaker model doesn't also make the grading softer.

## Pick the pieces

**Database**, by URL. Each run creates its own `rag_bench_<id>` table or collection and drops it at the
end; nothing else in your database is read or changed.

| URL | |
|---|---|
| `memory://` | Exact search in Python. No setup; slow above ~50K chunks. |
| `postgresql://user:pass@host:5432/db` | Postgres with pgvector, including Neon (`pip install -e ".[postgres]"`). HNSW index up to 2,000 dimensions. |
| `qdrant://host:6333` | Qdrant. `qdrants://:API_KEY@host:6333` for HTTPS with an API key. |

**Neon** works as Postgres: in the Neon console press Connect, copy the connection string (it ends in
`?sslmode=require`), and use it as the URL. pgvector comes with Neon; the run turns it on if it isn't yet.
The pooled (`-pooler`) and direct strings both work. Two things to know: search times include the round
trip from your machine to Neon's region, so run near it for numbers you'll compare; and the first run on
a suspended database waits for it to wake. See `examples/neon.toml`.

Put a URL with a password in an environment variable and name it with `url_env`, so it stays out of
the config file. Passwords are never printed or saved in results.

**Embedding model**: any OpenAI-compatible `/embeddings` API (OpenAI, Ollama, vLLM, LM Studio, and
others). Embeddings of the corpus are saved in `.rag-bench/`, so comparing databases doesn't pay for them twice.

**LLM**: any OpenAI-compatible `/chat/completions` API, or `provider = "anthropic"`. Leave out `[llm]`
to measure retrieval only.

API keys are read only from the environment variable named in `key_env`.

**Dataset** (`rag-bench datasets`):

| Name | |
|---|---|
| `tiny` | 10 made-up handbook pages, 8 questions with reference answers. Runs in seconds. |
| `scifact` | BEIR SciFact: 5,183 scientific abstracts, 300 claims. |
| `nfcorpus` | BEIR NFCorpus: 3,633 medical articles, 323 health questions. |
| `fiqa` | BEIR FiQA: 57,638 finance forum posts, 648 questions. |

BEIR sets are downloaded on first use from the [BEIR project](https://github.com/beir-cellar/beir), each
under its own license. They list the right documents but not reference answers, so on them correctness
is reported as n/a; faithfulness is still scored.

## Config

```toml
dataset = "scifact"
questions = 100        # a fixed sample, so runs compare; 0 = all
k = 10
chunk_size = 1000      # characters
chunk_overlap = 100
concurrency = [1, 8]

[db]
url_env = "RAG_BENCH_DB_URL"

[embedder]
model = "nomic-embed-text"
base_url = "http://localhost:11434/v1"

[llm]
model = "llama3.2"
base_url = "http://localhost:11434/v1"
# key_env = "MY_KEY_VAR"   for hosted APIs

# [judge]               optional, same fields as [llm]
```

More in `examples/`.

## How the code is laid out

| File | |
|---|---|
| `rag_bench/rag.py` | The pipeline: chunking, the answer prompt, the judge prompts. |
| `rag_bench/stores.py` | One small class per database: `create`, `add`, `search`, `drop`. Add a database here. |
| `rag_bench/embed.py`, `llm.py` | The API clients. |
| `rag_bench/datasets.py` | Demo data. |
| `rag_bench/metrics.py` | The scores. |
| `rag_bench/bench.py` | One run, start to finish. |

## Tests

```sh
pip install -e ".[dev,postgres]" && pytest -q
```

Tests use a local fake API, so they're free and offline. To run them against real databases too, set
`RAG_BENCH_TEST_PG_URL` and `RAG_BENCH_TEST_QDRANT_URL` (CI does, with both as services).

## Contributing

Sign off your commits (`git commit -s`). Keep the core free of dependencies: a new database's driver
goes in an optional extra.

## License

Apache-2.0
