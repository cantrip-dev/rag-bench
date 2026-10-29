"""rag-bench run CONFIG.toml  |  rag-bench datasets"""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

from . import __version__, bench, datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-bench", description=__doc__)
    parser.add_argument("--version", action="version", version=f"rag-bench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a benchmark from a config file")
    run.add_argument("config", type=Path)
    run.add_argument("--out", type=Path, help="where to save the full results (default: results/<time>.json)")
    sub.add_parser("datasets", help="list the demo datasets")
    args = parser.parse_args(argv)

    if args.command == "datasets":
        for name, about in datasets.DATASETS.items():
            print(f"{name:10} {about}")
        return 0

    result = bench.run(tomllib.loads(args.config.read_text()))
    out = args.out or Path("results") / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(summary(result))
    print(f"\nFull results, with every question and answer: {out}")
    return 0


def summary(r: dict) -> str:
    run = r["run"]
    lines = [f"\n{run['dataset']}: {run['questions']} questions, {run['chunks']} chunks, k={run['k']}",
             f"  embedder  {run['embedder']} ({r['embedder']['dim']} dims)",
             f"  db        {run['db']} (index: {run['db_index']})"]
    if run["llm"]:
        lines.append(f"  llm       {run['llm']}, judged by {run['judge']}")
    lines.append("\nRetrieval   " + "  ".join(f"{k} {v:.3f}" for k, v in r["retrieval"].items()))
    db = r["db"]
    ann = next(k for k in db if k.startswith("ann_recall"))
    lines.append(f"DB          {ann} {db[ann]:.3f}  search p50 {db['search_ms']['p50']} ms"
                 f"  p95 {db['search_ms']['p95']} ms  load {db['load_s']} s")
    lines.append("            queries/s " + "  ".join(f"{w} parallel: {q}" for w, q in db["queries_per_s"].items()))
    e = r["embedder"]
    lines.append(f"Embedder    query p50 {e['query_embed_ms']['p50']} ms  corpus "
                 f"{'cached' if e['corpus_cached'] else str(e['corpus_embed_s']) + ' s'}"
                 f"  tokens {e['corpus_tokens'] + e['query_tokens']}")
    if a := r["answers"]:
        correct = "n/a (no reference answers)" if a["correctness"] is None else f"{a['correctness']:.3f}"
        lines.append(f"Answers     faithfulness {a['faithfulness']:.3f}  correctness {correct}"
                     f"  llm p50 {a['llm_ms']['p50']} ms  tokens {a['llm_tokens']['in']} in / {a['llm_tokens']['out']} out")
    return "\n".join(lines)
