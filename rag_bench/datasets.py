"""Demo datasets: questions with known relevant documents, so retrieval can be scored.

`tiny` ships with rag-bench and has reference answers. The others are BEIR collections
(https://github.com/beir-cellar/beir), downloaded on first use; each carries its own license.
BEIR has no reference answers, so on those answer correctness is not scored (faithfulness still is).
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

DATASETS = {
    "tiny": "10 made-up handbook pages, 8 questions with reference answers. Runs in seconds.",
    "scifact": "BEIR SciFact: about 5K scientific abstracts, 300 claims to check.",
    "nfcorpus": "BEIR NFCorpus: about 3.6K medical articles, 323 health questions.",
    "fiqa": "BEIR FiQA: about 57K finance forum posts, 648 finance questions.",
}


@dataclass
class Dataset:
    name: str
    docs: dict[str, str]             # doc id -> text
    questions: dict[str, str]        # question id -> text
    relevant: dict[str, set[str]]    # question id -> ids of the docs that answer it
    answers: dict[str, str] = field(default_factory=dict)  # question id -> reference answer, if known


def load(name: str, cache: Path) -> Dataset:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose one of: {', '.join(DATASETS)}")
    if name == "tiny":
        raw = json.loads(resources.files("rag_bench").joinpath("data/tiny.json").read_text())
        return Dataset(name, raw["docs"], raw["questions"],
                       {q: set(ids) for q, ids in raw["relevant"].items()}, raw["answers"])
    folder = cache / name
    if not folder.exists():
        print(f"Downloading {name} (first run only)...")
        with urllib.request.urlopen(BEIR_URL.format(name), timeout=300) as resp:
            zipfile.ZipFile(io.BytesIO(resp.read())).extractall(cache)
    return _beir(name, folder)


def _beir(name: str, folder: Path) -> Dataset:
    def rows(path: Path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    docs = {d["_id"]: f"{d.get('title', '')}\n{d['text']}".strip() for d in rows(folder / "corpus.jsonl")}
    relevant: dict[str, set[str]] = {}
    with open(folder / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                relevant.setdefault(qid, set()).add(doc_id)
    questions = {q["_id"]: q["text"] for q in rows(folder / "queries.jsonl") if q["_id"] in relevant}
    return Dataset(name, docs, questions, relevant)
