"""Retrieval scores for one question, and latency percentiles. Average the scores over all questions."""

from __future__ import annotations

import math


def recall(ranked: list[str], relevant: set[str], k: int) -> float:
    """Share of the relevant docs found in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant doc is in the top k."""
    return float(bool(set(ranked[:k]) & relevant))


def mrr(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant doc (0 if none in the top k)."""
    return next((1 / rank for rank, d in enumerate(ranked[:k], 1) if d in relevant), 0.0)


def ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    """How close the order is to ideal: relevant docs first."""
    dcg = sum(1 / math.log2(rank + 1) for rank, d in enumerate(ranked[:k], 1) if d in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def percentiles(ms: list[float]) -> dict[str, float]:
    s = sorted(ms)
    pick = lambda p: s[min(len(s) - 1, int(p / 100 * len(s)))]
    return {"p50": round(pick(50), 2), "p95": round(pick(95), 2), "p99": round(pick(99), 2)}


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0
