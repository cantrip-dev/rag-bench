"""The RAG itself, with no framework: split, embed, store, search, prompt, answer. Plus the judge."""

from __future__ import annotations

ANSWER_PROMPT = """Answer the question using only the context below. If the context does not say, reply "I don't know".

Context:
{context}

Question: {question}
Answer:"""

FAITHFUL_PROMPT = """Here is some context and an answer that was written from it.

Context:
{context}

Answer: {answer}

Is every claim in the answer supported by the context? (An answer of "I don't know" counts as supported.)
Reply with one word: yes or no."""

CORRECT_PROMPT = """Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Does the answer to grade agree with the reference answer on the facts? Reply with one word: yes or no."""


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows that overlap, so a sentence cut at one edge is whole in the next."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, max(1, len(text) - overlap), step)]


def doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def docs_in_order(chunk_ids: list[str]) -> list[str]:
    """The docs behind the retrieved chunks, best first, each once."""
    return list(dict.fromkeys(doc_of(c) for c in chunk_ids))


def answer(llm, question: str, contexts: list[str]) -> str:
    return llm.complete(ANSWER_PROMPT.format(context=_join(contexts), question=question)).strip()


def is_faithful(judge, answer_text: str, contexts: list[str]) -> bool:
    return _yes(judge.complete(FAITHFUL_PROMPT.format(context=_join(contexts), answer=answer_text)))


def is_correct(judge, question: str, answer_text: str, reference: str) -> bool:
    return _yes(judge.complete(CORRECT_PROMPT.format(question=question, reference=reference, answer=answer_text)))


def _join(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))


def _yes(reply: str) -> bool:
    return reply.strip().lower().lstrip("*\"'").startswith("yes")
