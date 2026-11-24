"""Measure retrieval quality on a labeled question set.

Each line of eval/questions.jsonl looks like:
  {"question": "...", "source": "employee-handbook.md", "answer_contains": "16 weeks"}

A chunk counts as relevant when it comes from `source` and contains the
`answer_contains` text. Defining relevance this way (instead of by chunk id)
means the labels stay valid when you change the chunk size, which is exactly
the experiment you want to run.

Run:  python -m ragbot.evaluate --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from ragbot.config import settings
from ragbot.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from ragbot.retriever import HybridRetriever
from ragbot.store import VectorStore


def load_questions(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def relevant_ids(retriever: HybridRetriever, item: dict) -> set[str]:
    needle = item["answer_contains"].lower()
    return {
        chunk.id
        for chunk in retriever.chunks_by_id.values()
        if chunk.source == item["source"] and needle in chunk.text.lower()
    }


def evaluate(retriever: HybridRetriever, questions: list[dict], k: int) -> dict[str, dict]:
    """Return {mode: {metric: value}} for vector, keyword and hybrid retrieval."""
    results = {}
    for mode in ("vector", "keyword", "hybrid"):
        recall, precision, hits, rr = [], [], [], []
        for item in questions:
            relevant = relevant_ids(retriever, item)
            if not relevant:
                raise ValueError(f"No chunk matches the label for: {item['question']!r}")
            retrieved = [r.chunk.id for r in retriever.retrieve(item["question"], k, mode=mode)]
            recall.append(recall_at_k(retrieved, relevant, k))
            precision.append(precision_at_k(retrieved, relevant, k))
            hits.append(hit_at_k(retrieved, relevant, k))
            rr.append(reciprocal_rank(retrieved, relevant))
        results[mode] = {
            f"Recall@{k}": mean(recall),
            f"Precision@{k}": mean(precision),
            f"Hit@{k}": mean(hits),
            "MRR": mean(rr),
        }
    return results


def format_table(results: dict[str, dict]) -> str:
    metrics = list(next(iter(results.values())))
    lines = ["| Mode | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for mode, values in results.items():
        lines.append(f"| {mode} | " + " | ".join(f"{values[m]:.2f}" for m in metrics) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--k", type=int, default=settings.top_k)
    args = parser.parse_args()

    store = VectorStore(settings.db_dir, settings.collection)
    if store.count() == 0:
        raise SystemExit("The vector store is empty. Run `python -m ragbot.ingest` first.")
    retriever = HybridRetriever(store, candidates=settings.candidates)
    questions = load_questions(args.questions)

    print(f"Evaluating {len(questions)} questions, k={args.k}, {store.count()} chunks indexed\n")
    print(format_table(evaluate(retriever, questions, args.k)))


if __name__ == "__main__":
    main()