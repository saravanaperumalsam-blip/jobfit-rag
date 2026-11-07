"""Step 6: evaluate retrieval quality and compare settings.

For each test question we know which job holds the answer. We measure:
- Hit@1 / Hit@3: is the correct job in the top 1 / top 3 retrieved chunks?
- MRR (mean reciprocal rank): 1 / rank of the first correct chunk, averaged.
Questions are tagged "keyword" (share words with the job text) or "paraphrase"
(same meaning, different words), which shows where semantic embeddings help.

Run:  python -m src.evaluate                     (MiniLM, falls back to TF-IDF)
      python -m src.evaluate --backends minilm tfidf hybrid
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from . import config
from .retriever import Retriever

QUESTIONS = config.ROOT / "eval" / "questions.csv"
SETTINGS = [(40, 10), (120, 20), (250, 40)]  # (chunk_size words, overlap words)


def evaluate(retriever: Retriever, questions: pd.DataFrame, k: int = 5) -> dict:
    hit1 = hit3 = rr = 0.0
    misses = []
    for q in questions.itertuples():
        ranked = [h["job_id"] for h in retriever.search(q.question, k=k)]
        rank = ranked.index(q.expected_job_id) + 1 if q.expected_job_id in ranked else None
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr += 1 / rank if rank else 0
        if rank != 1:
            misses.append((q.question, q.expected_job_id, ranked[0] if ranked else "(no match)"))
    n = max(len(questions), 1)
    return {"hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr / n, "misses": misses}


def main(backends: list[str]) -> None:
    questions = pd.read_csv(QUESTIONS)
    rows, miss_log = [], []
    for backend in backends:
        for size, overlap in SETTINGS:
            r = Retriever(backend, size, overlap)
            m = evaluate(r, questions)
            kw = evaluate(r, questions[questions["type"] == "keyword"])
            para = evaluate(r, questions[questions["type"] == "paraphrase"])
            actual = r.embedder.name
            rows.append({"embeddings": actual, "chunk_size": size, "overlap": overlap,
                         "chunks": r.collection.count(), "hit@1": m["hit@1"],
                         "hit@3": m["hit@3"], "mrr": m["mrr"],
                         "hit@1 keyword": kw["hit@1"], "hit@1 paraphrase": para["hit@1"]})
            miss_log += [(actual, size, *x) for x in m["misses"]]
            print(f"{actual:6s} size={size:3d}  hit@1={m['hit@1']:.2f}  hit@3={m['hit@3']:.2f}  "
                  f"mrr={m['mrr']:.2f}  (keyword {kw['hit@1']:.2f} / paraphrase {para['hit@1']:.2f})")

    df = pd.DataFrame(rows).drop_duplicates(subset=["embeddings", "chunk_size"])
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "eval_results.csv", index=False)

    metric_cols = ["hit@1", "hit@3", "mrr", "hit@1 keyword", "hit@1 paraphrase"]
    table = df.assign(**{c: df[c].map(lambda v: f"{v:.2f}") for c in metric_cols})
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in table.values)
    misses = "\n".join(f"- [{b}, {s}] {q} -> expected `{e}`, got `{g}`" for b, s, q, e, g in miss_log)
    (config.RESULTS_DIR / "eval_results.md").write_text(
        f"# Retrieval evaluation\n\nRun on {dt.date.today()} with {len(questions)} questions "
        f"({(questions['type'] == 'keyword').sum()} keyword, {(questions['type'] == 'paraphrase').sum()} paraphrase).\n\n"
        f"{header}{body}\n\n## Questions not ranked first\n\n{misses or 'None'}\n", encoding="utf-8")
    print(f"\nSaved results to {config.RESULTS_DIR / 'eval_results.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=[config.EMBEDDING_BACKEND],
                        choices=["minilm", "tfidf", "hybrid"])
    main(parser.parse_args().backends)
