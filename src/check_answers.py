"""Step 7: end-to-end answer check (retrieval + LLM).

Runs the labelled test questions through the full pipeline and compares each LLM answer
with the reference job. Writes a side-by-side report to results/llm_answers.md.

For each question it records:
- PASS / PARTIAL / FAIL: does the answer name every reference job (company and title)?
- Right job retrieved: was the reference job among the passages sent to the LLM?
- Citations valid: does every [citation] exactly match one of the retrieved sources?

Needs an LLM configured in .env (LLM_PROVIDER and LLM_API_KEY).

Run:  python -m src.check_answers              (all questions)
      python -m src.check_answers --limit 5    (first 5, quick check)
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import pandas as pd

from . import config, llm
from .rag import answer, citation
from .retriever import Retriever

QUESTIONS = config.ROOT / "eval" / "questions.csv"
REPORT = config.RESULTS_DIR / "llm_answers.md"
# Extra question with several correct jobs (ids separated by "|").
EXTRA = [{"question": "Which jobs need security clearance?",
          "expected_job_id": "awe_safety_placement|cgi_applied_ai_graduate|cgi_data_analysis_graduate",
          "type": "multi"}]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class DailyLimitReached(Exception):
    """The provider's per-day quota is used up; retrying now cannot succeed."""


def ask(question: str, retriever: Retriever, retries: int = 6) -> dict:
    """Call the pipeline. The LLM layer already switches to backup models; if no model can
    answer, wait and retry a per-minute limit, but stop at once on a daily limit."""
    for attempt in range(retries):
        result = answer(question, retriever)
        if not result["notice"]:
            return result
        if "daily" in result["notice"] and "per-minute" not in result["notice"]:
            raise DailyLimitReached(result["notice"])
        wait = 15 * (attempt + 1)
        print(f"  {result['notice']} Waiting {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError("The AI model stayed unavailable; try again later.")


def check(question: str, expected: list[str], jobs: pd.DataFrame, result: dict) -> dict:
    text, sources = result["answer"], result["sources"]
    # Company names like "Renishaw (via Graduate Recruitment Bureau)": match on the main name.
    found = [norm(jobs.loc[j, "company"].split(" (")[0]) in norm(text)
             and norm(jobs.loc[j, "title"]) in norm(text) for j in expected]
    retrieved = {s["job_id"] for s in sources}
    valid = {citation(s) for s in sources}
    cites = re.findall(r"\[[^\]]*\]", text)
    bad = [c for c in cites if c not in valid]
    return {
        "verdict": "PASS" if all(found) else ("PARTIAL" if any(found) else "FAIL"),
        "in_context": all(j in retrieved for j in expected),
        "cite_ok": bool(cites) and not bad,
        "bad_cites": bad,
    }


def write_report(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    counts = df["verdict"].value_counts().to_dict()
    lines = [
        f"# LLM answer check ({llm.describe()}, search: {config.EMBEDDING_BACKEND}, "
        f"chunk {config.CHUNK_SIZE}, top {config.TOP_K})", "",
        f"Questions: {len(df)} · PASS {counts.get('PASS', 0)} · PARTIAL {counts.get('PARTIAL', 0)} · "
        f"FAIL {counts.get('FAIL', 0)} · right job retrieved {int(df['in_context'].sum())}/{len(df)} · "
        f"citations valid {int(df['cite_ok'].sum())}/{len(df)}", "",
        "PASS = the answer names every reference job (company and title). "
        "'Right job retrieved' = the reference job was among the passages given to the LLM. "
        "'Citations valid' = every [citation] matches a retrieved source exactly.", "",
    ]
    for r in rows:
        cites = "all valid" if r["cite_ok"] else "problem: " + (", ".join(r["bad_cites"]) or "none given")
        lines += [
            f"## {r['n']}. {r['question']}  ({r['type']})", "",
            f"**Reference:** {r['reference']}  ",
            f"**Verdict:** {r['verdict']} · right job retrieved: {'yes' if r['in_context'] else 'no'} · "
            f"citations: {cites} · model: {r['model']}", "",
            "**Model answer:**", "",
            *("> " + line for line in r["answer"].splitlines()), "",
        ]
    config.RESULTS_DIR.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(limit: int | None = None, pause: float = 2.0) -> None:
    if not llm.is_configured():
        sys.exit("No LLM configured: set LLM_PROVIDER and LLM_API_KEY in .env first.")
    print(f"LLM: {llm.describe()}", flush=True)

    jobs = pd.read_csv(config.JOBS_CSV).set_index("job_id")
    questions = pd.concat([pd.DataFrame(EXTRA), pd.read_csv(QUESTIONS)], ignore_index=True)
    if limit:
        questions = questions.head(limit)
    retriever = Retriever()

    rows = []
    for n, q in enumerate(questions.itertuples(), 1):
        expected = q.expected_job_id.split("|")
        try:
            result = ask(q.question, retriever)
        except DailyLimitReached as exc:
            print(f"\nStopped at question {n} of {len(questions)}: {exc}.")
            if not rows:
                sys.exit("No answers yet, so the previous report was left unchanged.")
            print(f"Saving the {len(rows)} answers so far.")
            break
        row = check(q.question, expected, jobs, result)
        rows.append({"n": n, "question": q.question, "type": q.type, "answer": result["answer"],
                     "model": result["model"],
                     "reference": "; ".join(f"{jobs.loc[j, 'company']} - {jobs.loc[j, 'title']}"
                                            for j in expected), **row})
        print(f"{n:2d}. {row['verdict']:7s} retrieved={'yes' if row['in_context'] else 'NO '} "
              f"citations={'ok ' if row['cite_ok'] else 'BAD'} {q.question}"
              + (f"  [backup: {result['model']}]" if result["model"] != llm._settings()[1] else ""), flush=True)
        time.sleep(pause)  # stay under free-tier rate limits

    write_report(rows)
    df = pd.DataFrame(rows)
    print(f"\nPASS {int((df['verdict'] == 'PASS').sum())}/{len(df)} · "
          f"right job retrieved {int(df['in_context'].sum())}/{len(df)} · "
          f"citations valid {int(df['cite_ok'].sum())}/{len(df)}")
    by_type = df.groupby("type")["verdict"].apply(lambda s: f"{(s == 'PASS').sum()}/{len(s)}")
    print("PASS by question type:", by_type.to_dict())
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check LLM answers against the labelled questions")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between API calls")
    args = parser.parse_args()
    main(args.limit, args.pause)
