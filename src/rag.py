"""Step 4: retrieval-augmented question answering with source citations.

Run:  python -m src.rag "Which roles need Power BI?"
"""
from __future__ import annotations

import re
import sys

from . import config, llm
from .retriever import Retriever

SYSTEM_PROMPT = """You are JobFit, an assistant that answers questions about a set of job descriptions.
Rules:
- Use ONLY the context passages provided. Do not use outside knowledge.
- Each passage starts with a line "Cite as: [...]". After each fact, copy that citation exactly,
  in plain square brackets, for example [Arm - Data Analytics Intern].
  Never cite with passage numbers, other bracket styles or placeholder text such as [Company - Job title].
- Only name a job if its passage directly supports the answer. If the context does not
  contain the answer, reply exactly: "I don't know based on these job descriptions."
- Be concise. Use short bullet points when listing several jobs."""


def citation(hit: dict) -> str:
    return f"[{hit['company']} - {hit['title']}]"


def normalise_citations(text: str) -> str:
    """Some models (e.g. gpt-oss) write citations in CJK brackets and use special spaces or dashes.
    Convert them to the plain [Company - Job title] form the rest of the app expects."""
    text = text.replace("\u3010", "[").replace("\u3011", "]")  # 【 】 brackets
    text = re.sub(r"[\u00a0\u2009\u202f]", " ", text)  # non-breaking and thin spaces
    text = re.sub(r"[\u2010\u2011]", "-", text)  # Unicode hyphens, e.g. in "Rolls-Royce"
    # inside [...] only: a dash with spaces around it (en/em dash) is the company/title separator
    text = re.sub(r"\[[^\]]*\]", lambda m: re.sub(r"\s+[\u2012-\u2015-]\s+", " - ", m.group()), text)
    return re.sub(r"\[\s+([^\]]*?)\s*\]|\[([^\]]*?)\s+\]", lambda m: f"[{m.group(1) or m.group(2)}]", text)


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"Cite as: {citation(h)}\n{h['text']}" for h in hits)


def extractive_answer(hits: list[dict], intro: str = "No LLM configured, so here are the most relevant passages:") -> str:
    """Fallback when no LLM is configured or available: show the most relevant passages."""
    if not hits:
        return "No relevant passages found."
    lines = [intro]
    for h in hits[:3]:
        snippet = h["text"].split(":", 1)[-1].strip()
        lines.append(f"- [{h['company']} - {h['title']}] {snippet[:300]}...")
    return "\n".join(lines)


def answer(question: str, retriever: Retriever | None = None, k: int = config.TOP_K,
           company: str | None = None) -> dict:
    """Answer a question. The result has "answer", "sources", "model" (the LLM that wrote the
    answer, or None) and "notice" (why the LLM could not be used, or None)."""
    retriever = retriever or Retriever()
    hits = retriever.search(question, k=k, company=company)
    model = notice = None
    if llm.is_configured() and hits:
        user = f"Context:\n{format_context(hits)}\n\nQuestion: {question}"
        try:
            text = normalise_citations(llm.chat(SYSTEM_PROMPT, user))
            model = llm.last_model
        except llm.LLMUnavailable as exc:
            notice = f"The AI model is unavailable right now ({exc})."
            text = extractive_answer(hits, "Here are the most relevant passages instead:")
    else:
        text = extractive_answer(hits)
    return {"question": question, "answer": text, "sources": hits, "model": model, "notice": notice}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which roles ask for Power BI?"
    result = answer(q)
    print(f"Q: {q}\n\n{result['answer']}\n\nSources:")
    for s in result["sources"]:
        print(f"  {s['score']:.3f}  {s['company']} - {s['title']}")
