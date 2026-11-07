"""Skill-gap analysis: compare one job's requirements with a CV.

Skills are matched with a transparent keyword dictionary (easy to explain and test).
If an LLM is configured, it also writes short, practical advice on closing the gaps.

Run:  python -m src.skills arm_data_analytics_intern
"""
from __future__ import annotations

import re
import sys

from . import config, llm
from .ingest import load_jobs

# canonical skill -> regex patterns (case-insensitive, word boundaries added automatically)
SKILLS: dict[str, list[str]] = {
    "Python": [r"python"],
    "SQL": [r"sql"],
    "R": [r"R(?=[,\s])"],  # "Python, R or SQL"
    "Scala": [r"scala"],
    "Excel": [r"excel"],
    "Power Query": [r"power query"],
    "Power BI": [r"power bi"],
    "Tableau": [r"tableau"],
    "Statistics": [r"statistic(s|al)", r"hypothesis testing", r"bayesian"],
    "Machine learning": [r"machine learning", r"ml"],
    "Regression": [r"regression"],
    "Random forests / decision trees": [r"random forests?", r"decision trees?"],
    "Gradient boosting": [r"gradient boosting", r"boosting"],
    "Neural networks / deep learning": [r"neural networks?", r"deep learning", r"cnns?"],
    "Generative AI / LLMs": [r"generative ai", r"large language models?", r"llms?", r"llm-powered"],
    "RAG / retrieval": [r"rag", r"retrieval"],
    "AI agents": [r"agents?", r"agentic", r"tool calling", r"tool use", r"orchestration"],
    "Data visualisation": [r"visuali[sz]ation", r"dashboards?", r"charts?"],
    "Data cleaning / quality": [r"data quality", r"clean(ing)?", r"validat(e|ion)"],
    "Data pipelines / ETL": [r"pipelines?", r"etl"],
    "Cloud (AWS/Azure/GCP)": [r"cloud", r"aws", r"azure", r"gcp"],
    "Spark / Hadoop": [r"spark", r"hadoop"],
    "Version control (Git)": [r"version control", r"git(hub)?"],
    "Data structures & algorithms": [r"data structures", r"algorithms"],
    "Automated testing": [r"automated tests?", r"automated testing"],
    "C# / C++": [r"c#", r"c\+\+"],
    "Mobile (iOS/Android)": [r"ios", r"android"],
    "Agile": [r"agile"],
    "Communication": [r"communicat(e|ion|or|es)", r"explain"],
    "Teamwork": [r"team(s|work)?", r"collaborat(e|ive|ion)"],
    "Problem solving": [r"problem[- ]solv(ing|er)"],
    "Attention to detail": [r"attention to detail", r"detail-oriented", r"checking details"],
    "Requirements gathering": [r"requirements from stakeholders", r"business requirements", r"gather requirements"],
    "Japanese": [r"japanese"],
}


CASE_SENSITIVE = {"R"}  # a single capital letter must not match the letter r inside words


def _compile(skill: str, pattern: str) -> re.Pattern:
    """Wrap a pattern in word boundaries (unless it ends in a symbol, e.g. C++ or C#)."""
    end = r"\b" if pattern[-1].isalnum() or pattern[-1] in "?)" else ""
    flags = 0 if skill in CASE_SENSITIVE else re.I
    return re.compile(rf"(?<![\w]){pattern}{end}", flags)


_COMPILED = {skill: [_compile(skill, p) for p in pats] for skill, pats in SKILLS.items()}


def extract_skills(text: str) -> set[str]:
    return {skill for skill, pats in _COMPILED.items() if any(p.search(text) for p in pats)}


def skill_gap(job_id: str, cv_text: str) -> dict:
    jobs = {j["job_id"]: j for j in load_jobs()}
    if job_id not in jobs:
        raise KeyError(f"Unknown job_id: {job_id}")
    job = jobs[job_id]
    job_skills = extract_skills(job["text"])
    cv_skills = extract_skills(cv_text)
    have = sorted(job_skills & cv_skills)
    missing = sorted(job_skills - cv_skills)
    coverage = round(100 * len(have) / len(job_skills)) if job_skills else 0

    advice = notice = None
    if llm.is_configured():
        system = ("You are a careers coach. Using only the job requirements and CV given, write 3-5 short, "
                  "practical bullet points on how this candidate can close their skill gaps before applying "
                  "or interviewing. Mention any eligibility rules (visa, clearance, graduation dates) that "
                  "could stop them applying. Do not invent experience.")
        user = (f"JOB: {job['company']} - {job['title']}\n{job['text']}\n\nCV:\n{cv_text}\n\n"
                f"Skills matched: {', '.join(have) or 'none'}\nSkills missing: {', '.join(missing) or 'none'}")
        try:
            advice = llm.chat(system, user, temperature=0.2)
        except llm.LLMUnavailable as exc:
            notice = f"AI advice is unavailable right now ({exc}). The skill match above still works."

    return {"job": job, "have": have, "missing": missing, "coverage": coverage, "advice": advice,
            "notice": notice}


if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else "arm_data_analytics_intern"
    result = skill_gap(job_id, config.CV_PATH.read_text(encoding="utf-8"))
    print(f"{result['job']['company']} - {result['job']['title']}  (skill coverage {result['coverage']}%)")
    print("Have:   ", ", ".join(result["have"]) or "-")
    print("Missing:", ", ".join(result["missing"]) or "-")
    if result["advice"]:
        print("\nAdvice:\n" + result["advice"])
