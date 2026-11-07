"""Step 2 + 3: load, clean, chunk and embed job descriptions into ChromaDB.

Run:  python -m src.ingest
"""
from __future__ import annotations

import argparse
import re

import chromadb
import pandas as pd

from . import config
from .embeddings import get_embedder

BOILERPLATE = [
    r"equal opportunit(y|ies).*",
    r"reasonable adjustments?.*",
    r"we (welcome|encourage) applications from.*",
]


def load_jobs() -> list[dict]:
    """Read jobs.csv and the matching text file for each job."""
    df = pd.read_csv(config.JOBS_CSV)
    jobs = []
    for row in df.to_dict(orient="records"):
        text = (config.DATA_DIR / row["file"]).read_text(encoding="utf-8")
        jobs.append({**row, "text": clean_text(text)})
    return jobs


def clean_text(text: str) -> str:
    """Normalise whitespace and drop generic boilerplate lines."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if any(re.match(p, line, flags=re.I) for p in BOILERPLATE):
            continue
        lines.append(line)
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `chunk_size` words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


def make_chunks(jobs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Chunk every job. Each chunk starts with a short header (company, title and location) so
    it still makes sense on its own after retrieval, even when chunks are small."""
    records = []
    for job in jobs:
        header = f"{job['company']} - {job['title']} ({job['location']}):"
        for i, chunk in enumerate(chunk_text(job["text"], chunk_size, overlap)):
            records.append({
                "id": f"{job['job_id']}__{i}",
                "text": f"{header} {chunk}",
                "metadata": {
                    "job_id": job["job_id"],
                    "company": job["company"],
                    "title": job["title"],
                    "location": job["location"],
                    "closing_date": str(job["closing_date"]),
                    "chunk": i,
                },
            })
    return records


def collection_name(backend: str, chunk_size: int, overlap: int) -> str:
    return f"jobs_{backend}_{chunk_size}_{overlap}"


def build_index(backend: str | None = None, chunk_size: int = config.CHUNK_SIZE,
                overlap: int = config.CHUNK_OVERLAP, verbose: bool = True) -> str:
    """Create (or rebuild) a ChromaDB collection and return its name.
    "hybrid" builds both the MiniLM and the TF-IDF collections."""
    if (backend or config.EMBEDDING_BACKEND).lower() == "hybrid":
        build_index("minilm", chunk_size, overlap, verbose)
        return build_index("tfidf", chunk_size, overlap, verbose)
    jobs = load_jobs()
    records = make_chunks(jobs, chunk_size, overlap)
    texts = [r["text"] for r in records]

    embedder = get_embedder(backend)
    embedder.fit(texts)
    name = collection_name(embedder.name, chunk_size, overlap)
    embedder.save(config.DB_DIR / f"{name}.pkl")

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"},
                                          embedding_function=None)
    collection.add(
        ids=[r["id"] for r in records],
        documents=texts,
        metadatas=[r["metadata"] for r in records],
        embeddings=embedder.embed(texts),
    )
    if verbose:
        print(f"Indexed {len(jobs)} jobs as {len(records)} chunks into '{name}'.")
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the job description vector index")
    parser.add_argument("--backend", default=config.EMBEDDING_BACKEND, choices=["minilm", "tfidf", "hybrid"])
    parser.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    args = parser.parse_args()
    build_index(args.backend, args.chunk_size, args.overlap)
