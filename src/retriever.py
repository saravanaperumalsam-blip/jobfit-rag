"""Step 3: semantic search over the indexed job descriptions.

Backends: "minilm" (semantic), "tfidf" (keyword) or "hybrid" (both, merged with
reciprocal rank fusion, so exact keyword matches and paraphrases are both found).
"""
from __future__ import annotations

from types import SimpleNamespace

import chromadb

from . import config
from .embeddings import get_embedder
from .ingest import build_index, collection_name

RRF_K = 60  # standard reciprocal-rank-fusion constant


class Retriever:
    def __init__(self, backend: str | None = None, chunk_size: int = config.CHUNK_SIZE,
                 overlap: int = config.CHUNK_OVERLAP):
        backend = (backend or config.EMBEDDING_BACKEND).lower()
        self.parts: list[Retriever] = []
        if backend == "hybrid":
            semantic = Retriever("minilm", chunk_size, overlap)
            keyword = Retriever("tfidf", chunk_size, overlap)
            # If MiniLM could not load it falls back to TF-IDF; then there is nothing to fuse.
            self.parts = [semantic, keyword] if semantic.embedder.name == "minilm" else [keyword]
            self.embedder = SimpleNamespace(name="hybrid" if len(self.parts) == 2 else "tfidf")
            self.name, self.collection = keyword.name, keyword.collection
            return

        self.embedder = get_embedder(backend)
        self.name = collection_name(self.embedder.name, chunk_size, overlap)
        client = chromadb.PersistentClient(path=str(config.DB_DIR))
        existing = [c.name if hasattr(c, "name") else c for c in client.list_collections()]
        if self.name not in existing:  # build the index automatically on first run
            build_index(self.embedder.name, chunk_size, overlap)
        self.embedder.load(config.DB_DIR / f"{self.name}.pkl")
        self.collection = client.get_collection(self.name, embedding_function=None)

    def search(self, query: str, k: int = config.TOP_K, company: str | None = None) -> list[dict]:
        """Return the top-k chunks as dicts with text, metadata and similarity score."""
        if len(self.parts) == 2:
            return self._hybrid_search(query, k, company)
        if self.parts:
            return self.parts[0].search(query, k, company)
        where = {"company": company} if company else None
        vector = self.embedder.embed([query])
        if not any(vector[0]):
            # TF-IDF gives an all-zero vector when no query word appears in any job.
            # Similarity is then meaningless and the order would be random, so return nothing.
            return []
        result = self.collection.query(
            query_embeddings=vector,
            n_results=min(k, self.collection.count()),
            where=where,
        )
        hits = []
        for cid, text, meta, dist in zip(result["ids"][0], result["documents"][0],
                                         result["metadatas"][0], result["distances"][0]):
            hits.append({"id": cid, "text": text, **meta, "score": round(1 - dist, 3)})
        return hits

    def _hybrid_search(self, query: str, k: int, company: str | None) -> list[dict]:
        """Reciprocal rank fusion: each chunk scores sum(1 / (RRF_K + rank)) over both rankings.
        The score is scaled so 1.0 means ranked first by both methods."""
        pool = max(k * 3, 15)
        fused: dict[str, float] = {}
        hits_by_id: dict[str, dict] = {}
        for part in self.parts:
            for rank, hit in enumerate(part.search(query, pool, company), 1):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1 / (RRF_K + rank)
                hits_by_id.setdefault(hit["id"], hit)
        best = len(self.parts) / (RRF_K + 1)
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**hits_by_id[cid], "score": round(fused[cid] / best, 3)} for cid in ranked]

    def companies(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["company"] for m in metas})
