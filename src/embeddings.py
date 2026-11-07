"""Embedding backends.

- "minilm": sentence-transformers/all-MiniLM-L6-v2 (semantic, runs locally, recommended)
- "tfidf":  scikit-learn TF-IDF (keyword baseline, no model download needed)

If sentence-transformers cannot be loaded, the code falls back to TF-IDF so the
project still runs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from . import config


class MiniLMEmbedder:
    name = "minilm"

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:  # nothing to fit for a pre-trained model
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        pass


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def embed(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).tolist()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)


def get_embedder(backend: str | None = None):
    """Return an embedder, falling back to TF-IDF if MiniLM is unavailable."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()
    if backend == "minilm":
        try:
            return MiniLMEmbedder()
        except Exception as exc:  # missing package or no internet to download the model
            print(f"[embeddings] MiniLM unavailable ({exc.__class__.__name__}); using TF-IDF instead.")
            return TfidfEmbedder()
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")
