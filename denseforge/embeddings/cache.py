"""Semantic Query Cache with vector similarity lookup."""
import time
import pickle
from pathlib import Path
from typing import Optional
import numpy as np


class SemanticQueryCache:
    """Cache that matches by query similarity, not exact text."""

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 10000,
                 default_ttl: float = 3600, persist_path: Optional[str] = None):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = persist_path
        self._entries: list[dict] = []  # [{query, embedding, response, created_at}]
        self._hits = 0
        self._misses = 0

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        now = time.time()
        best_match = None
        best_score = 0.0

        for entry in self._entries:
            if now - entry["created_at"] > self.default_ttl:
                continue
            sim = self._cosine_similarity(query_embedding, entry["embedding"])
            if sim >= self.similarity_threshold and sim > best_score:
                best_score = sim
                best_match = entry["response"]

        if best_match:
            self._hits += 1
            return best_match
        self._misses += 1
        return None

    def put(self, query: str, query_embedding: np.ndarray, response: dict):
        if len(self._entries) >= self.max_size:
            self._entries = self._entries[self.max_size // 2:]
        self._entries.append({
            "query": query, "embedding": query_embedding.copy(),
            "response": response, "created_at": time.time(),
        })

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._entries), "hits": self._hits, "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def save(self):
        if self.persist_path:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump(self._entries, f)

    def load(self):
        if self.persist_path and Path(self.persist_path).exists():
            with open(self.persist_path, "rb") as f:
                self._entries = pickle.load(f)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
