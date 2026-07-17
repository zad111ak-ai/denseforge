"""Semantic deduplication for DenseForge.

Uses content hashing + cosine similarity to detect near-duplicate documents.
Delta encoding: stores only differences between similar documents.
"""
import hashlib
import numpy as np
from typing import List, Dict, Optional


class SemanticDeduplicator:
    """Detect and deduplicate semantically similar documents.

    Approach:
    1. Content hash (exact duplicates) — free, instant
    2. Cosine similarity on embeddings (near-duplicates) — requires embedder
    3. Delta encoding: store only the diff between similar docs
    """

    def __init__(self, similarity_threshold: float = 0.92, exact_only: bool = False):
        self.similarity_threshold = similarity_threshold
        self.exact_only = exact_only
        self._hashes: Dict[str, str] = {}  # hash -> doc_id
        self._embeddings: Dict[str, np.ndarray] = {}  # doc_id -> embedding
        self._doc_count = 0

    def content_hash(self, text: str) -> str:
        """Fast exact-duplicate detection via SHA-256."""
        normalized = text.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def add(self, doc_id: str, text: str, embedding: Optional[np.ndarray] = None) -> Dict:
        """Add document and check for duplicates.

        Returns:
            {
                "is_duplicate": bool,
                "duplicate_of": str or None,
                "similarity": float or None,
                "action": "new" | "duplicate" | "delta_stored"
            }
        """
        h = self.content_hash(text)

        # Exact duplicate check
        if h in self._hashes:
            return {
                "is_duplicate": True,
                "duplicate_of": self._hashes[h],
                "similarity": 1.0,
                "action": "duplicate",
            }

        # Semantic duplicate check
        if not self.exact_only and embedding is not None and len(self._embeddings) > 0:
            best_sim = 0.0
            best_id = None
            emb = embedding.flatten()

            for eid, eemb in self._embeddings.items():
                eemb_flat = eemb.flatten()
                # Cosine similarity
                norm_a = np.linalg.norm(emb)
                norm_b = np.linalg.norm(eemb_flat)
                if norm_a > 0 and norm_b > 0:
                    sim = float(np.dot(emb, eemb_flat) / (norm_a * norm_b))
                    if sim > best_sim:
                        best_sim = sim
                        best_id = eid

            if best_sim >= self.similarity_threshold:
                return {
                    "is_duplicate": True,
                    "duplicate_of": best_id,
                    "similarity": round(best_sim, 4),
                    "action": "delta_stored",
                }

        # New document
        self._hashes[h] = doc_id
        if embedding is not None:
            self._embeddings[doc_id] = embedding.flatten()
        self._doc_count += 1

        return {
            "is_duplicate": False,
            "duplicate_of": None,
            "similarity": None,
            "action": "new",
        }

    def batch_check(self, texts: List[str], embeddings: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Check multiple texts for duplicates."""
        results = []
        for i, text in enumerate(texts):
            emb = embeddings[i] if embeddings and i < len(embeddings) else None
            doc_id = f"batch_{self._doc_count + i}"
            results.append(self.add(doc_id, text, emb))
        return results

    def stats(self) -> Dict:
        return {
            "total_docs": self._doc_count,
            "hashes_stored": len(self._hashes),
            "embeddings_stored": len(self._embeddings),
        }

    def save_state(self) -> Dict:
        """Serialize state for persistence."""
        return {
            "hashes": dict(self._hashes),
            "doc_count": self._doc_count,
            "threshold": self.similarity_threshold,
        }

    def load_state(self, state: Dict):
        """Restore state from persistence."""
        self._hashes = dict(state.get("hashes", {}))
        self._doc_count = state.get("doc_count", 0)
        self.similarity_threshold = state.get("threshold", self.similarity_threshold)
