"""Shared Attention — единый attention map над documents."""
import numpy as np
from typing import Dict, List


class SharedAttentionContext:
    """Один attention map — все модули читают и уточняют."""

    def __init__(self):
        self.doc_scores: Dict[int, Dict[str, float]] = {}

    def add_scores(self, doc_id: int, source: str, score: float):
        if doc_id not in self.doc_scores:
            self.doc_scores[doc_id] = {}
        self.doc_scores[doc_id][source] = score

    def add_batch(self, doc_ids: List[int], source: str, scores: List[float]):
        for doc_id, score in zip(doc_ids, scores):
            self.add_scores(doc_id, source, score)

    def get_fused_score(self, doc_id: int, weights: Dict[str, float] | None = None) -> float:
        weights = weights or {"retrieval": 0.25, "rerank": 0.55, "feedback": 0.20}
        scores = self.doc_scores.get(doc_id, {})
        if not scores:
            return 0.0
        total_w, fused = 0.0, 0.0
        for source, w in weights.items():
            if source in scores:
                fused += w * scores[source]
                total_w += w
        return fused / total_w if total_w > 0 else 0.0

    def get_top_k(self, k: int, weights: Dict = None) -> List[int]:
        scored = [(doc_id, self.get_fused_score(doc_id, weights)) for doc_id in self.doc_scores]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored[:k]]

    def get_attention_vector(self, doc_ids: List[int]) -> np.ndarray:
        return np.array([self.get_fused_score(d) for d in doc_ids])

    def explain_score(self, doc_id: int) -> dict:
        scores = self.doc_scores.get(doc_id, {})
        return {
            "doc_id": doc_id,
            "score_breakdown": scores,
            "fused_score": self.get_fused_score(doc_id),
            "top_contributor": max(scores.items(), key=lambda x: x[1])[0] if scores else None,
        }

    def clear(self):
        self.doc_scores.clear()

    def stats(self) -> dict:
        return {"tracked_docs": len(self.doc_scores)}
