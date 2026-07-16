"""Bidirectional Feedback — замыкание петли Generation → Retrieval."""
import time
from typing import Dict, List
from collections import defaultdict


class BidirectionalFeedback:
    """Usage signals: cited docs → boost, skipped → demotion."""

    def __init__(self, decay_hours: float = 168.0):
        self.decay_hours = decay_hours
        self.doc_boost: Dict[int, float] = defaultdict(float)
        self.pair_usage: Dict[tuple, int] = defaultdict(int)
        self.last_update: Dict[int, float] = {}

    def record_usage(self, query_hash: str, retrieved_docs: List[int],
                     cited_docs: List[int], user_feedback: str = "neutral"):
        now = time.time()
        for doc_id in cited_docs:
            self.doc_boost[doc_id] += 1.0
            self.last_update[doc_id] = now
        for doc_id in retrieved_docs:
            if doc_id not in cited_docs:
                self.doc_boost[doc_id] -= 0.1
                self.last_update[doc_id] = now
        mult = {"positive": 2.0, "neutral": 1.0, "negative": -1.0}.get(user_feedback, 1.0)
        for doc_id in cited_docs:
            self.doc_boost[doc_id] += 0.5 * mult
            self.pair_usage[(query_hash, doc_id)] += 1

    def get_doc_boost(self, doc_id: int) -> float:
        if doc_id not in self.doc_boost:
            return 0.0
        last = self.last_update.get(doc_id, 0)
        hours = (time.time() - last) / 3600
        decay = 0.5 ** (hours / self.decay_hours)
        return self.doc_boost[doc_id] * decay

    def get_pair_affinity(self, query_hash: str, doc_id: int) -> float:
        return float(self.pair_usage.get((query_hash, doc_id), 0))

    def apply_to_retrieval_scores(self, results: List[Dict], query_hash: str = None) -> List[Dict]:
        alpha, beta = 0.2, 0.05
        for r in results:
            doc_id = r.get("doc_id")
            original = r.get("score", 0)
            boost = self.get_doc_boost(doc_id)
            affinity = self.get_pair_affinity(query_hash, doc_id) if query_hash else 0
            boosted = original * (1 + alpha * max(-0.5, min(boost, 2.0)))
            boosted += beta * min(affinity, 5.0)
            r["original_score"] = original
            r["feedback_boost"] = boost
            r["score"] = max(0.0, boosted)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def stats(self) -> dict:
        return {
            "tracked_documents": len(self.doc_boost),
            "tracked_pairs": len(self.pair_usage),
        }
