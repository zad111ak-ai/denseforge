"""Adaptive Parameter Controller — self-tuning thresholds."""
import time
from collections import deque
from typing import Dict

import numpy as np


class AdaptiveParameterController:
    """Auto-tune params via gradient-free optimization."""

    PARAM_BOUNDS = {
        "cache_ttl": (600, 86400),
        "cache_similarity": (0.85, 0.98),
        "fusion_bm25": (0.1, 0.5),
        "fusion_dense": (0.3, 0.7),
        "top_k_candidates": (20, 100),
        "rerank_weight": (0.5, 0.9),
    }

    def __init__(self, initial_params: Dict[str, float] | None = None):
        self.params = initial_params or {
            "cache_ttl": 3600, "cache_similarity": 0.92,
            "fusion_bm25": 0.30, "fusion_dense": 0.50,
            "top_k_candidates": 50, "rerank_weight": 0.70,
        }
        self.history: deque = deque(maxlen=10000)
        self.exploration_rate = 0.1

    def get_params(self) -> Dict[str, float]:
        if np.random.random() < self.exploration_rate:
            return self._explore()
        return self.params.copy()

    def record_outcome(self, metrics: Dict[str, float]):
        self.history.append({"params": self.params.copy(), "metrics": metrics, "timestamp": time.time()})
        if len(self.history) % 100 == 0:
            self._tune()

    def _tune(self):
        if len(self.history) < 50:
            return

        def objective(m):
            return (m.get("answer_quality", 0) * 0.4 + m.get("user_satisfaction", 0) * 0.3
                    + m.get("cache_hit_rate", 0) * 0.2 - min(m.get("cost_usd", 0) * 10, 0.3)
                    - min(m.get("latency_ms", 0) / 10000, 0.2))

        sorted_h = sorted(self.history, key=lambda h: objective(h["metrics"]), reverse=True)
        top_k = max(5, len(sorted_h) // 10)
        top_entries = sorted_h[:top_k]

        for param in self.params:
            values = [e["params"][param] for e in top_entries]
            target = float(np.mean(values))
            self.params[param] = 0.7 * self.params[param] + 0.3 * target
            low, high = self.PARAM_BOUNDS[param]
            self.params[param] = max(low, min(high, self.params[param]))
        self.exploration_rate = max(0.02, self.exploration_rate * 0.99)

    def _explore(self) -> Dict[str, float]:
        explored = {}
        for param, value in self.params.items():
            low, high = self.PARAM_BOUNDS[param]
            std = (high - low) * 0.1
            explored[param] = max(low, min(high, value + np.random.normal(0, std)))
        return explored

    def get_dashboard(self) -> dict:
        return {"current_params": self.params, "exploration_rate": self.exploration_rate,
                "history_size": len(self.history)}

    def stats(self) -> dict:
        return self.get_dashboard()
