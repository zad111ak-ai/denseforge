"""Holistic Cost Optimizer — end-to-end budget allocation."""
from typing import Dict, List


class HolisticCostOptimizer:
    """Распределяет budget между модулями для максимизации quality."""

    MODULE_COSTS = {
        "embedding":    {"time_ms": 10,  "cost_usd": 0.0001, "energy_wh": 0.01},
        "retrieval":    {"time_ms": 30,  "cost_usd": 0.0002, "energy_wh": 0.02},
        "reranker":     {"time_ms": 150, "cost_usd": 0.0005, "energy_wh": 0.08},
        "speculative":  {"time_ms": 400, "cost_usd": 0.002,  "energy_wh": 0.3},
        "self_rag":     {"time_ms": 200, "cost_usd": 0.0008, "energy_wh": 0.1},
        "generation":   {"time_ms": 300, "cost_usd": 0.001,  "energy_wh": 0.2},
    }

    MODULE_QUALITY = {
        "embedding": 0.10, "retrieval": 0.30, "reranker": 0.25,
        "speculative": 0.20, "self_rag": 0.10, "generation": 0.05,
    }

    REQUIRED_BY_INTENT = {
        "simple":    ["embedding", "retrieval", "generation"],
        "factoid":   ["embedding", "retrieval", "reranker", "generation"],
        "complex":   ["embedding", "retrieval", "reranker", "self_rag", "generation"],
        "multi_hop": ["embedding", "retrieval", "reranker", "speculative", "generation"],
    }

    def __init__(self, budget: Dict[str, float] | None = None):
        self.budget = budget or {"time_ms": 2000, "cost_usd": 0.01, "energy_wh": 1.0}

    def plan_pipeline(self, query_intent: str = "simple", urgency: str = "normal") -> Dict:
        urgency_mult = {"critical": 3.0, "high": 1.5, "normal": 1.0, "low": 0.5}
        mult = urgency_mult.get(urgency, 1.0)
        available = {k: v * mult for k, v in self.budget.items()}

        required = self.REQUIRED_BY_INTENT.get(query_intent, self.REQUIRED_BY_INTENT["simple"])
        selected = list(required)
        remaining = available.copy()
        for mod in required:
            for dim in remaining:
                remaining[dim] -= self.MODULE_COSTS[mod][dim]

        candidates = [m for m in self.MODULE_COSTS if m not in selected]
        candidates.sort(key=lambda m: self.MODULE_QUALITY[m] / max(sum(self.MODULE_COSTS[m].values()), 1e-6), reverse=True)

        for mod in candidates:
            if all(remaining[dim] >= self.MODULE_COSTS[mod][dim] for dim in remaining):
                selected.append(mod)
                for dim in remaining:
                    remaining[dim] -= self.MODULE_COSTS[mod][dim]

        total_time = sum(self.MODULE_COSTS[m]["time_ms"] for m in selected)
        total_cost = sum(self.MODULE_COSTS[m]["cost_usd"] for m in selected)
        total_energy = sum(self.MODULE_COSTS[m]["energy_wh"] for m in selected)
        total_quality = sum(self.MODULE_QUALITY[m] for m in selected)

        return {
            "selected_modules": selected,
            "expected_time_ms": total_time,
            "expected_cost_usd": total_cost,
            "expected_energy_wh": total_energy,
            "expected_quality": round(min(total_quality, 1.0), 3),
            "budget_utilization": {dim: 1 - remaining[dim] / max(available[dim], 1) for dim in available},
        }

    def adjust_budget(self, feedback: Dict):
        for dim in ["time_ms", "cost_usd", "energy_wh"]:
            actual = feedback.get(f"actual_{dim}", 0)
            expected = feedback.get(f"expected_{dim}", 0)
            if expected > 0:
                ratio = actual / expected
                if ratio > 1.2: self.budget[dim] *= 1.1
                elif ratio < 0.7: self.budget[dim] *= 0.95

    def stats(self) -> dict:
        return {"budget": self.budget}
