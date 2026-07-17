"""UQR — Unified Query Representation. Единый паспорт запроса для всех модулей."""
import hashlib
import time
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class QueryProfile:
    """Единое представление запроса."""
    query_id: str
    raw_text: str
    created_at: float
    # Embeddings (многоуровневые)
    embedding_512: np.ndarray = field(default_factory=lambda: np.zeros(512, dtype=np.float32))
    embedding_128: np.ndarray = field(default_factory=lambda: np.zeros(128, dtype=np.float32))
    embedding_binary: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.uint8))
    # Semantic features
    intent: str = "simple"          # simple/factoid/complex/multi_hop
    domain: str = "general"
    complexity: float = 0.5
    has_temporal: bool = False
    has_causal: bool = False
    has_comparison: bool = False
    # Operational
    urgency: str = "normal"
    estimated_cost_wh: float = 0.01
    estimated_latency_ms: int = 50
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    cache_key: str = ""


class UQRBuilder:
    """Строит UQR за один pass — экономит ~40% compute vs раздельного анализа."""

    INTENT_PROFILES = {
        "simple": (0.01, 50),
        "factoid": (0.05, 200),
        "complex": (0.3, 800),
        "multi_hop": (0.8, 2000),
    }

    def __init__(self, embedder=None):
        self.embedder = embedder
        self._cache: dict[str, QueryProfile] = {}
        self._cache_max = 1000

    def build(self, query: str, user_id: str | None = None,
              session_id: str | None = None) -> QueryProfile:
        cache_key = self._hash_query(query, user_id)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            cached.session_id = session_id
            return cached

        features = self._extract_features(query)
        intent = self._classify_intent(query, features)
        cost, latency = self._predict_resources(intent, features)

        emb_512 = np.zeros(512, dtype=np.float32)
        emb_128 = np.zeros(128, dtype=np.float32)
        emb_bin = np.zeros(64, dtype=np.uint8)
        if self.embedder:
            try:
                result = self.embedder.encode(query)
                emb_512 = result.vectors.get(512, emb_512)
                emb_128 = result.vectors.get(128, emb_128)
                emb_bin = result.binary if hasattr(result, 'binary') else emb_bin
            except Exception:
                pass

        profile = QueryProfile(
            query_id=cache_key, raw_text=query, created_at=time.time(),
            embedding_512=emb_512, embedding_128=emb_128, embedding_binary=emb_bin,
            intent=intent, domain=self._detect_domain(query),
            complexity=self._complexity_score(features),
            has_temporal=features["temporal"], has_causal=features["causal"],
            has_comparison=features["comparison"],
            urgency=self._infer_urgency(query, features),
            estimated_cost_wh=cost, estimated_latency_ms=latency,
            user_id=user_id, session_id=session_id, cache_key=cache_key,
        )

        if len(self._cache) >= self._cache_max:
            oldest = min(self._cache.values(), key=lambda p: p.created_at)
            del self._cache[oldest.cache_key]
        self._cache[cache_key] = profile
        return profile

    def _extract_features(self, query: str) -> dict:
        q = query.lower()
        return {
            "temporal": bool(re.search(
                r'\b(when|когда|today|сегодня|yesterday|вчера|tomorrow|завтра|\d{4}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', q)),
            "causal": bool(re.search(
                r'\b(why|почему|because|потому|cause|причина|result|следствие)\b', q)),
            "comparison": bool(re.search(
                r'\b(compare|сравни|better|лучше|worse|хуже|difference|разница|vs|versus)\b', q)),
            "length": len(query.split()),
            "has_numbers": bool(re.search(r'\d', q)),
            "is_question": "?" in query or bool(re.search(r'\b(what|how|who|where|кто|что|где|как|почему)\b', q)),
        }

    def _classify_intent(self, query: str, features: dict) -> str:
        if features["length"] <= 5 and features["is_question"]:
            return "simple"
        if features["is_question"] and not features["causal"] and features["length"] < 15:
            return "factoid"
        if features["causal"] or features["comparison"]:
            return "complex"
        if features["length"] > 20:
            return "multi_hop"
        return "factoid"

    def _detect_domain(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["code", "api", "function", "bug", "python", "code"]):
            return "tech"
        if any(w in q for w in ["price", "cost", "market", "stock", "revenue"]):
            return "finance"
        if any(w in q for w in ["patient", "diagnosis", "treatment", "symptom"]):
            return "medical"
        return "general"

    def _complexity_score(self, features: dict) -> float:
        score = 0.3
        if features["causal"]:
            score += 0.25
        if features["comparison"]:
            score += 0.2
        if features["temporal"]:
            score += 0.1
        if features["length"] > 20:
            score += 0.15
        return min(score, 1.0)

    def _predict_resources(self, intent: str, features: dict) -> tuple[float, int]:
        base_cost, base_lat = self.INTENT_PROFILES.get(intent, (0.1, 300))
        if features["causal"]:
            base_cost *= 1.5
            base_lat = int(base_lat * 1.5)
        if features["comparison"]:
            base_cost *= 1.3
            base_lat = int(base_lat * 1.3)
        return base_cost, base_lat

    def _infer_urgency(self, query: str, features: dict) -> str:
        if any(k in query.lower() for k in ["urgent", "asap", "critical", "срочно"]):
            return "critical"
        if features["length"] > 30:
            return "high"
        return "normal"

    @staticmethod
    def _hash_query(query: str, user_id: str | None) -> str:
        key = f"{user_id or 'anon'}:{query.strip().lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def stats(self) -> dict:
        return {"cached_profiles": len(self._cache)}
