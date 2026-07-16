"""Query Router — classify queries and select retrieval channels."""
import re
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    query_type: str  # "simple", "complex", "multi_hop", "causal", "factual"
    channels: list[str]
    use_reranker: bool
    retrieve_k: int
    confidence: float


class QueryRouter:
    """Classify query complexity and select optimal retrieval strategy."""

    COMPLEXITY_PATTERNS = {
        "causal": [r"\bwhy\b", r"\bbecause\b", r"\bcause\b", r"\breason\b"],
        "multi_hop": [r"\band then\b", r"\bafter that\b", r"\bfirst.*then\b", r"\bstep\b"],
        "comparison": [r"\bcompare\b", r"\bdifference\b", r"\bvs\b", r"\bversus\b"],
        "aggregation": [r"\ball\b", r"\bevery\b", r"\blist all\b", r"\bcount\b"],
    }

    def route(self, query: str) -> RoutingDecision:
        query_lower = query.lower()
        detected_types = []
        for qtype, patterns in self.COMPLEXITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_types.append(qtype)
                    break

        # Determine query type
        if not detected_types:
            query_type = "simple"
            channels = ["bm25", "dense"]
            use_reranker = False
            retrieve_k = 10
            confidence = 0.9
        elif "causal" in detected_types:
            query_type = "causal"
            channels = ["bm25", "dense", "binary"]
            use_reranker = True
            retrieve_k = 20
            confidence = 0.8
        elif "multi_hop" in detected_types:
            query_type = "multi_hop"
            channels = ["bm25", "dense", "binary"]
            use_reranker = True
            retrieve_k = 30
            confidence = 0.7
        else:
            query_type = "complex"
            channels = ["bm25", "dense", "binary"]
            use_reranker = True
            retrieve_k = 20
            confidence = 0.75

        return RoutingDecision(
            query_type=query_type, channels=channels,
            use_reranker=use_reranker, retrieve_k=retrieve_k, confidence=confidence,
        )
