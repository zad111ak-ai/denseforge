"""IsolatedReranker — re-rank retrieved documents with cross-encoder or keyword fallback."""
from __future__ import annotations

import re

from loguru import logger


class IsolatedReranker:
    """Re-rank candidate documents for a given query.

    Attempts to load a cross-encoder model for high-quality scoring.  When the
    model is unavailable (missing dependency or download failure), falls back to
    a transparent keyword-overlap scoring strategy so the pipeline always works.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._cross_encoder = None
        self._use_cross_encoder: bool | None = None  # None = not probed yet

        logger.info(
            "IsolatedReranker initialised (model={model}, device={device})",
            model=model_name,
            device=device,
        )

    # ------------------------------------------------------------------
    # Lazy cross-encoder loading
    # ------------------------------------------------------------------
    def _ensure_cross_encoder(self) -> bool:
        """Try to load the cross-encoder once; cache the result."""
        if self._use_cross_encoder is not None:
            return self._use_cross_encoder

        try:
            from sentence_transformers import CrossEncoder  # noqa: F811

            self._cross_encoder = CrossEncoder(self.model_name, device=self.device)
            self._use_cross_encoder = True
            logger.info("Cross-encoder loaded successfully: {m}", m=self.model_name)
        except Exception as exc:
            logger.warning(
                "Cross-encoder unavailable ({err}); using keyword fallback", err=exc
            )
            self._use_cross_encoder = False

        return self._use_cross_encoder

    # ------------------------------------------------------------------
    # Scoring strategies
    # ------------------------------------------------------------------
    @staticmethod
    def _keyword_score(query: str, text: str) -> float:
        """Compute a normalised keyword-overlap score between *query* and *text*.

        The score is a weighted Jaccard-like metric that gives extra credit to
        query tokens found as substrings inside document tokens (helps with
        morphological variants like ``running`` vs ``run``).
        """
        query_tokens = set(re.findall(r"\w+", query.lower()))
        doc_tokens = set(re.findall(r"\w+", text.lower()))

        if not query_tokens:
            return 0.0

        # Exact token overlap
        overlap = query_tokens & doc_tokens
        exact_score = len(overlap) / len(query_tokens)

        # Substring overlap (partial credit for morphological variants)
        substring_bonus = 0.0
        for qt in query_tokens:
            if qt in doc_tokens:
                continue
            for dt in doc_tokens:
                if qt in dt or dt in qt:
                    substring_bonus += 0.3 / len(query_tokens)
                    break

        # Keyword density in the document
        doc_len = max(len(doc_tokens), 1)
        density = len(overlap) / doc_len

        # Combine: 60 % exact match, 20 % substring bonus, 20 % density
        score = 0.6 * exact_score + 0.2 * min(substring_bonus, 0.4) + 0.2 * min(density, 1.0)
        return round(score, 6)

    def _cross_encode_score(self, query: str, candidates: list[dict]) -> list[float]:
        """Score *candidates* with the loaded cross-encoder."""
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        # Normalise to 0..1 range (sigmoid)
        import math

        return [1.0 / (1.0 + math.exp(-float(s))) for s in scores]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Return *candidates* sorted by relevance to *query* (highest first).

        Each candidate dict should contain at least a ``text`` key.  The
        reranker adds a ``rerank_score`` key and returns a new list.

        Parameters
        ----------
        query:
            The user query.
        candidates:
            List of candidate dicts (e.g. from a vector store).  Each dict may
            carry arbitrary metadata; only ``text`` is read by the reranker.

        Returns
        -------
        list[dict]
            The same candidate dicts, sorted by descending ``rerank_score``.
        """
        if not candidates:
            logger.debug("No candidates to rerank")
            return []

        use_ce = self._ensure_cross_encoder()

        if use_ce:
            scores = self._cross_encode_score(query, candidates)
            logger.debug(
                "Cross-encoder reranked {n} candidates (top score={s:.4f})",
                n=len(candidates),
                s=max(scores),
            )
        else:
            scores = [self._keyword_score(query, c.get("text", "")) for c in candidates]
            logger.debug(
                "Keyword fallback reranked {n} candidates (top score={s:.4f})",
                n=len(candidates),
                s=max(scores),
            )

        scored = []
        for candidate, score in zip(candidates, scores):
            entry = dict(candidate)  # shallow copy to avoid mutation
            entry["rerank_score"] = round(float(score), 6)
            scored.append(entry)

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored
