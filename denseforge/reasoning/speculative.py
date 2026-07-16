"""Speculative RAG: generate multiple draft answers and pick the best."""
from typing import Any, Callable
from loguru import logger


class SpeculativeRAG:
    """Speculative RAG that generates multiple draft answers via an LLM and
    selects the highest-scoring one.

    Args:
        llm_fn: Callable with signature ``(prompt: str) -> dict``.  The dict
            returned must contain at least an ``answer`` key.  Optionally it
            may contain a ``score`` key (float 0-1); if absent, the answer is
            scored by simple heuristics.
        n_drafts: Number of speculative drafts to generate (default 3).
    """

    def __init__(self, llm_fn: Callable[[str], dict[str, Any]], n_drafts: int = 3):
        self.llm_fn = llm_fn
        self.n_drafts = max(n_drafts, 1)
        logger.debug("SpeculativeRAG initialised with n_drafts={n}", n=self.n_drafts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_answer(self, draft: dict[str, Any], query: str) -> float:
        """Return a confidence score for a draft.  If the LLM already supplied
        one, clamp it to [0, 1] and return it.  Otherwise fall back to a
        length-and-relevance heuristic."""
        if "score" in draft and draft["score"] is not None:
            return max(0.0, min(1.0, float(draft["score"])))

        answer_text = str(draft.get("answer", ""))
        if not answer_text:
            return 0.0

        # Simple heuristic: longer, more detailed answers that echo query
        # terms score higher.
        length_score = min(len(answer_text) / 500.0, 1.0)
        query_words = set(query.lower().split())
        answer_words = set(answer_text.lower().split())
        overlap = len(query_words & answer_words)
        relevance_score = min(overlap / max(len(query_words), 1), 1.0)
        return round(0.5 * length_score + 0.5 * relevance_score, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, query: str, context: str) -> dict[str, Any]:
        """Generate *n_drafts* speculative answers and return the best one.

        Args:
            query: The user question.
            context: Context string (e.g. retrieved documents) to ground the
                answer.

        Returns:
            Dict with keys:
                ``answer``  – the selected best answer text.
                ``score``   – its confidence score (float 0-1).
                ``drafts``  – list of all generated drafts (for audit).
                ``selected_index`` – index of the winning draft.
        """
        logger.debug(
            "SpeculativeRAG generating {n} drafts for query: {q}",
            n=self.n_drafts, q=query[:80],
        )

        prompt_template = (
            "You are a helpful assistant. Answer the following question using ONLY "
            "the provided context.\n\n"
            "Question: {query}\n\n"
            "Context:\n{context}\n\n"
            "Provide a concise, accurate answer."
        )

        drafts: list[dict[str, Any]] = []
        for i in range(self.n_drafts):
            prompt = prompt_template.format(query=query, context=context)
            try:
                result = self.llm_fn(prompt)
                if not isinstance(result, dict):
                    result = {"answer": str(result)}
            except Exception as exc:
                logger.warning("Draft {i} LLM call failed: {e}", i=i, e=exc)
                result = {"answer": "", "score": 0.0}

            result["score"] = self._score_answer(result, query)
            result["draft_index"] = i
            drafts.append(result)

        if not drafts:
            return {"answer": "", "score": 0.0, "drafts": [], "selected_index": -1}

        # Select the best draft
        best_idx = max(range(len(drafts)), key=lambda i: drafts[i]["score"])
        best = drafts[best_idx]

        logger.info(
            "SpeculativeRAG selected draft {i}/{n} with score {s:.4f}",
            i=best_idx, n=self.n_drafts, s=best["score"],
        )

        return {
            "answer": best.get("answer", ""),
            "score": best["score"],
            "drafts": drafts,
            "selected_index": best_idx,
        }
