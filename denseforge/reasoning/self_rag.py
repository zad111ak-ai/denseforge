"""Self-RAG Reflection: evaluate retrieval sufficiency and suggest query rewrites."""
from typing import Any, Callable
from loguru import logger


class SelfRAGReflection:
    """Self-RAG reflection module that judges whether retrieved documents are
    sufficient to answer a query, and optionally suggests a rewritten query.

    Args:
        llm_fn: Callable ``(prompt: str) -> dict``.  Must return at least the
            keys the individual methods expect.
    """

    def __init__(self, llm_fn: Callable[[str], dict[str, Any]]):
        self.llm_fn = llm_fn
        logger.debug("SelfRAGReflection initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_retrieval(self, query: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Judge whether the retrieved documents are sufficient to answer *query*.

        Args:
            query: The user question.
            docs: List of retrieved documents (each a dict with at least a
                ``text`` or ``content`` key).

        Returns:
            Dict with:
                ``is_sufficient`` (bool) – True if the docs provide enough info.
                ``reason`` (str)        – human-readable explanation.
                ``confidence`` (float)  – 0-1 confidence in the judgement.
                ``missing_topics`` (list[str]) – topics the docs fail to cover.
        """
        logger.debug(
            "Evaluating retrieval for query: {q} ({n} docs)",
            q=query[:80], n=len(docs),
        )

        doc_texts = "\n\n".join(
            f"[Doc {i}] {d.get('text', d.get('content', str(d)))}"
            for i, d in enumerate(docs)
        )

        prompt = (
            "You are a retrieval-quality judge.  Given a user query and a set of "
            "retrieved documents, determine whether the documents collectively "
            "contain enough information to fully answer the query.\n\n"
            "Query: {query}\n\n"
            "Documents:\n{docs}\n\n"
            "Respond in JSON with these keys:\n"
            '  "is_sufficient": boolean,\n'
            '  "reason": string (brief explanation),\n'
            '  "confidence": float between 0 and 1,\n'
            '  "missing_topics": list of strings (topics not covered by the docs)\n'
        ).format(query=query, docs=doc_texts)

        try:
            result = self.llm_fn(prompt)
            if not isinstance(result, dict):
                logger.warning("LLM returned non-dict; falling back to heuristic")
                result = self._heuristic_evaluate(query, docs)
        except Exception as exc:
            logger.warning("LLM evaluate_retrieval failed: {e}; using heuristic", e=exc)
            result = self._heuristic_evaluate(query, docs)

        # Ensure required keys
        result.setdefault("is_sufficient", False)
        result.setdefault("reason", "Evaluation failed; defaulting to insufficient.")
        result.setdefault("confidence", 0.0)
        result.setdefault("missing_topics", [])

        logger.info(
            "Retrieval evaluation: sufficient={s}, confidence={c:.2f}",
            s=result["is_sufficient"], c=result["confidence"],
        )
        return result

    def suggest_rewrite(self, query: str) -> str:
        """Ask the LLM to suggest an improved retrieval query.

        Args:
            query: The original user query.

        Returns:
            A rewritten query string optimised for better retrieval.
        """
        logger.debug("Suggesting rewrite for query: {q}", q=query[:80])

        prompt = (
            "You are a search-query optimiser.  Given a user question that may "
            "be vague, overly broad, or phrased sub-optimally for retrieval, "
            "produce a concise rewrite that would retrieve better documents.\n\n"
            "Original query: {query}\n\n"
            "Return ONLY the rewritten query text, nothing else."
        ).format(query=query)

        try:
            result = self.llm_fn(prompt)
            rewrite = result.get("answer", result.get("query", str(result))) if isinstance(result, dict) else str(result)
        except Exception as exc:
            logger.warning("LLM suggest_rewrite failed: {e}", e=exc)
            rewrite = query  # fall back to original

        logger.info("Query rewrite: {r}", r=rewrite[:120])
        return str(rewrite).strip()

    # ------------------------------------------------------------------
    # Fallback heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_evaluate(query: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Simple keyword-overlap heuristic when the LLM is unavailable."""
        query_words = set(query.lower().split())
        combined_doc_words: set[str] = set()
        for d in docs:
            text = d.get("text", d.get("content", str(d)))
            combined_doc_words.update(text.lower().split())

        overlap = len(query_words & combined_doc_words)
        coverage = overlap / max(len(query_words), 1)
        is_sufficient = coverage >= 0.4

        return {
            "is_sufficient": is_sufficient,
            "reason": (
                "Heuristic: keyword coverage is {:.0%}.".format(coverage)
            ),
            "confidence": round(min(coverage, 1.0), 3),
            "missing_topics": [
                w for w in query_words if w not in combined_doc_words
            ],
        }
