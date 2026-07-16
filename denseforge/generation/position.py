"""PositionAwareAssembler — arrange context to combat the lost-in-the-middle problem."""
from __future__ import annotations

from typing import Literal

from loguru import logger


# Rough word → token ratio used for budget estimation when no tokenizer is available.
_AVG_WORDS_PER_TOKEN = 1.3


class PositionAwareAssembler:
    """Arrange context documents in positions that mitigate lost-in-the-middle.

    Research shows LLMs attend disproportionately to information at the
    *beginning* and *end* of a long context window.  The U-shape strategy
    places the most relevant documents at those positions while relegating
    less-relevant ones to the middle.
    """

    def __init__(self) -> None:
        logger.info("PositionAwareAssembler initialised")

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Very rough token count based on whitespace-split words."""
        return max(1, int(len(text.split()) / _AVG_WORDS_PER_TOKEN))

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------
    @staticmethod
    def _u_shape_order(docs: list[dict]) -> list[dict]:
        """Place most-relevant docs at the start and end, least in the middle.

        Assumes *docs* are already sorted by relevance (descending).

        The interleaving pattern is:
          position 0  → doc[0]  (best)
          position n-1→ doc[1]  (2nd best)
          position 1  → doc[2]
          position n-2→ doc[3]
          ...

        This creates a U-shaped attention profile.
        """
        if len(docs) <= 2:
            return list(docs)

        result: list[dict | None] = [None] * len(docs)
        left, right = 0, len(docs) - 1

        for i, doc in enumerate(docs):
            if i % 2 == 0:
                result[left] = doc
                left += 1
            else:
                result[right] = doc
                right -= 1

        return [d for d in result if d is not None]  # type: ignore[misc]

    @staticmethod
    def _flat_order(docs: list[dict]) -> list[dict]:
        """No reordering — return documents in original sequence."""
        return list(docs)

    @staticmethod
    def _recency_order(docs: list[dict]) -> list[dict]:
        """Place the most recently created documents first."""
        sorted_docs = sorted(
            docs,
            key=lambda d: d.get("created_at", 0),
            reverse=True,
        )
        return sorted_docs

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------
    def assemble_context(
        self,
        docs: list[dict],
        max_tokens: int = 8000,
        strategy: Literal["u_shape", "flat", "recency"] = "u_shape",
    ) -> str:
        """Assemble document texts into a single context string.

        Parameters
        ----------
        docs:
            List of document dicts, each with a ``text`` key.  Documents are
            assumed to be pre-sorted by relevance (descending) unless *strategy*
            specifies otherwise.
        max_tokens:
            Approximate token budget for the assembled context.
        strategy:
            ``"u_shape"``  — U-shaped placement (default, best for RAG).
            ``"flat"``     — original order, first-fit into budget.
            ``"recency"``  — newest first, then fill into budget.

        Returns
        -------
        str
            A single string containing the concatenated document texts,
            separated by ``\\n---\\n`` delimiters, within the token budget.
        """
        if not docs:
            logger.debug("No documents to assemble")
            return ""

        if strategy == "u_shape":
            ordered = self._u_shape_order(docs)
        elif strategy == "recency":
            ordered = self._recency_order(docs)
        else:
            ordered = self._flat_order(docs)

        # --- Greedy fill within token budget ---
        parts: list[str] = []
        tokens_used = 0

        for doc in ordered:
            text = doc.get("text", "")
            if not text:
                continue

            est_tokens = self._estimate_tokens(text)

            if tokens_used + est_tokens > max_tokens:
                # Try to fit a truncated version
                remaining_budget = max_tokens - tokens_used
                if remaining_budget > 50:
                    words = text.split()
                    truncated_words = words[: int(remaining_budget * _AVG_WORDS_PER_TOKEN)]
                    truncated_text = " ".join(truncated_words) + "…"
                    parts.append(truncated_text)
                    tokens_used = max_tokens  # budget exhausted
                logger.debug(
                    "Truncated output at {n}/{budget} tokens (strategy={s})",
                    n=tokens_used,
                    budget=max_tokens,
                    s=strategy,
                )
                break

            parts.append(text)
            tokens_used += est_tokens

        context = "\n---\n".join(parts)

        logger.debug(
            "Assembled context: {n} docs, ~{tok} tokens (strategy={s})",
            n=len(parts),
            tok=tokens_used,
            s=strategy,
        )
        return context
