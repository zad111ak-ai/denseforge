"""AttributionGenerator — generate answers with inline citations."""
from __future__ import annotations

import re
from typing import Callable

from loguru import logger


class AttributionGenerator:
    """Generate answers with numbered citations sourced from retrieved chunks.

    Parameters
    ----------
    llm_fn:
        A callable ``(prompt: str) -> str`` that invokes the underlying LLM.
        This keeps the class decoupled from any specific provider.
    """

    def __init__(self, llm_fn: Callable[[str], str]):
        self.llm_fn = llm_fn
        logger.info("AttributionGenerator initialised")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prompt(query: str, chunks: list[dict]) -> str:
        """Compose a prompt that instructs the LLM to cite sources."""

        context_parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", chunk.get("doc_id", f"Source {idx}"))
            text = chunk.get("text", "")
            context_parts.append(f"[{idx}] ({title}) {text}")

        context_block = "\n\n".join(context_parts)

        return (
            "You are a helpful assistant that answers questions based ONLY on "
            "the provided sources below.  When you use information from a "
            "source, cite it inline using the reference number like [1], [2], "
            "etc.\n\n"
            "SOURCES:\n"
            f"{context_block}\n\n"
            "QUESTION: {query}\n\n"
            "INSTRUCTIONS:\n"
            "- Answer using ONLY the information in the sources above.\n"
            "- Cite every factual claim with its source number, e.g. [1].\n"
            "- If the sources do not contain enough information, say so.\n"
            "- Be concise and accurate.\n\n"
            "ANSWER:"
        ).format(query=query)

    @staticmethod
    def _extract_citations(answer: str, num_sources: int) -> list[dict]:
        """Parse inline citation markers from the generated answer.

        Returns a list of citation dicts with keys ``ref`` (the number) and
        ``count`` (how many times that source was referenced).
        """
        pattern = re.compile(r"\[(\d+)\]")
        raw_refs = [int(m.group(1)) for m in pattern.finditer(answer)]

        citation_counts: dict[int, int] = {}
        for ref in raw_refs:
            if 1 <= ref <= num_sources:
                citation_counts[ref] = citation_counts.get(ref, 0) + 1

        citations = [
            {"ref": ref, "count": count}
            for ref, count in sorted(citation_counts.items())
        ]
        return citations

    @staticmethod
    def _compute_coverage(chunks: list[dict], citations: list[dict]) -> float:
        """Return the fraction of chunks that were cited at least once."""
        if not chunks:
            return 0.0
        cited_refs = {c["ref"] for c in citations}
        return round(len(cited_refs) / len(chunks), 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate an answer with attributions.

        Parameters
        ----------
        query:
            The user question.
        chunks:
            Retrieved context chunks.  Each dict should contain ``text`` and
            optionally ``title`` or ``doc_id``.

        Returns
        -------
        dict
            ``answer``  — the generated text (with inline ``[N]`` markers).
            ``citations`` — list of ``{"ref": int, "count": int}`` dicts.
            ``num_sources`` — how many chunks were provided.
            ``coverage`` — fraction of chunks that were cited.
        """
        if not chunks:
            logger.warning("No chunks provided; returning empty answer")
            return {
                "answer": "I could not find any relevant information to answer this question.",
                "citations": [],
                "num_sources": 0,
                "coverage": 0.0,
            }

        prompt = self._build_prompt(query, chunks)

        logger.debug(
            "Generating attributed answer for query ({q_len} chars, {n} sources)",
            q_len=len(query),
            n=len(chunks),
        )

        answer = self.llm_fn(prompt)

        citations = self._extract_citations(answer, num_sources=len(chunks))
        coverage = self._compute_coverage(chunks, citations)

        logger.info(
            "Attribution generated: {cites} citations covering {cov:.0%} of sources",
            cites=len(citations),
            cov=coverage,
        )

        return {
            "answer": answer,
            "citations": citations,
            "num_sources": len(chunks),
            "coverage": coverage,
        }
