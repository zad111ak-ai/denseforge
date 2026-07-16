"""Conflict Resolution Engine: detect and resolve conflicting information across documents."""
from typing import Any, Callable
from loguru import logger


class ConflictResolutionEngine:
    """Detects conflicting claims across retrieved documents and resolves them.

    Args:
        llm_fn: Callable ``(prompt: str) -> dict``.  Used for conflict
            detection and resolution.
    """

    def __init__(self, llm_fn: Callable[[str], dict[str, Any]]):
        self.llm_fn = llm_fn
        logger.debug("ConflictResolutionEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_conflicts(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Analyse *docs* and return a list of detected conflicts.

        Each conflict dict contains:
            ``claim`` (str)          – the factual claim that is disputed.
            ``positions`` (list)     – list of {doc_id, doc_index, stance, excerpt}.
            ``severity`` (str)       – "low", "medium", or "high".
            ``topic`` (str)          – short topic label.

        Args:
            docs: List of documents.  Each must have at least ``text`` (or
                ``content``) and ideally an ``id`` / ``doc_id`` key.

        Returns:
            List of conflict dicts (may be empty if no conflicts detected).
        """
        if not docs:
            logger.info("No documents provided; nothing to check for conflicts.")
            return []

        logger.debug("Detecting conflicts across {n} documents", n=len(docs))

        doc_block = "\n\n".join(
            f"[Doc {i} id={d.get('id', d.get('doc_id', i))}]\n"
            f"{d.get('text', d.get('content', str(d)))}"
            for i, d in enumerate(docs)
        )

        prompt = (
            "You are a fact-checking analyst.  Given a set of documents about "
            "a similar topic, identify any factual conflicts, contradictions, "
            "or disagreements between them.\n\n"
            "Documents:\n{docs}\n\n"
            "Respond in JSON: a list of conflicts.  Each conflict is an object with:\n"
            '  "claim": the disputed factual claim (string),\n'
            '  "positions": list of objects, each with "doc_index" (int), '
            '"doc_id" (string or int), "stance" ("supports"|"contradicts"|"neutral"), '
            'and "excerpt" (relevant text snippet),\n'
            '  "severity": "low"|"medium"|"high",\n'
            '  "topic": short topic label.\n\n'
            "If there are no conflicts, return an empty list."
        ).format(docs=doc_block)

        try:
            result = self.llm_fn(prompt)
            if isinstance(result, dict):
                conflicts = result.get("conflicts", result.get("answer", []))
            elif isinstance(result, list):
                conflicts = result
            else:
                conflicts = []
        except Exception as exc:
            logger.warning("LLM detect_conflicts failed: {e}; falling back to heuristic", e=exc)
            conflicts = self._heuristic_detect(docs)

        if not isinstance(conflicts, list):
            conflicts = []

        logger.info("Detected {n} conflicts", n=len(conflicts))
        return conflicts

    def resolve(
        self,
        conflicts: list[dict[str, Any]],
        docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Resolve detected conflicts and produce a reconciled summary.

        Args:
            conflicts: Output of :py:meth:`detect_conflicts`.
            docs: The original document list.

        Returns:
            Dict with:
                ``resolved_claims`` (list[dict]) – for each conflict, the
                    reconciled claim with supporting doc indices.
                ``overall_confidence`` (float) – confidence in the resolution.
                ``notes`` (str) – free-text notes about the resolution process.
        """
        if not conflicts:
            logger.info("No conflicts to resolve.")
            return {
                "resolved_claims": [],
                "overall_confidence": 1.0,
                "notes": "No conflicts detected.",
            }

        logger.debug("Resolving {n} conflicts", n=len(conflicts))

        doc_block = "\n\n".join(
            f"[Doc {i} id={d.get('id', d.get('doc_id', i))}]\n"
            f"{d.get('text', d.get('content', str(d)))}"
            for i, d in enumerate(docs)
        )

        conflict_block = "\n".join(
            f"- Conflict {i}: {c.get('claim', 'N/A')} "
            f"(severity={c.get('severity', 'medium')})"
            for i, c in enumerate(conflicts)
        )

        prompt = (
            "You are a fact-reconciliation expert.  Given a list of documents "
            "and detected factual conflicts between them, produce a reconciled "
            "set of claims.  Prefer information from more reliable or "
            "authoritative sources when available.\n\n"
            "Documents:\n{docs}\n\n"
            "Conflicts:\n{conflicts}\n\n"
            "Respond in JSON with:\n"
            '  "resolved_claims": list of objects each with "claim" (string), '
            '"resolution" (string – the reconciled statement), '
            '"supporting_docs" (list of doc indices),\n'
            '  "overall_confidence": float 0-1,\n'
            '  "notes": string with any caveats.\n'
        ).format(docs=doc_block, conflicts=conflict_block)

        try:
            result = self.llm_fn(prompt)
            if not isinstance(result, dict):
                result = self._default_resolution(conflicts)
        except Exception as exc:
            logger.warning("LLM resolve failed: {e}; using default resolution", e=exc)
            result = self._default_resolution(conflicts)

        result.setdefault("resolved_claims", [])
        result.setdefault("overall_confidence", 0.5)
        result.setdefault("notes", "Resolution produced with LLM assistance.")

        logger.info(
            "Resolved {n} claims with confidence {c:.2f}",
            n=len(result["resolved_claims"]), c=result["overall_confidence"],
        )
        return result

    # ------------------------------------------------------------------
    # Fallback heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_detect(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect potential conflicts by looking for negation-keyword clashes."""
        conflicts: list[dict[str, Any]] = []
        negation_markers = {"not", "never", "no", "cannot", "cannot", "isn't", "aren't", "wasn't"}

        for i, doc_a in enumerate(docs):
            text_a = doc_a.get("text", doc_a.get("content", str(doc_a))).lower()
            words_a = set(text_a.split())
            has_negation_a = bool(words_a & negation_markers)

            for j in range(i + 1, len(docs)):
                doc_b = docs[j]
                text_b = doc_b.get("text", doc_b.get("content", str(doc_b))).lower()
                words_b = set(text_b.split())
                has_negation_b = bool(words_b & negation_markers)

                # Simple heuristic: if one document contains negation markers
                # and shares significant vocabulary with the other, flag it.
                shared = words_a & words_b
                if has_negation_a != has_negation_b and len(shared) >= 5:
                    conflicts.append({
                        "claim": f"Potential factual tension between Doc {i} and Doc {j}",
                        "positions": [
                            {"doc_index": i, "doc_id": doc_a.get("id", i), "stance": "unknown", "excerpt": doc_a.get("text", str(doc_a))[:200]},
                            {"doc_index": j, "doc_id": doc_b.get("id", j), "stance": "unknown", "excerpt": doc_b.get("text", str(doc_b))[:200]},
                        ],
                        "severity": "medium",
                        "topic": ", ".join(list(shared)[:5]),
                    })

        return conflicts

    @staticmethod
    def _default_resolution(conflicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Produce a trivial default resolution when the LLM is unavailable."""
        resolved = []
        for c in conflicts:
            positions = c.get("positions", [])
            all_doc_indices = [p.get("doc_index", -1) for p in positions]
            resolved.append({
                "claim": c.get("claim", "Unknown claim"),
                "resolution": "Unable to resolve automatically; manual review needed.",
                "supporting_docs": [i for i in all_doc_indices if i >= 0],
            })
        return {
            "resolved_claims": resolved,
            "overall_confidence": 0.1,
            "notes": "Default resolution; LLM was unavailable.",
        }
