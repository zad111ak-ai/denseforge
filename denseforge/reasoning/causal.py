"""Causal Reasoning Engine: extract causal relations and answer 'why' / 'what-if' questions."""
from typing import Any, Callable
from loguru import logger


class CausalReasoningEngine:
    """Extracts causal chains from documents and supports counterfactual
    simulation for 'why' and 'what-if' queries.

    Args:
        llm_fn: Callable ``(prompt: str) -> dict``.
    """

    def __init__(self, llm_fn: Callable[[str], dict[str, Any]]):
        self.llm_fn = llm_fn
        # In-memory causal graph: node_id -> list of {effect, relation}
        self._causal_graph: dict[str, list[dict[str, Any]]] = {}
        # Document index: doc_id -> extracted causal triples
        self._doc_causes: dict[str, list[dict[str, Any]]] = {}
        logger.debug("CausalReasoningEngine initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_document(self, text: str, doc_id: str | int) -> list[dict[str, Any]]:
        """Extract causal relations from *text* and index them.

        Args:
            text: Document text.
            doc_id: Identifier for the document.

        Returns:
            List of causal triples, each a dict with keys:
                ``cause`` (str), ``effect`` (str), ``relation`` (str),
                ``confidence`` (float 0-1).
        """
        doc_id_str = str(doc_id)
        logger.debug("Extracting causal relations from doc {d}", d=doc_id_str)

        prompt = (
            "Extract all causal relations from the following text.  For each "
            "relation return cause, effect, relation type (e.g. 'causes', "
            "'prevents', 'increases', 'decreases'), and a confidence score.\n\n"
            "Text:\n{text}\n\n"
            "Respond in JSON: a list of objects with keys "
            '"cause", "effect", "relation", "confidence" (float 0-1).\n'
            "If no causal relations are found, return an empty list."
        ).format(text=text)

        try:
            result = self.llm_fn(prompt)
            if isinstance(result, dict):
                triples = result.get("relations", result.get("causal_relations", result.get("answer", [])))
            elif isinstance(result, list):
                triples = result
            else:
                triples = []
        except Exception as exc:
            logger.warning("LLM extract_from_document failed: {e}; using heuristic", e=exc)
            triples = self._heuristic_extract(text)

        if not isinstance(triples, list):
            triples = []

        # Normalise
        normalised = []
        for t in triples:
            if not isinstance(t, dict):
                continue
            entry = {
                "cause": str(t.get("cause", "")),
                "effect": str(t.get("effect", "")),
                "relation": str(t.get("relation", "causes")),
                "confidence": max(0.0, min(1.0, float(t.get("confidence", 0.5)))),
            }
            normalised.append(entry)

            # Update causal graph
            cause_key = entry["cause"].lower().strip()
            if cause_key not in self._causal_graph:
                self._causal_graph[cause_key] = []
            self._causal_graph[cause_key].append({
                "effect": entry["effect"],
                "relation": entry["relation"],
                "confidence": entry["confidence"],
            })

        self._doc_causes[doc_id_str] = normalised
        logger.info(
            "Extracted {n} causal triples from doc {d}",
            n=len(normalised), d=doc_id_str,
        )
        return normalised

    def query_causal_chain(self, effect: str, max_depth: int = 5) -> dict[str, Any]:
        """Trace backwards from *effect* through the causal graph to find
        root causes.

        Args:
            effect: The effect to trace back.
            max_depth: Maximum chain depth (default 5).

        Returns:
            Dict with:
                ``effect`` (str) – the queried effect.
                ``chains`` (list[list[dict]]) – each chain is a list of
                    causal links from root cause to the queried effect.
                ``root_causes`` (list[str]) – deduplicated root causes.
                ``confidence`` (float) – average confidence across all chains.
        """
        logger.debug(
            "Querying causal chain for effect: {e} (max_depth={d})",
            e=effect, d=max_depth,
        )

        # BFS / DFS through the causal graph
        chains: list[list[dict[str, Any]]] = []
        self._trace(effect.lower().strip(), [], chains, max_depth, set())

        # If graph is empty, ask the LLM
        if not self._causal_graph and not chains:
            chains = self._llm_causal_chain(effect, max_depth)

        root_causes: list[str] = []
        for chain in chains:
            if chain:
                root_causes.append(chain[0]["cause"])
        root_causes = list(dict.fromkeys(root_causes))  # deduplicate, preserve order

        all_confs = [link["confidence"] for chain in chains for link in chain]
        avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0

        result = {
            "effect": effect,
            "chains": chains,
            "root_causes": root_causes,
            "confidence": round(avg_confidence, 4),
        }

        logger.info(
            "Found {c} causal chains with {r} root causes for effect '{e}'",
            c=len(chains), r=len(root_causes), e=effect[:60],
        )
        return result

    def simulate_counterfactual(
        self, intervention: str, target: str
    ) -> dict[str, Any]:
        """Simulate what would happen if *intervention* were applied, in
        relation to *target*.

        Args:
            intervention: The counterfactual change (e.g. "remove funding").
            target: The outcome being evaluated (e.g. "research output").

        Returns:
            Dict with:
                ``intervention`` (str), ``target`` (str),
                ``predicted_effect`` (str) – LLM's prediction,
                ``confidence`` (float 0-1),
                ``reasoning_chain`` (list[str]) – step-by-step reasoning,
                ``supporting_evidence`` (list[dict]) – relevant causal links.
        """
        logger.debug(
            "Simulating counterfactual: if '{i}' then what about '{t}'",
            i=intervention, t=target,
        )

        # Gather supporting evidence from the causal graph
        evidence = self._gather_evidence(intervention, target)

        evidence_text = ""
        if evidence:
            evidence_text = "\nRelevant causal links from existing knowledge:\n"
            for e in evidence:
                evidence_text += (
                    f"  - {e['cause']} -> ({e['relation']}) -> {e['effect']} "
                    f"[conf={e['confidence']:.2f}]\n"
                )

        prompt = (
            "You are a causal reasoning engine.  Simulate the following "
            "counterfactual scenario step by step.\n\n"
            "Intervention (what-if): {intervention}\n"
            "Target outcome to evaluate: {target}\n"
            "{evidence}"
            "\nProvide your analysis in JSON with:\n"
            '  "predicted_effect": string describing the likely outcome,\n'
            '  "confidence": float 0-1,\n'
            '  "reasoning_chain": list of step-by-step reasoning strings,\n'
            '  "supporting_evidence": list of relevant causal links '
            '(cause, effect, relation, confidence).\n'
        ).format(
            intervention=intervention,
            target=target,
            evidence=evidence_text,
        )

        try:
            result = self.llm_fn(prompt)
            if not isinstance(result, dict):
                result = {"predicted_effect": str(result), "confidence": 0.3, "reasoning_chain": [], "supporting_evidence": evidence}
        except Exception as exc:
            logger.warning("LLM simulate_counterfactual failed: {e}", e=exc)
            result = self._fallback_counterfactual(intervention, target, evidence)

        result.setdefault("intervention", intervention)
        result.setdefault("target", target)
        result.setdefault("predicted_effect", "")
        result.setdefault("confidence", 0.0)
        result.setdefault("reasoning_chain", [])
        result.setdefault("supporting_evidence", evidence)

        logger.info(
            "Counterfactual simulation complete: confidence={c:.2f}",
            c=result["confidence"],
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trace(
        self,
        effect_key: str,
        current_chain: list[dict[str, Any]],
        all_chains: list[list[dict[str, Any]]],
        max_depth: int,
        visited: set[str],
    ) -> None:
        """Recursive trace through the causal graph."""
        if max_depth <= 0 or effect_key in visited:
            return
        visited.add(effect_key)

        found = False
        for cause_key, links in self._causal_graph.items():
            for link in links:
                if link["effect"].lower().strip() == effect_key:
                    found = True
                    new_link = {
                        "cause": cause_key,
                        "effect": link["effect"],
                        "relation": link["relation"],
                        "confidence": link["confidence"],
                    }
                    new_chain = [new_link] + current_chain
                    self._trace(cause_key, new_chain, all_chains, max_depth - 1, visited.copy())

        if not found and current_chain:
            all_chains.append(current_chain)

    def _llm_causal_chain(self, effect: str, max_depth: int) -> list[list[dict[str, Any]]]:
        """Ask the LLM to produce a causal chain when graph data is unavailable."""
        prompt = (
            "Given general world knowledge, provide a causal chain that leads "
            "to the following effect: '{effect}'.  Trace back up to {depth} "
            "steps to root causes.\n\n"
            "Respond in JSON: a list of chains.  Each chain is a list of objects "
            'with "cause", "effect", "relation", "confidence" (float 0-1).\n'
        ).format(effect=effect, depth=max_depth)

        try:
            result = self.llm_fn(prompt)
            if isinstance(result, dict):
                return result.get("chains", result.get("causal_chains", []))
            elif isinstance(result, list):
                return result
        except Exception as exc:
            logger.warning("LLM causal chain query failed: {e}", e=exc)
        return []

    def _gather_evidence(self, intervention: str, target: str) -> list[dict[str, Any]]:
        """Collect relevant causal links from the graph for a counterfactual query."""
        evidence: list[dict[str, Any]] = []
        intervention_words = set(intervention.lower().split())
        target_words = set(target.lower().split())

        for cause_key, links in self._causal_graph.items():
            for link in links:
                effect_text = link["effect"].lower()
                cause_text = cause_key.lower()
                # Check relevance by word overlap
                cause_overlap = len(intervention_words & set(cause_text.split()))
                effect_overlap = len(target_words & set(effect_text.split()))
                if cause_overlap > 0 or effect_overlap > 0:
                    evidence.append({
                        "cause": cause_key,
                        "effect": link["effect"],
                        "relation": link["relation"],
                        "confidence": link["confidence"],
                    })
        return evidence

    @staticmethod
    def _heuristic_extract(text: str) -> list[dict[str, Any]]:
        """Keyword-based extraction of likely causal sentences."""
        import re
        causal_patterns = [
            r"(.+?)\s+(?:causes?|leads?\s+to|results?\s+in|triggers?)\s+(.+)",
            r"(.+?)\s+(?:prevents?|stops?|inhibits?|reduces?)\s+(.+)",
            r"(?:because|since|due\s+to)\s+(.+?),\s+(.+)",
        ]
        triples: list[dict[str, Any]] = []
        sentences = re.split(r"[.!?\n]", text)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            for pattern in causal_patterns:
                m = re.search(pattern, sent, re.IGNORECASE)
                if m:
                    triples.append({
                        "cause": m.group(1).strip(),
                        "effect": m.group(2).strip(),
                        "relation": "causes",
                        "confidence": 0.4,
                    })
                    break  # one match per sentence
        return triples

    @staticmethod
    def _fallback_counterfactual(
        intervention: str, target: str, evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Produce a basic counterfactual response when the LLM fails."""
        if evidence:
            best = max(evidence, key=lambda e: e["confidence"])
            return {
                "intervention": intervention,
                "target": target,
                "predicted_effect": (
                    f"Based on the strongest causal link ({best['cause']} -> "
                    f"{best['effect']}), applying '{intervention}' may affect "
                    f"'{target}'.  Manual review recommended."
                ),
                "confidence": best["confidence"] * 0.5,
                "reasoning_chain": [
                    f"Found relevant link: {best['cause']} ({best['relation']}) {best['effect']}",
                    "Insufficient data for precise prediction.",
                ],
                "supporting_evidence": evidence,
            }
        return {
            "intervention": intervention,
            "target": target,
            "predicted_effect": "Insufficient causal data to simulate this scenario.",
            "confidence": 0.0,
            "reasoning_chain": [],
            "supporting_evidence": [],
        }
