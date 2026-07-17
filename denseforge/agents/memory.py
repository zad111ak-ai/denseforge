"""ThreeTierMemory — Episodic + Semantic + Procedural memory for agents."""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class _EpisodicMemory:
    """Time-stamped interaction records per user.

    Episodes are the raw experiential memories: what happened, when, and
    how it turned out.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = defaultdict(list)

    def store(
        self,
        user_id: str,
        query: str,
        response: str,
        context: str,
        outcome: str,
        feedback: Optional[str] = None,
    ) -> dict:
        episode = {
            "episode_id": str(uuid.uuid4()),
            "user_id": user_id,
            "query": query,
            "response": response,
            "context": context,
            "outcome": outcome,
            "feedback": feedback,
            "timestamp": time.time(),
        }
        self._store[user_id].append(episode)
        logger.debug(
            "Stored episode {eid} for user {uid}",
            eid=episode["episode_id"][:8],
            uid=user_id,
        )
        return episode

    def retrieve(self, user_id: str, top_k: int = 10) -> list[dict]:
        episodes = self._store.get(user_id, [])
        # Return most recent first
        return list(reversed(episodes[-top_k:]))

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            return len(self._store.get(user_id, []))
        return sum(len(v) for v in self._store.values())


class _SemanticMemory:
    """Aggregated knowledge distilled from episodes.

    Stores key facts, preferences, and patterns that generalise across
    interactions.
    """

    def __init__(self) -> None:
        self._facts: dict[str, list[dict]] = defaultdict(list)
        self._patterns: dict[str, dict] = {}

    def add_fact(self, user_id: str, fact: str, source_ep_id: str, confidence: float = 0.8):
        entry = {
            "fact": fact,
            "source_ep_id": source_ep_id,
            "confidence": confidence,
            "created_at": time.time(),
        }
        self._facts[user_id].append(entry)
        logger.debug("Added semantic fact for user {uid}: {f}", uid=user_id, f=fact[:80])

    def get_facts(self, user_id: str, top_k: int = 20) -> list[dict]:
        facts = self._facts.get(user_id, [])
        # Sort by confidence descending, then recency
        sorted_facts = sorted(
            facts, key=lambda f: (f["confidence"], f["created_at"]), reverse=True,
        )
        return sorted_facts[:top_k]

    def update_pattern(self, user_id: str, pattern_name: str, pattern_data: dict):
        key = f"{user_id}:{pattern_name}"
        self._patterns[key] = {**pattern_data, "updated_at": time.time()}

    def get_pattern(self, user_id: str, pattern_name: str) -> Optional[dict]:
        return self._patterns.get(f"{user_id}:{pattern_name}")

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            return len(self._facts.get(user_id, []))
        return sum(len(v) for v in self._facts.values())


class _ProceduralMemory:
    """Learned procedures and skill traces.

    Tracks which strategies / action sequences have been successful for
    particular task types so the agent can replay them.
    """

    def __init__(self) -> None:
        self._procedures: dict[str, list[dict]] = defaultdict(list)

    def store(
        self,
        user_id: str,
        task_type: str,
        steps: list[str],
        success: bool,
        metadata: Optional[dict] = None,
    ) -> dict:
        entry = {
            "procedure_id": str(uuid.uuid4()),
            "user_id": user_id,
            "task_type": task_type,
            "steps": steps,
            "success": success,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "uses": 0,
        }
        self._procedures[f"{user_id}:{task_type}"].append(entry)
        logger.debug(
            "Stored procedure for {uid}/{tt} (success={s})",
            uid=user_id,
            tt=task_type,
            s=success,
        )
        return entry

    def retrieve(
        self, user_id: str, task_type: str, only_successful: bool = True,
    ) -> list[dict]:
        key = f"{user_id}:{task_type}"
        procedures = self._procedures.get(key, [])
        if only_successful:
            procedures = [p for p in procedures if p["success"]]
        # Sort by recency
        return list(reversed(procedures))

    def increment_uses(self, user_id: str, task_type: str, procedure_id: str):
        key = f"{user_id}:{task_type}"
        for proc in self._procedures.get(key, []):
            if proc["procedure_id"] == procedure_id:
                proc["uses"] += 1
                break

    def count(self, user_id: Optional[str] = None) -> int:
        if user_id:
            return sum(
                len(v)
                for k, v in self._procedures.items()
                if k.startswith(f"{user_id}:")
            )
        return sum(len(v) for v in self._procedures.values())


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ThreeTierMemory:
    """Three-tier memory system combining Episodic, Semantic, and Procedural stores.

    Parameters
    ----------
    embedder:
        An object with an ``encode(text) -> np.ndarray`` method used for
        similarity-based retrieval from episodic and semantic memory.  May be
        ``None`` to fall back on keyword matching.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder
        self.episodic = _EpisodicMemory()
        self.semantic = _SemanticMemory()
        self.procedural = _ProceduralMemory()
        logger.info("ThreeTierMemory initialised (embedder={e})", e=type(embedder).__name__ if embedder else None)

    # ------------------------------------------------------------------
    # Episodic
    # ------------------------------------------------------------------
    def store_episode(
        self,
        user_id: str,
        query: str,
        response: str,
        context: str,
        outcome: str,
        feedback: Optional[str] = None,
    ) -> dict:
        """Store an interaction episode across all three tiers.

        Returns the newly created episode dict.
        """
        episode = self.episodic.store(
            user_id=user_id,
            query=query,
            response=response,
            context=context,
            outcome=outcome,
            feedback=feedback,
        )

        # Auto-distill interesting facts into semantic memory
        if outcome in ("success", "positive") or feedback in ("positive", "helpful"):
            self.semantic.add_fact(
                user_id=user_id,
                fact=f"Successfully answered: {query[:120]}",
                source_ep_id=episode["episode_id"],
                confidence=0.9,
            )

        # Track successful procedures
        if outcome in ("success", "positive"):
            self.procedural.store(
                user_id=user_id,
                task_type=self._classify_task(query),
                steps=[f"query: {query[:100]}"],
                success=True,
            )

        return episode

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query_memory(self, user_id: str, query: str) -> dict:
        """Query all three memory tiers and return a combined context.

        Returns a dict with ``episodes``, ``facts``, ``procedures``, and
        ``context_str`` (a formatted string suitable for LLM prompting).
        """
        # Episodic retrieval
        episodes = self._retrieve_relevant_episodes(user_id, query, top_k=5)

        # Semantic retrieval
        facts = self.semantic.get_facts(user_id, top_k=10)

        # Procedural retrieval
        task_type = self._classify_task(query)
        procedures = self.procedural.retrieve(user_id, task_type, only_successful=True)

        # Assemble context
        context_parts: list[str] = []

        if facts:
            context_parts.append("KNOWN FACTS:")
            for f in facts[:5]:
                context_parts.append(f"  - {f['fact']}")

        if procedures:
            context_parts.append("SUCCESSFUL PROCEDURES:")
            for p in procedures[:3]:
                context_parts.append(f"  - [{p['task_type']}] steps: {' → '.join(p['steps'])}")

        if episodes:
            context_parts.append("RELEVANT EPISODES:")
            for e in episodes[:3]:
                context_parts.append(
                    f"  - Q: {e['query'][:80]}  |  A: {e['response'][:80]}  |  outcome: {e['outcome']}"
                )

        context_str = "\n".join(context_parts) if context_parts else "(no relevant memory found)"

        result = {
            "episodes": episodes,
            "facts": facts,
            "procedures": procedures,
            "context_str": context_str,
        }

        logger.debug(
            "Memory query for {uid}: {ne} episodes, {nf} facts, {np} procedures",
            uid=user_id,
            ne=len(episodes),
            nf=len(facts),
            np=len(procedures),
        )
        return result

    def _retrieve_relevant_episodes(
        self, user_id: str, query: str, top_k: int = 5,
    ) -> list[dict]:
        """Retrieve episodes most relevant to *query*."""
        all_episodes = self.episodic.retrieve(user_id, top_k=50)
        if not all_episodes:
            return []

        if self.embedder is not None:
            # Embedding-based similarity
            try:
                q_vec = self.embedder.encode(query)
                if isinstance(q_vec, np.ndarray):
                    scores = []
                    for ep in all_episodes:
                        ep_text = f"{ep['query']} {ep['response']}"
                        ep_vec = self.embedder.encode(ep_text)
                        if isinstance(ep_vec, np.ndarray):
                            sim = float(np.dot(q_vec, ep_vec) / (
                                np.linalg.norm(q_vec) * np.linalg.norm(ep_vec) + 1e-9
                            ))
                            scores.append((sim, ep))
                        else:
                            scores.append((0.0, ep))
                    scores.sort(key=lambda x: x[0], reverse=True)
                    return [ep for _, ep in scores[:top_k]]
            except Exception as exc:
                logger.warning("Embedding retrieval failed, falling back to recency: {e}", e=exc)

        # Fallback: return most recent
        return all_episodes[:top_k]

    # ------------------------------------------------------------------
    # Task classification (simple keyword heuristic)
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_task(query: str) -> str:
        """Assign a coarse task type label to a query."""
        q = query.lower()
        if any(w in q for w in ("code", "program", "debug", "function", "class", "implement")):
            return "code"
        if any(w in q for w in ("summarize", "summary", "overview", "explain")):
            return "summarize"
        if any(w in q for w in ("search", "find", "look", "retrieve", "where")):
            return "search"
        if any(w in q for w in ("write", "draft", "compose", "create")):
            return "writing"
        if any(w in q for w in ("analyse", "analyze", "compare", "evaluate")):
            return "analysis"
        return "general"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return aggregate statistics across all three memory tiers."""
        episodic_total = self.episodic.count()
        semantic_total = self.semantic.count()
        procedural_total = self.procedural.count()

        return {
            "episodic": {
                "total_episodes": episodic_total,
            },
            "semantic": {
                "total_facts": semantic_total,
            },
            "procedural": {
                "total_procedures": procedural_total,
            },
            "total_memories": episodic_total + semantic_total + procedural_total,
        }
