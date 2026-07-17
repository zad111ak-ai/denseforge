"""Session-Aware Pipeline — session context для предсказания и оптимизации."""
import time
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionState:
    session_id: str
    user_id: str
    started_at: float
    queries: List[Dict] = field(default_factory=list)
    retrieved_docs: List[int] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


class SessionAwarePipeline:
    """Follow-up detection, topic drift, predictive caching."""

    def __init__(self, max_session_hours: float = 2.0):
        self.max_duration = max_session_hours * 3600
        self.sessions: Dict[str, SessionState] = {}

    def get_or_create_session(self, session_id: str, user_id: str) -> SessionState:
        now = time.time()
        expired = [sid for sid, s in self.sessions.items() if now - s.started_at > self.max_duration]
        for sid in expired:
            del self.sessions[sid]
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id, user_id=user_id, started_at=now)
        return self.sessions[session_id]

    def enrich_query(self, query: str, session_id: str, user_id: str) -> Dict:
        session = self.get_or_create_session(session_id, user_id)
        is_followup = self._is_followup(query, session)
        enrichment = {
            "is_followup": is_followup,
            "session_position": len(session.queries),
            "recent_topics": session.topics[-3:],
            "carry_over_docs": session.retrieved_docs[-5:],
        }
        session.queries.append({"query": query, "timestamp": time.time()})
        return enrichment

    def record_retrieval(self, session_id: str, doc_ids: List[int]):
        if session_id in self.sessions:
            self.sessions[session_id].retrieved_docs.extend(doc_ids)
            self.sessions[session_id].retrieved_docs = self.sessions[session_id].retrieved_docs[-100:]

    def _is_followup(self, query: str, session: SessionState) -> bool:
        if not session.queries:
            return False
        followup_kw = ["а", "ещё", "подробнее", "а что", "почему", "уточни", "more", "elaborate"]
        q_lower = query.lower()
        if any(k in q_lower for k in followup_kw):
            return True
        last_len = len(session.queries[-1]["query"].split())
        if last_len > 10 and len(query.split()) < 5:
            return True
        return False

    def predictive_preload(self, session_id: str) -> List[str]:
        if session_id not in self.sessions or len(self.sessions[session_id].queries) < 2:
            return []
        last_q = self.sessions[session_id].queries[-1]["query"]
        return [f"more about {last_q}", f"examples of {last_q}", f"compare {last_q}"]

    def stats(self) -> dict:
        return {"active_sessions": len(self.sessions)}
