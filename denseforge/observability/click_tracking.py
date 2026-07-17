"""Click-Through Tracking — user feedback loop for relevance.

Tracks which results users actually use → improves future retrieval.
"""
import time
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger


@dataclass
class ClickEvent:
    """Single click event."""
    query: str
    doc_id: int
    timestamp: float
    dwell_time: float = 0.0  # seconds spent viewing
    position: int = 0  # position in results
    session_id: Optional[str] = None


@dataclass
class QueryFeedback:
    """Aggregated feedback for a query."""
    query: str
    total_clicks: int = 0
    unique_docs: int = 0
    avg_dwell_time: float = 0.0
    click_positions: list[int] = field(default_factory=list)
    last_clicked_doc: Optional[int] = None


class ClickTracker:
    """Track user clicks for relevance feedback.
    
    Features:
    - Click-through rate (CTR)
    - Dwell time analysis
    - Position bias detection
    - Session tracking
    """
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._events: list[ClickEvent] = []
        self._query_feedback: dict[str, QueryFeedback] = {}
        self._session_clicks: dict[str, list[ClickEvent]] = defaultdict(list)
        self._stats = {"total_clicks": 0, "total_queries": 0}
    
    def record_click(
        self,
        query: str,
        doc_id: int,
        position: int = 0,
        dwell_time: float = 0.0,
        session_id: Optional[str] = None,
    ):
        """Record a user click."""
        event = ClickEvent(
            query=query,
            doc_id=doc_id,
            timestamp=time.time(),
            dwell_time=dwell_time,
            position=position,
            session_id=session_id,
        )
        
        self._events.append(event)
        self._stats["total_clicks"] += 1
        
        # Update query feedback
        if query not in self._query_feedback:
            self._query_feedback[query] = QueryFeedback(query=query)
            self._stats["total_queries"] += 1
        
        feedback = self._query_feedback[query]
        feedback.total_clicks += 1
        feedback.click_positions.append(position)
        feedback.last_clicked_doc = doc_id
        
        # Update dwell time average
        n = feedback.total_clicks
        feedback.avg_dwell_time = (
            (feedback.avg_dwell_time * (n - 1) + dwell_time) / n
        )
        
        # Track unique docs
        clicked_docs = set(e.doc_id for e in self._events if e.query == query)
        feedback.unique_docs = len(clicked_docs)
        
        # Session tracking
        if session_id:
            self._session_clicks[session_id].append(event)
    
    def get_relevance_scores(self, query: str) -> dict[int, float]:
        """Get relevance scores for documents based on clicks.
        
        Returns:
            Dict of doc_id → relevance score (0-1)
        """
        scores = defaultdict(float)
        query_events = [e for e in self._events if e.query == query]
        
        if not query_events:
            return {}
        
        # Normalize by max clicks
        max_clicks = max(1, len(query_events))
        
        for event in query_events:
            # Click score (normalized)
            click_score = 1.0 / max_clicks
            
            # Dwell time bonus (more time = more relevant)
            dwell_bonus = min(1.0, event.dwell_time / 30.0)  # 30s max
            
            # Position penalty (higher position = less relevant)
            position_penalty = 1.0 / (1.0 + event.position * 0.1)
            
            # Combined score
            scores[event.doc_id] += (click_score + dwell_bonus * 0.3) * position_penalty
        
        # Normalize to 0-1
        max_score = max(scores.values()) if scores else 1.0
        return {doc_id: score / max_score for doc_id, score in scores.items()}
    
    def get_popular_queries(self, top_k: int = 10) -> list[dict]:
        """Get most popular queries."""
        sorted_queries = sorted(
            self._query_feedback.values(),
            key=lambda x: x.total_clicks,
            reverse=True,
        )[:top_k]
        
        return [
            {
                "query": q.query,
                "clicks": q.total_clicks,
                "unique_docs": q.unique_docs,
                "avg_dwell": q.avg_dwell_time,
            }
            for q in sorted_queries
        ]
    
    def get_session_summary(self, session_id: str) -> dict:
        """Get summary for a session."""
        clicks = self._session_clicks.get(session_id, [])
        if not clicks:
            return {"session_id": session_id, "clicks": 0}
        
        return {
            "session_id": session_id,
            "clicks": len(clicks),
            "queries": list(set(e.query for e in clicks)),
            "total_dwell": sum(e.dwell_time for e in clicks),
            "avg_dwell": sum(e.dwell_time for e in clicks) / len(clicks),
        }
    
    def save(self):
        """Save click data to disk."""
        if not self.persist_path:
            return
        
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "events": [
                    {
                        "query": e.query,
                        "doc_id": e.doc_id,
                        "timestamp": e.timestamp,
                        "dwell_time": e.dwell_time,
                        "position": e.position,
                        "session_id": e.session_id,
                    }
                    for e in self._events
                ],
                "stats": self._stats,
            }
            
            # Atomic write
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            Path(tmp_path).rename(self.persist_path)
            
            logger.debug(f"Click data saved: {len(self._events)} events")
        except Exception as e:
            logger.error(f"Failed to save click data: {e}")
    
    def load(self):
        """Load click data from disk."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            self._events = [
                ClickEvent(
                    query=e["query"],
                    doc_id=e["doc_id"],
                    timestamp=e["timestamp"],
                    dwell_time=e.get("dwell_time", 0),
                    position=e.get("position", 0),
                    session_id=e.get("session_id"),
                )
                for e in data.get("events", [])
            ]
            self._stats = data.get("stats", {"total_clicks": 0, "total_queries": 0})
            
            # Rebuild query feedback
            for event in self._events:
                if event.query not in self._query_feedback:
                    self._query_feedback[event.query] = QueryFeedback(query=event.query)
                fb = self._query_feedback[event.query]
                fb.total_clicks += 1
                fb.click_positions.append(event.position)
                fb.last_clicked_doc = event.doc_id
            
            logger.debug(f"Click data loaded: {len(self._events)} events")
        except Exception as e:
            logger.error(f"Failed to load click data: {e}")
    
    def stats(self) -> dict:
        """Get tracker statistics."""
        return {
            **self._stats,
            "total_events": len(self._events),
            "unique_queries": len(self._query_feedback),
            "unique_sessions": len(self._session_clicks),
        }
