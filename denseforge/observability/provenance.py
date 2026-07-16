"""Retrieval Provenance — track the full chain from query to answer."""
from typing import Optional
from loguru import logger


class RetrievalProvenanceGraph:
    """Record and query the provenance of every answer."""

    def __init__(self):
        self._queries: list[dict] = []
        self._doc_index: dict[str, list[dict]] = {}

    def record_query(self, query: str, retrieved_docs: list[dict],
                     generated_answer: str, metadata: Optional[dict] = None):
        """Record full provenance chain for a query."""
        entry = {
            "query": query, "retrieved_docs": [
                {"doc_id": d.get("doc_id"), "score": d.get("score"), "text": d.get("text", "")[:200]}
                for d in retrieved_docs
            ],
            "answer": generated_answer[:500], "metadata": metadata or {},
        }
        self._queries.append(entry)

        for doc in retrieved_docs:
            doc_id = str(doc.get("doc_id", ""))
            if doc_id not in self._doc_index:
                self._doc_index[doc_id] = []
            self._doc_index[doc_id].append(len(self._queries) - 1)

    def get_provenance_for_doc(self, doc_id: str) -> dict:
        """Get all queries that used this document."""
        query_indices = self._doc_index.get(str(doc_id), [])
        return {
            "doc_id": doc_id,
            "used_in_queries": [self._queries[i] for i in query_indices],
            "total_uses": len(query_indices),
        }

    def get_query_history(self, limit: int = 10) -> list[dict]:
        return self._queries[-limit:]

    def stats(self) -> dict:
        return {"total_queries": len(self._queries), "indexed_docs": len(self._doc_index)}
