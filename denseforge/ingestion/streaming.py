"""Streaming RAG — incremental document ingestion."""


class StreamingRAG:
    """Stream documents into the knowledge base incrementally."""

    def __init__(self):
        self._buffer: list[dict] = []
        self._count = 0

    def add(self, doc: dict):
        self._buffer.append(doc)
        self._count += 1

    def flush(self) -> list[dict]:
        batch = self._buffer.copy()
        self._buffer.clear()
        return batch

    def stats(self) -> dict:
        return {"buffered": len(self._buffer), "total": self._count}
