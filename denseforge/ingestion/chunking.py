"""Late Chunking — contextualize chunks after encoding."""
from typing import List


class LateChunker:
    """Split text into chunks with overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, title: str = "") -> List[dict]:
        words = text.split()
        if len(words) <= self.chunk_size:
            return [{"text": text, "title": title, "chunk_idx": 0}]

        chunks = []
        start = 0
        idx = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"text": chunk_text, "title": title, "chunk_idx": idx})
            start += self.chunk_size - self.chunk_overlap
            idx += 1
        return chunks
