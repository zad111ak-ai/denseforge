"""Contextual Augmentation — enrich chunks with context."""


class ContextualAugmenter:
    """Add surrounding context to chunks for better retrieval."""

    def __init__(self, context_window: int = 2):
        self.context_window = context_window

    def augment(self, chunks: list[dict], all_chunks: list[dict] | None = None) -> list[dict]:
        if not all_chunks:
            all_chunks = chunks
        enriched = []
        for i, chunk in enumerate(chunks):
            start = max(0, i - self.context_window)
            end = min(len(all_chunks), i + self.context_window + 1)
            context_parts = [all_chunks[j]["text"] for j in range(start, end) if j != i]
            augmented = " ".join(context_parts) + " " + chunk["text"] if context_parts else chunk["text"]
            enriched.append({**chunk, "augmented": augmented.strip()})
        return enriched
