"""CAG — Cache-Augmented Generation for small knowledge bases."""


class CAGEngine:
    """Pre-load knowledge into LLM context window."""

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn
        self._preloaded: list[str] = []
        self._loaded = False

    def preload_knowledge(self, texts: list[str]):
        """Pre-load documents into context."""
        self._preloaded = texts
        self._loaded = True

    def generate(self, query: str, max_context_tokens: int = 8000) -> str:
        """Generate with pre-loaded knowledge."""
        if not self._loaded or not self.llm_fn:
            return ""
        context = "\n\n".join(self._preloaded)[:max_context_tokens * 4]
        prompt = f"Based on this knowledge:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.llm_fn(prompt)

    def stats(self) -> dict:
        return {"preloaded_docs": len(self._preloaded), "loaded": self._loaded}
