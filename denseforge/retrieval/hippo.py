"""HippoRAG — PageRank on knowledge graph for long-term memory."""
import networkx as nx


class HippoRAG:
    """Knowledge graph with PageRank for retrieval."""

    def __init__(self, embedder=None, llm_fn=None):
        self.graph = nx.Graph()
        self.embedder = embedder
        self.llm_fn = llm_fn
        self._doc_count = 0

    def index_document(self, text: str, doc_id: str):
        """Index document as graph node."""
        self.graph.add_node(doc_id, text=text[:1000], weight=1.0)
        self._doc_count += 1

        # Extract simple entity links (keyword-based)
        words = set(text.lower().split())
        for existing_node in list(self.graph.nodes):
            if existing_node == doc_id:
                continue
            existing_text = self.graph.nodes[existing_node].get("text", "")
            existing_words = set(existing_text.lower().split())
            overlap = len(words & existing_words)
            if overlap > 3:
                self.graph.add_edge(doc_id, existing_node, weight=overlap)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using PageRank + keyword matching."""
        if not self.graph.nodes:
            return []

        query_words = set(query.lower().split())
        scores = {}
        for node in self.graph.nodes:
            text = self.graph.nodes[node].get("text", "")
            node_words = set(text.lower().split())
            keyword_score = len(query_words & node_words) / max(len(query_words), 1)
            scores[node] = keyword_score

        # Boost with PageRank
        try:
            pagerank = nx.pagerank(self.graph, weight="weight")
            for node in scores:
                scores[node] *= (1 + pagerank.get(node, 0))
        except Exception:
            pass

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"doc_id": node, "text": self.graph.nodes[node].get("text", ""), "score": score}
            for node, score in ranked
        ]

    def stats(self) -> dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
