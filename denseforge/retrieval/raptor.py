"""RAPTOR Tree — hierarchical clustering for multi-hop retrieval."""
import numpy as np
from typing import Optional
from loguru import logger


class RaptorTree:
    """Recursive Abstractive Processing for Tree-Organized Retrieval."""

    def __init__(self, max_levels: int = 3, cluster_ratio: float = 0.3):
        self.max_levels = max_levels
        self.cluster_ratio = cluster_ratio
        self.levels: list[list[dict]] = [[] for _ in range(max_levels)]
        self._doc_count = 0

    def add_incremental_batch(self, texts: list[str], embeddings: np.ndarray):
        """Add chunks to level 0, optionally build tree."""
        for i, text in enumerate(texts):
            self.levels[0].append({
                "text": text, "embedding": embeddings[i],
                "doc_id": self._doc_count, "level": 0,
            })
            self._doc_count += 1

        # Build higher levels if enough nodes
        if len(self.levels[0]) >= 10:
            self._build_tree()

    def _build_tree(self):
        """Cluster level N → create level N+1 centroids."""
        for level in range(self.max_levels - 1):
            nodes = self.levels[level]
            if len(nodes) < 4:
                break
            n_clusters = max(2, int(len(nodes) * self.cluster_ratio))
            embeddings = np.array([n["embedding"] for n in nodes])

            try:
                from scipy.cluster.hierarchy import fcluster, linkage
                Z = linkage(embeddings, method="ward", metric="euclidean")
                labels = fcluster(Z, t=n_clusters, criterion="maxclust")

                for cluster_id in range(1, n_clusters + 1):
                    mask = labels == cluster_id
                    cluster_nodes = [nodes[i] for i in range(len(nodes)) if mask[i]]
                    if not cluster_nodes:
                        continue
                    centroid = embeddings[mask].mean(axis=0)
                    combined_text = " ".join(n["text"][:200] for n in cluster_nodes[:5])
                    self.levels[level + 1].append({
                        "text": combined_text, "embedding": centroid,
                        "doc_id": self._doc_count, "level": level + 1,
                        "children": [n["doc_id"] for n in cluster_nodes],
                    })
                    self._doc_count += 1
            except ImportError:
                break

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search across all levels, return best matches."""
        results = []
        for level_nodes in self.levels:
            for node in level_nodes:
                sim = self._cosine_sim(query_embedding, node["embedding"])
                results.append({**node, "score": sim})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def stats(self) -> dict:
        return {
            "total_nodes": sum(len(lv) for lv in self.levels),
            "levels": [len(lv) for lv in self.levels],
        }
