"""Triple Hybrid Store: BM25 + Dense HNSW + Binary Pre-filter."""
import re
import numpy as np
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class StoredDocument:
    doc_id: int
    text: str
    augmented_text: str
    metadata: dict = field(default_factory=dict)


class TripleHybridStore:
    """BM25 + FAISS Dense + Binary index with RRF fusion."""

    def __init__(self, dim: int = 512, binary_dim: int = 96):
        self.dim = dim
        self.documents: list[StoredDocument] = []

        # Dense index (HNSW)
        try:
            import faiss
            self.dense_index = faiss.IndexHNSWFlat(dim, 32)
            self.dense_index.hnsw.efConstruction = 40
            self.dense_index.hnsw.efSearch = 64
            self.binary_index = faiss.IndexBinaryFlat(binary_dim)
            self._has_faiss = True
        except ImportError:
            self.dense_index = None
            self.binary_index = None
            self._has_faiss = False

        self._full_vectors: list[np.ndarray] = []
        self._bm25 = None
        self._bm25_corpus: list[list[str]] = []
        self.fusion_weights = {"bm25": 0.30, "dense": 0.50, "binary": 0.20}

    def add(self, text: str, augmented_text: str, embedding: np.ndarray,
            binary_vec: np.ndarray, metadata: dict | None = None) -> int:
        doc_id = len(self.documents)
        self.documents.append(StoredDocument(doc_id, text, augmented_text, metadata or {}))
        emb = embedding.astype(np.float32).reshape(1, -1)
        self._full_vectors.append(embedding.astype(np.float32))

        if self._has_faiss:
            self.dense_index.add(emb)
            self.binary_index.add(binary_vec.reshape(1, -1))

        self._bm25_corpus.append(self._tokenize(augmented_text))
        self._rebuild_bm25()
        return doc_id

    def add_batch(self, texts, augmented_texts, embeddings, binary_vecs, metadatas=None):
        metadatas = metadatas or [{} for _ in texts]
        start_id = len(self.documents)
        for i, (text, aug, meta) in enumerate(zip(texts, augmented_texts, metadatas)):
            self.documents.append(StoredDocument(start_id + i, text, aug, meta))
            self._full_vectors.append(embeddings[i].astype(np.float32))
            self._bm25_corpus.append(self._tokenize(aug))
        if self._has_faiss:
            self.dense_index.add(embeddings.astype(np.float32))
            # Embedder already produces packed binary — pass directly
            self.binary_index.add(binary_vecs)
        self._rebuild_bm25()
        return list(range(start_id, start_id + len(texts)))

    def search(self, query: str, query_embedding: np.ndarray,
               query_full: Optional[np.ndarray] = None,
               top_k: int = 10, channels: Optional[list[str]] = None) -> list[dict]:
        if not self.documents:
            return []
        channels = channels or ["bm25", "dense", "binary"]
        n = len(self.documents)
        candidate_k = min(top_k * 3, n)
        channel_scores: dict[str, dict[int, float]] = {}

        if "bm25" in channels and self._bm25:
            query_tokens = self._tokenize(query)
            bm25_scores = self._bm25.get_scores(query_tokens)
            channel_scores["bm25"] = {i: float(s) for i, s in enumerate(bm25_scores) if s > 0}

        if "dense" in channels and self._has_faiss:
            q = query_embedding.astype(np.float32).reshape(1, -1)
            D, I = self.dense_index.search(q, candidate_k)
            channel_scores["dense"] = {
                int(I[0][i]): float(D[0][i]) for i in range(len(I[0])) if I[0][i] != -1
            }

        if "binary" in channels and self._has_faiss and n >= 3:
            # Use query_full (768-dim) for binary search if available
            q_for_binary = query_full if query_full is not None else query_embedding
            # Pack: threshold → packbits (same as embedder)
            q_bin = (q_for_binary[:768] > 0).astype(np.uint8) if len(q_for_binary) >= 768 else (q_for_binary > 0).astype(np.uint8)
            q_bin_packed = np.packbits(q_bin).reshape(1, -1)
            D_bin, I_bin = self.binary_index.search(q_bin_packed, candidate_k * 2)
            candidate_ids = [int(i) for i in I_bin[0] if i != -1 and i < n]
            if candidate_ids:
                rescore = {}
                q_full_vec = query_embedding.astype(np.float32)
                q_norm = q_full_vec / max(np.linalg.norm(q_full_vec), 1e-8)
                for cid in candidate_ids:
                    v = self._full_vectors[cid]
                    v_norm = v / max(np.linalg.norm(v), 1e-8)
                    rescore[cid] = float(np.dot(q_norm, v_norm))
                channel_scores["binary"] = rescore

        fused = self._fuse_scores(channel_scores, channels)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"doc_id": doc_id, "text": self.documents[doc_id].text,
             "score": score, "metadata": self.documents[doc_id].metadata}
            for doc_id, score in ranked if 0 <= doc_id < n
        ]

    def _fuse_scores(self, channel_scores, channels):
        """Reciprocal Rank Fusion (RRF) — combines ranks, not raw scores.

        RRF is robust to score magnitude differences between channels.
        Score = sum(1 / (k + rank)) for each channel where document appears.
        Default k=60 is standard from the original RRF paper.
        """
        fused = defaultdict(float)
        k = 60  # RRF constant (original paper: k=60)
        for ch in channels:
            scores = channel_scores.get(ch, {})
            if not scores:
                continue
            # Sort by score descending, assign ranks
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_docs):
                fused[doc_id] += 1.0 / (k + rank + 1)
        return dict(fused)

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in re.findall(r"\w+", text.lower()) if len(t) > 2]

    def _pack_binary(self, binary_vecs: np.ndarray) -> np.ndarray:
        """Pack uint8 bits (0/1) → FAISS binary format (packed bytes)."""
        return np.packbits(binary_vecs.astype(np.uint8)).reshape(binary_vecs.shape[0], -1)

    def _rebuild_bm25(self):
        if not self._bm25_corpus:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._bm25_corpus)
        except ImportError:
            self._bm25 = None

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "dense_index_size": self.dense_index.ntotal if self._has_faiss else 0,
            "binary_index_size": self.binary_index.ntotal if self._has_faiss else 0,
        }
