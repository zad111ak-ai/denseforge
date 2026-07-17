"""DenseForge — Main orchestrator that ties ingestion, retrieval, and generation together."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from denseforge.config import DenseForgeConfig
from denseforge.embeddings.adaptive import AdaptiveEmbedder
from denseforge.embeddings.cache import SemanticQueryCache
from denseforge.retrieval.triple_hybrid import TripleHybridStore
from denseforge.retrieval.raptor import RaptorTree
from denseforge.retrieval.hippo import HippoRAG
from denseforge.retrieval.cag import CAGEngine
from denseforge.ingestion.chunking import LateChunker
from denseforge.ingestion.augmentation import ContextualAugmenter
from denseforge.ingestion.streaming import StreamingRAG
from denseforge.retrieval.advanced import MetadataFilter, SentenceWindow, IncrementalManager


class DenseForge:
    """Autonomous Cognitive Knowledge Platform — single entry-point."""

    def __init__(self, config: Optional[DenseForgeConfig] = None):
        self.config = config or DenseForgeConfig()
        self.config.post_init()
        logger.info("DenseForge v1.0.0 initialising (device={})", self.config.embedding.device)

        # Embeddings
        self.embedder = AdaptiveEmbedder(
            model_name=self.config.embedding.model_name,
            device=self.config.embedding.device,
        )

        # Cache
        self.cache = SemanticQueryCache(
            similarity_threshold=self.config.cache.similarity_threshold,
            max_size=self.config.cache.max_size,
            default_ttl=self.config.cache.default_ttl_seconds,
            persist_path=self.config.cache.persist_path,
        )

        # Ingestion
        self.chunker = LateChunker(
            chunk_size=512, chunk_overlap=50,
        )
        self.augmenter = ContextualAugmenter(context_window=2)
        self.streamer = StreamingRAG()

        # Retrieval backends
        self.triple_store = TripleHybridStore(dim=self.config.embedding.default_dim, binary_dim=768)
        self.raptor = RaptorTree(
            max_levels=self.config.retrieval.raptor_max_levels,
            cluster_ratio=self.config.retrieval.raptor_cluster_ratio,
        )
        self.hippo = HippoRAG()
        self.cag = CAGEngine()
        
        # Advanced retrieval features (from competitor analysis)
        self._sentence_window = SentenceWindow(window_size=2)
        self._incremental = IncrementalManager(self)

        # Bookkeeping
        self._doc_counter = 0
        self._query_counter = 0
        self._total_ingest_time = 0.0
        self._total_query_time = 0.0

        # v13.0: Semantic dedup + columnar metadata
        from denseforge.ingestion.dedup import SemanticDeduplicator
        from denseforge.ingestion.columnar import ColumnarMetadata
        self.deduplicator = SemanticDeduplicator()
        self.columnar_meta = ColumnarMetadata()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, text: str, title: str = "", metadata: dict | None = None) -> list[int]:
        """Chunk, augment, embed, and index a document."""
        import hashlib
        t0 = time.perf_counter()
        meta = metadata or {}
        if title:
            meta["title"] = title

        # v13.0: Semantic deduplication
        if hasattr(self, 'deduplicator') and self.deduplicator is not None:
            content_hash = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]
            if content_hash in self.deduplicator._hashes:
                logger.info("Dedup: skipping exact duplicate (hash={})", content_hash[:8])
                return []
            # Add to dedup tracker
            self.deduplicator.add(f"doc_{self._doc_counter}", text)

        chunks = self.chunker.split(text, title=meta.get("title", ""))
        augmented = self.augmenter.augment(chunks)

        texts = [c["augmented"] for c in augmented]
        embed_results = self.embedder.encode_batch(texts)
        embeddings = np.array([e.vectors[self.config.embedding.default_dim] for e in embed_results])
        binary_vecs = np.array([e.binary for e in embed_results])

        ids = self.triple_store.add_batch(
            texts=[c["text"] for c in chunks],
            augmented_texts=texts,
            embeddings=embeddings,
            binary_vecs=binary_vecs,
            metadatas=[{**meta, "chunk_idx": c["chunk_idx"]} for c in chunks],
        )

        self.raptor.add_incremental_batch(texts, embeddings)

        for chunk_text in [c["text"] for c in chunks]:
            doc_id = f"doc_{self._doc_counter}"
            self.hippo.index_document(chunk_text, doc_id)
            self._doc_counter += 1

        # v13.0: Columnar metadata tracking
        import time as _time
        self.columnar_meta.add(
            doc_id=f"batch_{self._doc_counter}",
            timestamp=int(_time.time()),
            source=meta.get("source", ""),
            title_hash=hash(text[:64]) if text else 0,
            embedding_hash=hash(embeddings[0].tobytes()) if len(embeddings) > 0 else 0,
        )

        elapsed = time.perf_counter() - t0
        self._total_ingest_time += elapsed
        logger.info("Ingested doc → {} chunks in {:.3f}s", len(ids), elapsed)
        return ids

    def ingest_batch(self, documents: list[dict]) -> int:
        """Ingest multiple documents with batched encoding for speed.

        Each dict must have a 'text' key. Optional: 'title', 'metadata'.
        """
        import time as _time
        t0 = _time.perf_counter()

        all_chunks = []
        all_augmented = []
        all_metadatas = []

        for doc in documents:
            meta = doc.get("metadata") or {}
            if doc.get("title"):
                meta["title"] = doc["title"]

            # Dedup check
            if hasattr(self, 'deduplicator') and self.deduplicator is not None:
                import hashlib
                content_hash = hashlib.sha256(doc["text"].strip().lower().encode()).hexdigest()[:16]
                if content_hash in self.deduplicator._hashes:
                    continue
                self.deduplicator.add(f"doc_{self._doc_counter}", doc["text"])

            chunks = self.chunker.split(doc["text"], title=meta.get("title", ""))
            augmented = self.augmenter.augment(chunks)

            for c in augmented:
                all_chunks.append(c)
                all_augmented.append(c["augmented"])
                all_metadatas.append({**meta, "chunk_idx": c["chunk_idx"]})

        if not all_chunks:
            return 0

        # Single batch encode for ALL chunks — 5-10x faster than per-doc
        texts = [c["augmented"] for c in all_chunks]
        embed_results = self.embedder.encode_batch(texts)
        embeddings = np.array([e.vectors[self.config.embedding.default_dim] for e in embed_results])
        binary_vecs = np.array([e.binary for e in embed_results])

        ids = self.triple_store.add_batch(
            texts=[c["text"] for c in all_chunks],
            augmented_texts=texts,
            embeddings=embeddings,
            binary_vecs=binary_vecs,
            metadatas=all_metadatas,
        )

        self.raptor.add_incremental_batch(texts, embeddings)

        for chunk_text, meta in zip([c["text"] for c in all_chunks], all_metadatas):
            doc_id = f"doc_{self._doc_counter}"
            self.hippo.index_document(chunk_text, doc_id)
            self._doc_counter += 1

        # Columnar metadata
        self.columnar_meta.add(
            doc_id=f"batch_{self._doc_counter}",
            timestamp=int(_time.time()),
            source=documents[0].get("metadata", {}).get("source", ""),
            title_hash=hash(documents[0].get("text", "")[:64]),
            embedding_hash=hash(embeddings[0].tobytes()) if len(embeddings) > 0 else 0,
        )

        elapsed = _time.perf_counter() - t0
        self._total_ingest_time += elapsed
        logger.info("Batch ingested {} docs → {} chunks in {:.3f}s", len(documents), len(ids), elapsed)
        return len(ids)

    def ingest_stream(self, doc: dict, flush_threshold: int = 32):
        """Buffer a document and flush when threshold is reached."""
        self.streamer.add(doc)
        if len(self.streamer._buffer) >= flush_threshold:
            return self.ingest_batch(self.streamer.flush())
        return 0

    def flush_stream(self):
        """Force-flush the streaming buffer."""
        return self.ingest_batch(self.streamer.flush())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int | None = None,
              channels: list[str] | None = None,
              filters: dict | None = None,
              use_sentence_window: bool = False) -> dict:
        """End-to-end query: cache → retrieve → format results.
        
        Args:
            question: Search query
            top_k: Number of results to return
            channels: Which search channels to use (dense, sparse, binary)
            filters: Metadata filters (e.g., {"source": {"$eq": "github"}})
            use_sentence_window: Expand top result with surrounding chunks
        
        Returns:
            Query results with context, sources, and timing
        """
        t0 = time.perf_counter()
        top_k = top_k or self.config.retrieval.top_k
        
        # Build metadata filter
        meta_filter = MetadataFilter(filters) if filters else None

        # Check cache first
        q_emb_result = self.embedder.encode(question, task="retrieval")
        q_vec = q_emb_result.vectors[self.config.embedding.default_dim]
        cached = self.cache.get(question, q_vec)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        # Triple hybrid retrieval — 512-dim for dense, 768 for binary
        q_full = q_emb_result.vectors.get(768, q_vec)
        results = self.triple_store.search(
            query=question, query_embedding=q_vec, query_full=q_full, top_k=top_k,
            channels=channels,
        )
        
        # Apply metadata filters if provided (from Qdrant/Chroma)
        if meta_filter:
            results = [
                r for r in results
                if meta_filter.match(r.get("metadata", {}))
            ]
        
        # Sentence window expansion (from LlamaIndex)
        if use_sentence_window and results:
            # Find top match index in triple store
            top_text = results[0]["text"]
            for i, chunk in enumerate(self.triple_store.texts):
                if chunk == top_text:
                    expanded = self._sentence_window.expand(
                        [{"text": t, "metadata": m or {}} 
                         for t, m in zip(self.triple_store.texts, self.triple_store.metadata)],
                        [i],
                        max_chunks=top_k * 2,
                    )
                    # Use expanded results
                    results = expanded[:top_k]
                    break

        # Augment with RAPTOR for hierarchical context
        raptor_results = self.raptor.search(q_vec, top_k=min(3, top_k))
        raptor_texts = [r["text"] for r in raptor_results]

        # Augment with HippoRAG for graph context
        hippo_results = self.hippo.search(question, top_k=min(3, top_k))
        hippo_texts = [r["text"] for r in hippo_results]

        # Merge diverse context
        seen = set()
        context_chunks = []
        for r in results:
            if r["text"] not in seen:
                seen.add(r["text"])
                context_chunks.append(r["text"])
        for t in raptor_texts + hippo_texts:
            if t not in seen:
                seen.add(t)
                context_chunks.append(t)

        answer_context = "\n\n---\n\n".join(context_chunks[:5])

        elapsed = time.perf_counter() - t0
        self._total_query_time += elapsed
        self._query_counter += 1

        response = {
            "query": question,
            "context": answer_context,
            "sources": results[:top_k],
            "raptor_context": raptor_texts,
            "hippo_context": hippo_texts,
            "cache_hit": False,
            "elapsed_ms": round(elapsed * 1000, 1),
        }

        # Store in cache
        self.cache.put(question, q_vec, response)
        return response

    # ------------------------------------------------------------------
    # Incremental updates (from Chroma)
    # ------------------------------------------------------------------
    def upsert(self, doc_id: str, text: str, metadata: dict | None = None) -> dict:
        """Add or update a document incrementally.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
            metadata: Optional metadata
            
        Returns:
            {"action": "added"|"updated", "doc_id": str, "version": int}
        """
        return self._incremental.upsert(doc_id, text, metadata)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID.
        
        Returns:
            True if document was found and marked for deletion
        """
        return self._incremental.delete(doc_id)
    
    def list_documents(self) -> list[str]:
        """List all tracked document IDs."""
        return self._incremental.list_documents()
    
    # ------------------------------------------------------------------
    # Stats & persistence
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return comprehensive stats for all subsystems."""
        try:
            return {
                "version": "1.0.0",
                "ingested_documents": self._doc_counter,
                "total_queries": self._query_counter,
                "total_ingest_time_s": round(self._total_ingest_time, 3),
                "total_query_time_s": round(self._total_query_time, 3),
                "embedder": {
                    "model": self.embedder.model_name,
                    "full_dim": self.embedder.full_dim,
                },
                "triple_store": self.triple_store.stats(),
                "raptor": self.raptor.stats(),
                "hippo": self.hippo.stats(),
                "cag": self.cag.stats(),
                "cache": self.cache.stats(),
                "streamer": self.streamer.stats(),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def search(self, question: str, top_k: int | None = None,
               channels: list[str] | None = None) -> dict:
        """Alias for query() — standard search API."""
        return self.query(question, top_k=top_k, channels=channels)

    def ask_why(self, effect: str, max_depth: int = 5) -> dict:
        """Causal reasoning: 'Why does X happen?'"""
        return {"effect": effect, "answer": f"Causal analysis of '{effect}' — requires LLM.", "chains": []}

    def ask_what_if(self, intervention: str, target: str) -> dict:
        """Counterfactual: 'What if X?'"""
        return {"intervention": intervention, "target": target, "answer": "Counterfactual requires LLM.", "predicted_effect": None}

    def plan_and_execute(self, task: str, user_id: str | None = None) -> dict:
        """Long-horizon task execution via multi-agent."""
        return {"task": task, "plan": [], "status": "planning", "note": "Requires LLM integration"}

    def start_mcp_server(self, host: str = "localhost", port: int = 8080):
        """Start MCP server for external agents."""
        from denseforge.protocols.mcp_server import MCPServer
        mcp = MCPServer(denseforge_instance=self)
        try:
            import uvicorn
            from fastapi import FastAPI
            app = FastAPI(title="DenseForge MCP Server")

            @app.post("/mcp")
            async def mcp_endpoint(request: dict):
                return await mcp.handle_request(request)

            @app.get("/health")
            async def health():
                return {"status": "ok", "version": "1.0.0"}

            uvicorn.run(app, host=host, port=port)
        except ImportError:
            print("pip install fastapi uvicorn to use MCP server")

    def save_cache(self):
        """Persist the semantic cache to disk."""
        self.cache.save()

    def load_cache(self):
        """Load the semantic cache from disk."""
        self.cache.load()

    # ------------------------------------------------------------------
    # Full persistence
    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> dict:
        """Save entire state (FAISS + docs + metadata) to disk.

        Args:
            path: Directory path. Defaults to config.storage.persist_dir.
        """
        from denseforge.persistence import save_forge
        target = path or getattr(self.config, "storage", None)
        if target is None:
            target = str(Path.home() / ".denseforge" / "data")
        elif hasattr(target, "persist_dir"):
            target = target.persist_dir
        return save_forge(self, target)

    def load(self, path: str | None = None) -> dict:
        """Load state from disk.

        Args:
            path: Directory path. Defaults to config.storage.persist_dir.
        """
        from denseforge.persistence import load_forge
        target = path or getattr(self.config, "storage", None)
        if target is None:
            target = str(Path.home() / ".denseforge" / "data")
        elif hasattr(target, "persist_dir"):
            target = target.persist_dir
        return load_forge(self, target)
