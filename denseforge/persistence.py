"""Persistence: save/load DenseForge state to disk."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from loguru import logger


def save_forge(forge, path: str | Path) -> dict:
    """Save entire DenseForge state to directory.

    Creates:
      - meta.json           — counters, config
      - triple_dense.faiss  — dense FAISS index
      - triple_binary.faiss — binary FAISS index
      - triple_docs.json    — documents list
      - triple_vectors.npy  — dense embeddings
      - bm25_corpus.json    — BM25 tokenized corpus
      - raptor.json         — RAPTOR tree levels
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Metadata
    meta = {
        "version": "12.0",
        "doc_counter": forge._doc_counter,
        "query_counter": forge._query_counter,
        "total_ingest_time": forge._total_ingest_time,
        "total_query_time": forge._total_query_time,
        "saved_at": time.time(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))

    # 2. Triple store — FAISS indices
    import faiss
    if forge.triple_store._has_faiss:
        faiss.write_index(forge.triple_store.dense_index, str(path / "triple_dense.faiss"))
        faiss.write_index_binary(forge.triple_store.binary_index, str(path / "triple_binary.faiss"))

    # 3. Triple store — documents
    docs = []
    for doc in forge.triple_store.documents:
        docs.append({
            "doc_id": doc.doc_id,
            "text": doc.text,
            "augmented_text": doc.augmented_text,
            "metadata": doc.metadata,
        })
    (path / "triple_docs.json").write_text(json.dumps(docs, ensure_ascii=False))

    # 4. Full vectors
    if forge.triple_store._full_vectors:
        vecs = np.array(forge.triple_store._full_vectors)
        np.save(str(path / "triple_vectors.npy"), vecs)

    # 5. BM25 corpus
    (path / "bm25_corpus.json").write_text(
        json.dumps(forge.triple_store._bm25_corpus)
    )

    # 6. RAPTOR tree — save levels (each node has text, embedding, doc_id, level, children)
    raptor_data = []
    for level_nodes in forge.raptor.levels:
        for node in level_nodes:
            entry = {
                "text": node["text"],
                "embedding": node["embedding"].tolist(),
                "doc_id": node["doc_id"],
                "level": node["level"],
            }
            if "children" in node:
                entry["children"] = node["children"]
            raptor_data.append(entry)
    (path / "raptor.json").write_text(json.dumps(raptor_data, ensure_ascii=False))

    # 7. Dedup state (v13.0)
    if hasattr(forge, 'deduplicator') and forge.deduplicator is not None:
        (path / "dedup.json").write_text(
            json.dumps(forge.deduplicator.save_state())
        )

    # 8. Columnar metadata (v13.0)
    if hasattr(forge, 'columnar_meta') and forge.columnar_meta is not None:
        forge.columnar_meta.save(str(path / "columnar.json"))

    elapsed = time.perf_counter() - t0
    n_docs = len(forge.triple_store.documents)
    logger.info("Saved DenseForge: {} docs → {} ({:.2f}s)", n_docs, path, elapsed)
    return {"path": str(path), "documents": n_docs, "elapsed_ms": round(elapsed * 1000, 1)}


def load_forge(forge, path: str | Path) -> dict:
    """Load DenseForge state from directory."""
    path = Path(path)
    if not path.exists():
        logger.warning("No DenseForge state at {}", path)
        return {"loaded": 0}

    t0 = time.perf_counter()

    # 1. Metadata
    meta_file = path / "meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
        forge._doc_counter = meta.get("doc_counter", 0)
        forge._query_counter = meta.get("query_counter", 0)
        forge._total_ingest_time = meta.get("total_ingest_time", 0)
        forge._total_query_time = meta.get("total_query_time", 0)

    # 2. FAISS indices
    import faiss
    dense_f = path / "triple_dense.faiss"
    binary_f = path / "triple_binary.faiss"
    if dense_f.exists() and forge.triple_store._has_faiss:
        forge.triple_store.dense_index = faiss.read_index(str(dense_f))
    if binary_f.exists() and forge.triple_store._has_faiss:
        forge.triple_store.binary_index = faiss.read_index_binary(str(binary_f))

    # 3. Documents
    docs_file = path / "triple_docs.json"
    if docs_file.exists():
        from denseforge.retrieval.triple_hybrid import StoredDocument
        docs = json.loads(docs_file.read_text())
        forge.triple_store.documents = [
            StoredDocument(d["doc_id"], d["text"], d["augmented_text"], d.get("metadata", {}))
            for d in docs
        ]

    # 4. Vectors
    vecs_file = path / "triple_vectors.npy"
    if vecs_file.exists():
        vecs = np.load(str(vecs_file))
        forge.triple_store._full_vectors = [v for v in vecs]

    # 5. BM25 corpus
    bm25_file = path / "bm25_corpus.json"
    if bm25_file.exists():
        forge.triple_store._bm25_corpus = json.loads(bm25_file.read_text())
        forge.triple_store._rebuild_bm25()

    # 6. RAPTOR tree
    raptor_file = path / "raptor.json"
    if raptor_file.exists():
        data = json.loads(raptor_file.read_text())
        # Reset levels
        forge.raptor.levels = [[] for _ in range(forge.raptor.max_levels)]
        forge.raptor._doc_count = 0
        for entry in data:
            level = entry["level"]
            if level < len(forge.raptor.levels):
                node = {
                    "text": entry["text"],
                    "embedding": np.array(entry["embedding"]),
                    "doc_id": entry["doc_id"],
                    "level": level,
                }
                if "children" in entry:
                    node["children"] = entry["children"]
                forge.raptor.levels[level].append(node)
                forge.raptor._doc_count = max(forge.raptor._doc_count, entry["doc_id"] + 1)

    # 7. Dedup state (v13.0)
    dedup_file = path / "dedup.json"
    if dedup_file.exists():
        from denseforge.ingestion.dedup import SemanticDeduplicator
        if not hasattr(forge, 'deduplicator') or forge.deduplicator is None:
            forge.deduplicator = SemanticDeduplicator()
        forge.deduplicator.load_state(json.loads(dedup_file.read_text()))

    # 8. Columnar metadata (v13.0)
    col_file = path / "columnar.json"
    if col_file.exists():
        from denseforge.ingestion.columnar import ColumnarMetadata
        if not hasattr(forge, 'columnar_meta') or forge.columnar_meta is None:
            forge.columnar_meta = ColumnarMetadata()
        forge.columnar_meta.load(str(col_file))

    elapsed = time.perf_counter() - t0
    n_docs = len(forge.triple_store.documents)
    logger.info("Loaded DenseForge: {} docs ← {} ({:.2f}s)", n_docs, path, elapsed)
    return {"loaded": n_docs, "elapsed_ms": round(elapsed * 1000, 1)}
