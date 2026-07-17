"""DenseForge — Advanced retrieval features inspired by competitors.

Features added from competitor analysis:
- Metadata filtering (Qdrant/Chroma style)
- Sentence window retrieval (LlamaIndex style)
- Incremental document updates (Chroma style)
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from loguru import logger


class MetadataFilter:
    """Filter search results by metadata fields.
    
    Supports operators: eq, ne, gt, lt, gte, lte, in, contains.
    
    Example:
        filter = MetadataFilter({
            "source": {"$eq": "github"},
            "date": {"$gte": "2024-01-01"},
            "tags": {"$contains": "python"}
        })
    """
    
    def __init__(self, filters: dict[str, Any]):
        self.filters = filters
    
    def match(self, metadata: dict[str, Any]) -> bool:
        """Check if metadata matches all filter conditions."""
        for field, condition in self.filters.items():
            value = metadata.get(field)
            if value is None:
                return False
            
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if not self._apply_op(value, op, target):
                        return False
            else:
                # Simple equality
                if value != condition:
                    return False
        return True
    
    def _apply_op(self, value: Any, op: str, target: Any) -> bool:
        """Apply a single operator."""
        try:
            if op == "$eq":
                return value == target
            elif op == "$ne":
                return value != target
            elif op == "$gt":
                return value > target
            elif op == "$lt":
                return value < target
            elif op == "$gte":
                return value >= target
            elif op == "$lte":
                return value <= target
            elif op == "$in":
                return value in target
            elif op == "$contains":
                if isinstance(value, str):
                    return target in value
                elif isinstance(value, list):
                    return target in value
                return False
            else:
                logger.warning(f"Unknown filter operator: {op}")
                return True
        except TypeError:
            return False


class SentenceWindow:
    """Sentence window retrieval — expand context around retrieved chunks.
    
    When a chunk matches, return surrounding chunks for better context.
    Inspired by LlamaIndex's SentenceWindowRetrieval.
    
    Example:
        window = SentenceWindow(window_size=2)
        expanded = window.expand(chunks, matched_indices=[3])
    """
    
    def __init__(self, window_size: int = 2):
        """Initialize with window size (chunks on each side)."""
        self.window_size = window_size
    
    def expand(
        self,
        chunks: list[dict[str, Any]],
        matched_indices: list[int],
        max_chunks: int = 10,
    ) -> list[dict[str, Any]]:
        """Expand matched chunks to include neighbors.
        
        Args:
            chunks: All chunks in order (must be sorted by chunk_idx)
            matched_indices: Indices of matched chunks
            max_chunks: Maximum chunks to return
            
        Returns:
            Expanded list of chunks with context
        """
        if not chunks or not matched_indices:
            return []
        
        # Sort chunks by chunk_idx if present
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_idx", 0))
        
        # Build index map
        chunk_map = {i: c for i, c in enumerate(sorted_chunks)}
        
        # Expand each match
        expanded_indices = set()
        for idx in matched_indices:
            # Add window
            start = max(0, idx - self.window_size)
            end = min(len(sorted_chunks) - 1, idx + self.window_size)
            for i in range(start, end + 1):
                expanded_indices.add(i)
        
        # Build result in order
        result = []
        for i in sorted(expanded_indices):
            if i in chunk_map:
                chunk = chunk_map[i].copy()
                # Mark original matches
                chunk["_is_match"] = i in matched_indices
                chunk["_window_position"] = i - matched_indices[0] if matched_indices else 0
                result.append(chunk)
        
        # Limit total chunks
        return result[:max_chunks]
    
    def get_center_chunk(
        self,
        chunks: list[dict[str, Any]],
        center_idx: int,
        window_size: int | None = None,
    ) -> dict[str, Any]:
        """Get a single chunk with expanded context as a dict.
        
        Returns:
            {"center": ..., "before": [...], "after": [...]}
        """
        ws = window_size or self.window_size
        sorted_chunks = sorted(chunks, key=lambda c: c.get("chunk_idx", 0))
        
        before = [
            sorted_chunks[i]
            for i in range(max(0, center_idx - ws), center_idx)
        ]
        after = [
            sorted_chunks[i]
            for i in range(center_idx + 1, min(len(sorted_chunks), center_idx + ws + 1))
        ]
        
        return {
            "center": sorted_chunks[center_idx] if center_idx < len(sorted_chunks) else None,
            "before": before,
            "after": after,
        }


class IncrementalManager:
    """Manage incremental document updates.
    
    Supports add, update, delete without full reindex.
    Inspired by Chroma's upsert capability.
    
    Example:
        manager = IncrementalManager(forge)
        manager.upsert("doc1", "New content", {"source": "web"})
        manager.delete("doc1")
    """
    
    def __init__(self, forge):
        """Initialize with DenseForge instance."""
        self.forge = forge
        self._doc_versions: dict[str, int] = {}  # doc_id -> version
    
    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add or update a document.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
            metadata: Optional metadata
            
        Returns:
            {"action": "added"|"updated", "doc_id": str, "version": int}
        """
        # Check if exists
        if doc_id in self._doc_versions:
            # Update: delete old, add new
            self.delete(doc_id, quiet=True)
            action = "updated"
        else:
            action = "added"
        
        # Ingest with metadata
        meta = metadata or {}
        meta["doc_id"] = doc_id
        
        ids = self.forge.ingest(text, metadata=meta)
        
        # Track version
        version = self._doc_versions.get(doc_id, 0) + 1
        self._doc_versions[doc_id] = version
        
        logger.info(f"Upserted document {doc_id} (v{version}), action={action}")
        
        return {
            "action": action,
            "doc_id": doc_id,
            "version": version,
            "chunk_count": len(ids),
        }
    
    def delete(self, doc_id: str, quiet: bool = False) -> bool:
        """Delete a document by ID.
        
        Note: This marks the document as deleted in metadata.
        Physical deletion requires reindex.
        
        Returns:
            True if document was found and marked for deletion
        """
        # Mark as deleted in triple store
        deleted = False
        
        # Check triple store
        for idx in range(len(self.forge.triple_store.metadata)):
            meta = self.forge.triple_store.metadata[idx]
            if meta and meta.get("doc_id") == doc_id:
                # Mark as deleted
                meta["_deleted"] = True
                deleted = True
        
        if deleted and doc_id in self._doc_versions:
            del self._doc_versions[doc_id]
        
        if not quiet and deleted:
            logger.info(f"Marked document {doc_id} for deletion")
        
        return deleted
    
    def get_version(self, doc_id: str) -> int | None:
        """Get current version of a document."""
        return self._doc_versions.get(doc_id)
    
    def list_documents(self) -> list[str]:
        """List all tracked document IDs."""
        return list(self._doc_versions.keys())
