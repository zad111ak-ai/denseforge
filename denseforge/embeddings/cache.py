"""Semantic Query Cache with vector similarity lookup — SAFE version.

Security fixes:
- CRITICAL: Replaced pickle.load() with JSON (prevents arbitrary code execution)
- Added input validation
- Added cache size limits
- Added thread safety
"""
import json
import time
import threading
from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger


class SemanticQueryCache:
    """Cache that matches by query similarity, not exact text.
    
    Security: Uses JSON serialization (safe) instead of pickle (unsafe).
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 10000,
        default_ttl: float = 3600,
        persist_path: Optional[str] = None,
    ):
        # Validate inputs
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be 0-1, got {similarity_threshold}")
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {default_ttl}")
        
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.persist_path = persist_path
        self._entries: list[dict] = []  # [{query, embedding, response, created_at}]
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
    
    def get(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        """Get cached response by query similarity."""
        # Validate inputs
        if not query or not isinstance(query, str):
            return None
        if not isinstance(query_embedding, np.ndarray):
            return None
        
        now = time.time()
        best_match = None
        best_score = 0.0
        
        with self._lock:
            for entry in self._entries:
                if now - entry["created_at"] > self.default_ttl:
                    continue
                try:
                    sim = self._cosine_similarity(query_embedding, entry["embedding"])
                    if sim >= self.similarity_threshold and sim > best_score:
                        best_score = sim
                        best_match = entry["response"]
                except Exception as e:
                    logger.warning(f"Cache similarity calc failed: {e}")
                    continue
        
        if best_match:
            self._hits += 1
            return best_match
        self._misses += 1
        return None
    
    def put(self, query: str, query_embedding: np.ndarray, response: dict):
        """Add entry to cache."""
        # Validate inputs
        if not query or not isinstance(query, str):
            return
        if not isinstance(query_embedding, np.ndarray):
            return
        if not isinstance(response, dict):
            return
        
        with self._lock:
            # Evict old entries if at capacity
            if len(self._entries) >= self.max_size:
                self._entries = self._entries[self.max_size // 2:]
                logger.debug(f"Cache evicted to {len(self._entries)} entries")
            
            self._entries.append({
                "query": query,
                "embedding": query_embedding.copy(),
                "response": response,
                "created_at": time.time(),
            })
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
        }
    
    def save(self):
        """Save cache to disk using JSON (SAFE, not pickle)."""
        if not self.persist_path:
            return
        
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert entries to JSON-serializable format
            serializable = []
            with self._lock:
                for entry in self._entries:
                    serializable.append({
                        "query": entry["query"],
                        "embedding": entry["embedding"].tolist(),  # numpy → list
                        "response": entry["response"],
                        "created_at": entry["created_at"],
                    })
            
            # Write with atomic rename (prevents corruption on crash)
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            
            # Atomic rename
            Path(tmp_path).rename(self.persist_path)
            logger.debug(f"Cache saved: {len(serializable)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def load(self):
        """Load cache from disk using JSON (SAFE, not pickle)."""
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            # Validate and convert
            entries = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "query" not in item or "embedding" not in item:
                    continue
                
                entries.append({
                    "query": item["query"],
                    "embedding": np.array(item["embedding"]),  # list → numpy
                    "response": item.get("response", {}),
                    "created_at": item.get("created_at", time.time()),
                })
            
            with self._lock:
                self._entries = entries
            
            logger.debug(f"Cache loaded: {len(entries)} entries")
            
        except json.JSONDecodeError as e:
            logger.error(f"Cache file corrupted: {e}")
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0
