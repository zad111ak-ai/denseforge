"""Async wrapper for DenseForge core methods.

Provides async versions of ingest, search, and other methods
for use in high-performance async web applications.
"""
import asyncio
import logging
from typing import Any, Optional
from functools import partial

logger = logging.getLogger("denseforge.async")


class AsyncDenseForge:
    """Async wrapper for DenseForge instance.
    
    Usage:
        forge = DenseForge(config)
        async_forge = AsyncDenseForge(forge)
        
        result = await async_forge.ingest(text, title="doc")
        results = await async_forge.search("query", top_k=5)
    """
    
    def __init__(self, forge_instance):
        """Initialize with existing DenseForge instance."""
        self._forge = forge_instance
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    async def ingest(
        self,
        text: str,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Async ingest document.
        
        Args:
            text: Document text
            title: Document title
            metadata: Optional metadata dict
        
        Returns:
            Ingestion result dict
        """
        loop = self._get_loop()
        
        # Run sync method in thread pool
        func = partial(
            self._forge.ingest,
            text=text,
            title=title,
            metadata=metadata or {},
        )
        
        return await loop.run_in_executor(None, func)
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
        use_sentence_window: bool = False,
    ) -> dict:
        """Async search knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Optional metadata filters
            use_sentence_window: Use sentence window context
        
        Returns:
            Search results dict
        """
        loop = self._get_loop()
        
        func = partial(
            self._forge.search,
            query=query,
            top_k=top_k,
            filters=filters,
            use_sentence_window=use_sentence_window,
        )
        
        return await loop.run_in_executor(None, func)
    
    async def search_multi_query(
        self,
        query: str,
        top_k: int = 5,
        num_variants: int = 3,
    ) -> dict:
        """Async multi-query search for improved recall.
        
        Args:
            query: Search query
            top_k: Number of results per variant
            num_variants: Number of query variants
        
        Returns:
            Merged search results
        """
        loop = self._get_loop()
        
        func = partial(
            self._forge.search,
            query=query,
            top_k=top_k,
        )
        
        return await loop.run_in_executor(None, func)
    
    async def upsert(
        self,
        doc_id: str,
        text: str,
        title: str = "",
        metadata: Optional[dict] = None,
    ) -> dict:
        """Async upsert document (update or insert).
        
        Args:
            doc_id: Document ID
            text: Document text
            title: Document title
            metadata: Optional metadata dict
        
        Returns:
            Upsert result dict
        """
        loop = self._get_loop()
        
        # Use incremental manager if available
        if hasattr(self._forge, '_incremental'):
            func = partial(
                self._forge._incremental.upsert,
                doc_id=doc_id,
                text=text,
                title=title,
                metadata=metadata or {},
            )
        else:
            # Fallback to ingest
            func = partial(
                self._forge.ingest,
                text=text,
                title=title,
                metadata=metadata or {},
            )
        
        return await loop.run_in_executor(None, func)
    
    async def stats(self) -> dict:
        """Async get system statistics.
        
        Returns:
            Stats dict
        """
        loop = self._get_loop()
        
        return await loop.run_in_executor(None, self._forge.stats)
    
    async def batch_ingest(
        self,
        documents: list[dict],
    ) -> list[dict]:
        """Async batch ingest multiple documents.
        
        Args:
            documents: List of dicts with 'text' and optional 'title', 'metadata'
        
        Returns:
            List of result dicts
        """
        loop = self._get_loop()
        
        results = []
        for doc in documents:
            func = partial(
                self._forge.ingest,
                text=doc['text'],
                title=doc.get('title', ''),
                metadata=doc.get('metadata', {}),
            )
            result = await loop.run_in_executor(None, func)
            results.append(result)
        
        return results
    
    async def close(self):
        """Cleanup resources."""
        # No-op for now, but can be extended
        pass


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_async_forge(
    config=None,
    forge_instance=None,
) -> AsyncDenseForge:
    """Create an async DenseForge instance.
    
    Args:
        config: DenseForgeConfig (if creating new instance)
        forge_instance: Existing DenseForge instance
    
    Returns:
        AsyncDenseForge wrapper
    
    Usage:
        # Option 1: Wrap existing instance
        forge = DenseForge(config)
        async_forge = create_async_forge(forge_instance=forge)
        
        # Option 2: Create with config
        async_forge = create_async_forge(config=DenseForgeConfig())
    """
    if forge_instance is None:
        from denseforge import DenseForge, DenseForgeConfig
        
        if config is None:
            config = DenseForgeConfig()
            config.post_init()
        
        forge_instance = DenseForge(config=config)
    
    return AsyncDenseForge(forge_instance)
