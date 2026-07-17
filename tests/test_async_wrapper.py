"""Tests for async wrapper module."""
import pytest
import asyncio
from unittest.mock import MagicMock, patch


class TestAsyncDenseForge:
    """Test async wrapper."""
    
    def test_init(self):
        from denseforge.async_wrapper import AsyncDenseForge
        
        mock_forge = MagicMock()
        async_forge = AsyncDenseForge(mock_forge)
        
        assert async_forge._forge == mock_forge
    
    @pytest.mark.asyncio
    async def test_stats(self):
        from denseforge.async_wrapper import AsyncDenseForge
        
        mock_forge = MagicMock()
        mock_forge.stats.return_value = {"documents": 10}
        
        async_forge = AsyncDenseForge(mock_forge)
        result = await async_forge.stats()
        
        assert result == {"documents": 10}
        mock_forge.stats.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ingest(self):
        from denseforge.async_wrapper import AsyncDenseForge
        
        mock_forge = MagicMock()
        mock_forge.ingest.return_value = {"status": "ok", "doc_id": 1}
        
        async_forge = AsyncDenseForge(mock_forge)
        result = await async_forge.ingest("test text", title="test")
        
        assert result["status"] == "ok"
        mock_forge.ingest.assert_called_once_with(
            text="test text",
            title="test",
            metadata={},
        )
    
    @pytest.mark.asyncio
    async def test_search(self):
        from denseforge.async_wrapper import AsyncDenseForge
        
        mock_forge = MagicMock()
        mock_forge.search.return_value = {"results": [{"text": "found"}]}
        
        async_forge = AsyncDenseForge(mock_forge)
        result = await async_forge.search("query", top_k=5)
        
        assert "results" in result
        mock_forge.search.assert_called_once_with(
            query="query",
            top_k=5,
            filters=None,
            use_sentence_window=False,
        )
    
    @pytest.mark.asyncio
    async def test_batch_ingest(self):
        from denseforge.async_wrapper import AsyncDenseForge
        
        mock_forge = MagicMock()
        mock_forge.ingest.return_value = {"status": "ok"}
        
        async_forge = AsyncDenseForge(mock_forge)
        docs = [
            {"text": "doc1", "title": "title1"},
            {"text": "doc2", "title": "title2"},
        ]
        
        results = await async_forge.batch_ingest(docs)
        
        assert len(results) == 2
        assert mock_forge.ingest.call_count == 2


class TestCreateAsyncForge:
    """Test convenience function."""
    
    def test_create_with_instance(self):
        from denseforge.async_wrapper import create_async_forge, AsyncDenseForge
        
        mock_forge = MagicMock()
        async_forge = create_async_forge(forge_instance=mock_forge)
        
        assert isinstance(async_forge, AsyncDenseForge)
        assert async_forge._forge == mock_forge
