"""Tests for production hardening module."""
import pytest
import tempfile
import time
import os
from pathlib import Path


class TestTLSConfig:
    """Test TLS configuration."""
    
    def test_default_disabled(self):
        from denseforge.production import TLSConfig
        
        config = TLSConfig()
        assert config.enabled is False
        ctx = config.create_ssl_context()
        assert ctx is None
    
    def test_create_ssl_context(self):
        from denseforge.production import TLSConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            certfile = os.path.join(tmpdir, "cert.pem")
            keyfile = os.path.join(tmpdir, "key.pem")
            
            # Create dummy files
            Path(certfile).touch()
            Path(keyfile).touch()
            
            config = TLSConfig(
                enabled=True,
                certfile=certfile,
                keyfile=keyfile,
            )
            
            # Will fail because files are empty, but tests the path
            try:
                ctx = config.create_ssl_context()
            except Exception as e:
                assert "ssl" in str(e).lower() or "pem" in str(e).lower()


class TestDocumentLimits:
    """Test document limits."""
    
    def test_can_ingest_normal(self):
        from denseforge.production import DocumentLimits
        
        limits = DocumentLimits()
        allowed, reason = limits.can_ingest(1000)
        
        assert allowed is True
        assert reason == "OK"
    
    def test_blocks_too_large(self):
        from denseforge.production import DocumentLimits
        
        limits = DocumentLimits(max_document_size_bytes=1000)
        allowed, reason = limits.can_ingest(2000)
        
        assert allowed is False
        assert "too large" in reason.lower()
    
    def test_blocks_at_limit(self):
        from denseforge.production import DocumentLimits
        
        limits = DocumentLimits(max_documents=2)
        limits.register_ingest(100)
        limits.register_ingest(100)
        
        allowed, reason = limits.can_ingest(100)
        assert allowed is False
        assert "limit reached" in reason.lower()
    
    def test_stats(self):
        from denseforge.production import DocumentLimits
        
        limits = DocumentLimits()
        limits.register_ingest(1024)
        
        stats = limits.get_stats()
        assert stats["documents"] == 1
        assert stats["total_size_mb"] == 0.0  # 1KB rounds to 0


class TestAuditLogger:
    """Test audit logging."""
    
    def test_log_event(self):
        from denseforge.production import AuditLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir)
            logger.log("test", "client1", {"key": "value"})
            logger._flush()
            
            # Check log file exists
            log_files = list(Path(tmpdir).glob("*.log"))
            assert len(log_files) > 0
    
    def test_log_ingest(self):
        from denseforge.production import AuditLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir)
            logger.log_ingest("client1", "doc123", 1024)
            logger._flush()
            
            stats = logger._stats
            assert stats.get("ingest", 0) == 1
    
    def test_log_search(self):
        from denseforge.production import AuditLogger
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = AuditLogger(tmpdir)
            logger.log_search("client1", "test query", 5)
            logger._flush()
            
            stats = logger._stats
            assert stats.get("search", 0) == 1


class TestReadOnlyGuard:
    """Test read-only mode."""
    
    def test_read_write_allows_all(self):
        from denseforge.production import ReadOnlyGuard, AccessMode
        
        guard = ReadOnlyGuard(AccessMode.READ_WRITE)
        assert guard.check_write("ingest") is True
        assert guard.check_write("delete") is True
    
    def test_read_only_blocks_writes(self):
        from denseforge.production import ReadOnlyGuard, AccessMode
        
        guard = ReadOnlyGuard(AccessMode.READ_ONLY)
        
        with pytest.raises(PermissionError):
            guard.check_write("ingest")
        
        with pytest.raises(PermissionError):
            guard.check_write("delete")
    
    def test_ingest_only_allows_ingest(self):
        from denseforge.production import ReadOnlyGuard, AccessMode
        
        guard = ReadOnlyGuard(AccessMode.INGEST_ONLY)
        assert guard.check_write("ingest") is True
        
        with pytest.raises(PermissionError):
            guard.check_write("delete")


class TestProductionManager:
    """Test unified production manager."""
    
    def test_init(self):
        from denseforge.production import ProductionManager, AccessMode
        
        pm = ProductionManager(access_mode=AccessMode.READ_WRITE)
        assert pm.tls is not None
        assert pm.limits is not None
    
    def test_check_ingest(self):
        from denseforge.production import ProductionManager, AccessMode
        
        pm = ProductionManager(access_mode=AccessMode.READ_WRITE)
        assert pm.check_ingest("client1", "test text") is True
    
    def test_check_ingest_blocked(self):
        from denseforge.production import ProductionManager, AccessMode
        
        pm = ProductionManager(access_mode=AccessMode.READ_ONLY)
        
        with pytest.raises(PermissionError):
            pm.check_ingest("client1", "test text")
    
    def test_stats(self):
        from denseforge.production import ProductionManager, AccessMode
        
        pm = ProductionManager(access_mode=AccessMode.READ_WRITE)
        stats = pm.stats()
        
        assert "tls_enabled" in stats
        assert "limits" in stats
        assert "access_mode" in stats
