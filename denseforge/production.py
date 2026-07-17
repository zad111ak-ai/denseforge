"""Production Hardening — TLS, limits, audit, read-only mode.

Security features for production deployment:
1. HTTPS/TLS encryption for daemon
2. Document count limits (DoS prevention)
3. Audit logging (GDPR compliance)
4. Read-only mode (safe production)
"""
import os
import ssl
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger("denseforge.production")


# ============================================================================
# 1. HTTPS/TLS SUPPORT
# ============================================================================

@dataclass
class TLSConfig:
    """TLS configuration for daemon."""
    enabled: bool = False
    certfile: Optional[str] = None
    keyfile: Optional[str] = None
    cafile: Optional[str] = None
    protocol: str = "TLSv1.2+"  # TLSv1.2 or TLSv1.3 only
    
    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context from config."""
        if not self.enabled:
            return None
        
        if not self.certfile or not self.keyfile:
            raise ValueError("TLS enabled but certfile/keyfile not specified")
        
        # Create secure context
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        
        # Load certificate chain
        ctx.load_cert_chain(self.certfile, self.keyfile)
        
        if self.cafile:
            ctx.load_verify_locations(self.cafile)
        
        # Disable old protocols
        ctx.options |= ssl.OP_NO_SSLv2
        ctx.options |= ssl.OP_NO_SSLv3
        ctx.options |= ssl.OP_NO_TLSv1
        ctx.options |= ssl.OP_NO_TLSv1_1
        
        # Strong ciphers only
        ctx.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20')
        
        return ctx
    
    @classmethod
    def generate_self_signed(cls, output_dir: str) -> 'TLSConfig':
        """Generate self-signed certificate for development."""
        import subprocess
        import tempfile
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        certfile = str(output_path / "cert.pem")
        keyfile = str(output_path / "key.pem")
        
        # Generate with openssl
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", keyfile,
            "-out", certfile,
            "-days", "365", "-nodes",
            "-subj", "/CN=localhost/O=DenseForge/C=US",
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        return cls(
            enabled=True,
            certfile=certfile,
            keyfile=keyfile,
        )


# ============================================================================
# 2. DOCUMENT LIMITS (DOS PREVENTION)
# ============================================================================

@dataclass
class DocumentLimits:
    """Limits for document ingestion."""
    max_documents: int = 1_000_000
    max_total_size_bytes: int = 10 * 1024 * 1024 * 1024  # 10GB
    max_document_size_bytes: int = 10 * 1024 * 1024  # 10MB
    max_ingest_per_minute: int = 100
    
    # Current state
    _current_count: int = field(default=0, repr=False)
    _current_size: int = field(default=0, repr=False)
    _ingest_times: List[float] = field(default_factory=list, repr=False)
    
    def can_ingest(self, size_bytes: int) -> tuple[bool, str]:
        """Check if ingestion is allowed.
        
        Returns:
            (allowed, reason)
        """
        now = time.time()
        
        # Check document count
        if self._current_count >= self.max_documents:
            return False, f"Document limit reached: {self._current_count}/{self.max_documents}"
        
        # Check total size
        if self._current_size + size_bytes > self.max_total_size_bytes:
            return False, f"Total size limit would be exceeded"
        
        # Check individual document size
        if size_bytes > self.max_document_size_bytes:
            return False, f"Document too large: {size_bytes} > {self.max_document_size_bytes}"
        
        # Check rate limit
        self._ingest_times = [t for t in self._ingest_times if now - t < 60]
        if len(self._ingest_times) >= self.max_ingest_per_minute:
            return False, f"Ingest rate limit: {self.max_ingest_per_minute}/min exceeded"
        
        return True, "OK"
    
    def register_ingest(self, size_bytes: int):
        """Register successful ingestion."""
        self._current_count += 1
        self._current_size += size_bytes
        self._ingest_times.append(time.time())
    
    def get_stats(self) -> dict:
        """Get current usage stats."""
        return {
            "documents": self._current_count,
            "max_documents": self.max_documents,
            "total_size_mb": round(self._current_size / 1024 / 1024, 2),
            "max_total_size_mb": round(self.max_total_size_bytes / 1024 / 1024, 2),
            "ingest_rate": len(self._ingest_times),
        }


# ============================================================================
# 3. AUDIT LOGGING
# ============================================================================

@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: float
    action: str
    client_id: str
    details: Dict[str, Any]
    success: bool
    ip_address: Optional[str] = None


class AuditLogger:
    """Audit logging for GDPR and security compliance.
    
    Tracks:
    - All data access (who, when, what)
    - Ingestion operations
    - Search queries
    - Deletion requests (GDPR)
    - Authentication attempts
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: List[AuditEntry] = []
        self._buffer_size = 100
        
        # Load existing stats
        self._stats_file = self.log_dir / "stats.json"
        self._stats = self._load_stats()
    
    def log(
        self,
        action: str,
        client_id: str,
        details: Dict[str, Any],
        success: bool = True,
        ip_address: Optional[str] = None,
    ):
        """Log an audit event."""
        entry = AuditEntry(
            timestamp=time.time(),
            action=action,
            client_id=client_id,
            details=details,
            success=success,
            ip_address=ip_address,
        )
        
        self._buffer.append(entry)
        self._stats[action] = self._stats.get(action, 0) + 1
        
        # Flush buffer
        if len(self._buffer) >= self._buffer_size:
            self._flush()
    
    def log_ingest(self, client_id: str, doc_id: str, size: int, ip: str = None):
        """Log document ingestion."""
        self.log(
            action="ingest",
            client_id=client_id,
            details={"doc_id": doc_id, "size": size},
            success=True,
            ip_address=ip,
        )
    
    def log_search(self, client_id: str, query: str, results: int, ip: str = None):
        """Log search query."""
        self.log(
            action="search",
            client_id=client_id,
            details={"query": query[:100], "results": results},
            success=True,
            ip_address=ip,
        )
    
    def log_deletion(self, client_id: str, doc_id: str, reason: str = "user_request", ip: str = None):
        """Log document deletion (GDPR compliance)."""
        self.log(
            action="delete",
            client_id=client_id,
            details={"doc_id": doc_id, "reason": reason},
            success=True,
            ip_address=ip,
        )
    
    def log_auth(self, client_id: str, success: bool, ip: str = None):
        """Log authentication attempt."""
        self.log(
            action="auth",
            client_id=client_id,
            details={"success": success},
            success=success,
            ip_address=ip,
        )
    
    def get_history(
        self,
        action: Optional[str] = None,
        client_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[dict]:
        """Get audit history."""
        self._flush()
        
        entries = []
        log_file = self.log_dir / f"{int(time.time() // 86400)}.log"
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        
                        if action and entry.get('action') != action:
                            continue
                        if client_id and entry.get('client_id') != client_id:
                            continue
                        
                        entries.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        return entries[-limit:]
    
    def _flush(self):
        """Flush buffer to disk."""
        if not self._buffer:
            return
        
        log_file = self.log_dir / f"{int(time.time() // 86400)}.log"
        
        with open(log_file, 'a') as f:
            for entry in self._buffer:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "action": entry.action,
                    "client_id": entry.client_id,
                    "details": entry.details,
                    "success": entry.success,
                    "ip_address": entry.ip_address,
                }) + "\n")
        
        self._buffer.clear()
        self._save_stats()
    
    def _load_stats(self) -> dict:
        """Load stats from disk."""
        if self._stats_file.exists():
            with open(self._stats_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_stats(self):
        """Save stats to disk."""
        with open(self._stats_file, 'w') as f:
            json.dump(self._stats, f, indent=2)


# ============================================================================
# 4. READ-ONLY MODE
# ============================================================================

class AccessMode(Enum):
    """Access mode for DenseForge."""
    READ_WRITE = "read_write"
    READ_ONLY = "read_only"
    INGEST_ONLY = "ingest_only"


class ReadOnlyGuard:
    """Guard for read-only mode in production.
    
    Prevents accidental modification of knowledge base.
    """
    
    def __init__(self, mode: AccessMode = AccessMode.READ_WRITE):
        self.mode = mode
        self._operations_blocked = 0
    
    def check_write(self, operation: str) -> bool:
        """Check if write operation is allowed.
        
        Returns:
            True if allowed, raises if blocked
        """
        if self.mode == AccessMode.READ_ONLY:
            if operation in ('ingest', 'update', 'delete', 'upsert'):
                self._operations_blocked += 1
                raise PermissionError(
                    f"Operation '{operation}' blocked in READ_ONLY mode. "
                    f"Switch to READ_WRITE to modify the knowledge base."
                )
        
        if self.mode == AccessMode.INGEST_ONLY:
            if operation in ('update', 'delete', 'upsert'):
                self._operations_blocked += 1
                raise PermissionError(
                    f"Operation '{operation}' blocked in INGEST_ONLY mode. "
                    f"Only ingestion is allowed."
                )
        
        return True
    
    def check_read(self) -> bool:
        """Check if read operation is allowed."""
        # Reads are always allowed
        return True
    
    def get_stats(self) -> dict:
        """Get access mode stats."""
        return {
            "mode": self.mode.value,
            "operations_blocked": self._operations_blocked,
        }


# ============================================================================
# 5. PRODUCTION MANAGER (COMBINES ALL)
# ============================================================================

class ProductionManager:
    """Unified production hardening manager.
    
    Combines:
    - TLS encryption
    - Document limits
    - Audit logging
    - Read-only mode
    """
    
    def __init__(
        self,
        tls_config: Optional[TLSConfig] = None,
        document_limits: Optional[DocumentLimits] = None,
        audit_logger: Optional[AuditLogger] = None,
        access_mode: AccessMode = AccessMode.READ_WRITE,
        audit_log_dir: str = "audit_logs",
    ):
        self.tls = tls_config or TLSConfig()
        self.limits = document_limits or DocumentLimits()
        self.audit = audit_logger or AuditLogger(audit_log_dir)
        self.guard = ReadOnlyGuard(access_mode)
        
        logger.info(f"ProductionManager initialized: mode={access_mode.value}")
    
    def check_ingest(self, client_id: str, text: str, ip: str = None) -> bool:
        """Check if ingestion is allowed."""
        # Check read-only mode
        self.guard.check_write("ingest")
        
        # Check limits
        size_bytes = len(text.encode('utf-8'))
        allowed, reason = self.limits.can_ingest(size_bytes)
        
        if not allowed:
            self.audit.log(
                action="ingest_denied",
                client_id=client_id,
                details={"reason": reason},
                success=False,
                ip_address=ip,
            )
            raise PermissionError(reason)
        
        return True
    
    def register_ingest(self, client_id: str, doc_id: str, text: str, ip: str = None):
        """Register successful ingestion."""
        size_bytes = len(text.encode('utf-8'))
        self.limits.register_ingest(size_bytes)
        self.audit.log_ingest(client_id, doc_id, size_bytes, ip)
    
    def check_search(self, client_id: str, query: str, ip: str = None):
        """Check and log search."""
        self.audit.log_search(client_id, query, 0, ip)
    
    def check_delete(self, client_id: str, doc_id: str, ip: str = None):
        """Check if deletion is allowed."""
        self.guard.check_write("delete")
        self.audit.log_deletion(client_id, doc_id, ip=ip)
    
    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get SSL context for HTTPS."""
        return self.tls.create_ssl_context()
    
    def stats(self) -> dict:
        """Get production stats."""
        return {
            "tls_enabled": self.tls.enabled,
            "limits": self.limits.get_stats(),
            "access_mode": self.guard.get_stats(),
        }
