"""Security Module — API authentication, rate limiting, input sanitization.

Covers:
1. API key authentication for daemon
2. DoS protection with rate limiting
3. BM25 query escaping (injection prevention)
4. Output sanitization (prompt injection prevention)
5. Input validation with strict limits
"""
import time
import secrets
import re
import logging
from typing import Optional, Dict
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("denseforge.security")


# ============================================================================
# 1. API KEY AUTHENTICATION
# ============================================================================

@dataclass
class APIKey:
    """API key with metadata."""
    key: str
    name: str
    created_at: float
    last_used: float = 0.0
    requests: int = 0
    rate_limit: int = 100  # requests per minute
    active: bool = True


class APIKeyManager:
    """Manage API keys for daemon authentication."""
    
    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path
        self._keys: Dict[str, APIKey] = {}
        self._load_or_create()
    
    def _load_or_create(self):
        """Load keys from disk or create default."""
        import json
        from pathlib import Path
        
        if self.persist_path and Path(self.persist_path).exists():
            try:
                with open(self.persist_path, 'r') as f:
                    data = json.load(f)
                for key, info in data.items():
                    self._keys[key] = APIKey(
                        key=key,
                        name=info.get('name', 'default'),
                        created_at=info.get('created_at', time.time()),
                        last_used=info.get('last_used', 0),
                        requests=info.get('requests', 0),
                        rate_limit=info.get('rate_limit', 100),
                        active=info.get('active', True),
                    )
                logger.info(f"Loaded {len(self._keys)} API keys")
            except Exception as e:
                logger.error(f"Failed to load API keys: {e}")
        
        # Create default key if none exist
        if not self._keys:
            self.create_key("default", rate_limit=1000)
    
    def create_key(self, name: str, rate_limit: int = 100) -> str:
        """Create a new API key."""
        key = "df_" + secrets.token_hex(32)
        self._keys[key] = APIKey(
            key=key,
            name=name,
            created_at=time.time(),
            rate_limit=rate_limit,
        )
        self._save()
        logger.info(f"Created API key: {name}")
        return key
    
    def validate(self, key: str) -> Optional[APIKey]:
        """Validate an API key."""
        api_key = self._keys.get(key)
        if not api_key:
            return None
        if not api_key.active:
            return None
        
        # Update usage
        api_key.last_used = time.time()
        api_key.requests += 1
        self._save()
        
        return api_key
    
    def revoke(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self._keys:
            self._keys[key].active = False
            self._save()
            return True
        return False
    
    def list_keys(self) -> list[dict]:
        """List all keys (without exposing full key)."""
        return [
            {
                "key_preview": k[:8] + "..." + k[-4:],
                "name": v.name,
                "active": v.active,
                "requests": v.requests,
                "rate_limit": v.rate_limit,
            }
            for k, v in self._keys.items()
        ]
    
    def _save(self):
        """Save keys to disk."""
        if not self.persist_path:
            return
        
        import json
        from pathlib import Path
        
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: {
                    "name": v.name,
                    "created_at": v.created_at,
                    "last_used": v.last_used,
                    "requests": v.requests,
                    "rate_limit": v.rate_limit,
                    "active": v.active,
                }
                for key, v in self._keys.items()
            }
            
            # Atomic write
            tmp_path = self.persist_path + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).rename(self.persist_path)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")


# ============================================================================
# 2. DOS PROTECTION (RATE LIMITING)
# ============================================================================

@dataclass
class RateLimitState:
    """Rate limit state for a client."""
    requests: list[float] = field(default_factory=list)
    blocked_until: float = 0.0
    violations: int = 0


class DoSProtection:
    """Rate limiting and DoS protection."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 100,
        max_requests_per_second: int = 10,
        burst_limit: int = 50,
        block_duration: float = 300,  # 5 minutes
    ):
        self.max_rpm = max_requests_per_minute
        self.max_rps = max_requests_per_second
        self.burst_limit = burst_limit
        self.block_duration = block_duration
        self._clients: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._global_requests: list[float] = []
    
    def check(self, client_id: str) -> tuple[bool, str]:
        """Check if request is allowed.
        
        Returns:
            (allowed, reason)
        """
        now = time.time()
        state = self._clients[client_id]
        
        # Check if blocked
        if state.blocked_until > now:
            remaining = int(state.blocked_until - now)
            return False, f"Blocked for {remaining}s (too many violations)"
        
        # Clean old requests
        state.requests = [t for t in state.requests if now - t < 60]
        self._global_requests = [t for t in self._global_requests if now - t < 1]
        
        # Check per-second limit
        recent_1s = sum(1 for t in state.requests if now - t < 1)
        if recent_1s >= self.max_rps:
            state.violations += 1
            if state.violations >= 3:
                state.blocked_until = now + self.block_duration
                return False, f"Blocked for {self.block_duration}s (3 violations)"
            return False, f"Rate limit: {self.max_rps} req/s exceeded"
        
        # Check per-minute limit
        if len(state.requests) >= self.max_rpm:
            state.violations += 1
            if state.violations >= 3:
                state.blocked_until = now + self.block_duration
                return False, f"Blocked for {self.block_duration}s (3 violations)"
            return False, f"Rate limit: {self.max_rpm} req/min exceeded"
        
        # Check burst
        recent_10s = sum(1 for t in state.requests if now - t < 10)
        if recent_10s >= self.burst_limit:
            state.violations += 1
            return False, f"Burst limit: {self.burst_limit} req/10s exceeded"
        
        # Check global rate
        if len(self._global_requests) >= 1000:  # 1000 req/s global
            return False, "Global rate limit exceeded"
        
        # Allow
        state.requests.append(now)
        self._global_requests.append(now)
        return True, "OK"
    
    def get_stats(self) -> dict:
        """Get rate limiting stats."""
        return {
            "active_clients": len(self._clients),
            "blocked_clients": sum(
                1 for s in self._clients.values()
                if s.blocked_until > time.time()
            ),
            "total_violations": sum(s.violations for s in self._clients.values()),
        }


# ============================================================================
# 3. BM25 QUERY ESCAPING (INJECTION PREVENTION)
# ============================================================================

class QueryEscaper:
    """Escape special characters in queries for BM25 safety."""
    
    # Characters that can cause issues in BM25/tokenization
    DANGEROUS_CHARS = re.compile(r'[<>{}[\]\\|`~!@#$%^&*()=+\s]{3,}')
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        re.compile(r'ignore\s+(previous|all|above)\s+instructions', re.IGNORECASE),
        re.compile(r'you\s+are\s+now', re.IGNORECASE),
        re.compile(r'forget\s+everything', re.IGNORECASE),
        re.compile(r'system\s*prompt', re.IGNORECASE),
        re.compile(r'\[INST\]', re.IGNORECASE),
        re.compile(r'<<SYS>>', re.IGNORECASE),
        re.compile(r'<\|im_start\|>', re.IGNORECASE),
    ]
    
    @classmethod
    def escape(cls, query: str) -> str:
        """Escape query for safe BM25 processing."""
        if not query or not isinstance(query, str):
            return ""
        
        # Trim
        query = query.strip()
        
        # Length limit
        if len(query) > 10000:
            query = query[:10000]
            logger.warning("Query truncated to 10000 chars")
        
        # Remove control characters
        query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', query)
        
        # Collapse multiple dangerous chars
        query = cls.DANGEROUS_CHARS.sub(' ', query)
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)
        
        return query.strip()
    
    @classmethod
    def detect_injection(cls, query: str) -> bool:
        """Detect potential prompt injection attempts."""
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected: {query[:50]}...")
                return True
        return False
    
    @classmethod
    def sanitize_for_bm25(cls, query: str) -> str:
        """Full sanitization pipeline for BM25 queries."""
        # Escape
        query = cls.escape(query)
        
        # Detect injection
        if cls.detect_injection(query):
            # Strip injection attempt, keep rest
            for pattern in cls.INJECTION_PATTERNS:
                query = pattern.sub('', query)
            query = cls.escape(query)
        
        return query


# ============================================================================
# 4. OUTPUT SANITIZATION (PROMPT INJECTION PREVENTION)
# ============================================================================

class OutputSanitizer:
    """Sanitize output to prevent prompt injection in downstream LLMs."""
    
    # Patterns that could be interpreted as instructions
    INSTRUCTION_PATTERNS = [
        re.compile(r'^(system|assistant|user)\s*:', re.IGNORECASE),
        re.compile(r'\[/?(INST|SYS|human|assistant)\]', re.IGNORECASE),
        re.compile(r'<\|(im_start|im_end|system|user|assistant)\|>', re.IGNORECASE),
        re.compile(r'<<SYS>>|<</SYS>>', re.IGNORECASE),
    ]
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitize text output for safe use in LLM prompts."""
        if not text:
            return ""
        
        # Truncate
        if len(text) > 50000:
            text = text[:50000]
        
        # Escape special markers
        for pattern in cls.INSTRUCTION_PATTERNS:
            text = pattern.sub(lambda m: '\\' + m.group(0), text)
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u2060-\u2064\ufeff]', '', text)
        
        return text
    
    @classmethod
    def wrap_for_context(cls, text: str) -> str:
        """Wrap text in clear delimiters for LLM context."""
        sanitized = cls.sanitize(text)
        return (
            f"<document_start>\n"
            f"{sanitized}\n"
            f"<document_end>"
        )


# ============================================================================
# 5. INPUT VALIDATOR
# ============================================================================

class InputValidator:
    """Validate all inputs to DenseForge."""
    
    MAX_TEXT_LENGTH = 1_000_000  # 1MB
    MAX_QUERY_LENGTH = 10_000
    MAX_TITLE_LENGTH = 1_000
    MAX_TOP_K = 100
    
    @classmethod
    def validate_text(cls, text: str, field_name: str = "text") -> str:
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            raise ValueError(f"{field_name} must be a string")
        
        text = text.strip()
        
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise ValueError(
                f"{field_name} too long: {len(text)} > {cls.MAX_TEXT_LENGTH}"
            )
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Validate search query."""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        query = query.strip()
        
        if not query:
            raise ValueError("Query cannot be empty")
        
        if len(query) > cls.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long: {len(query)} > {cls.MAX_QUERY_LENGTH}"
            )
        
        return cls.sanitize_for_bm25(query)
    
    @classmethod
    def validate_top_k(cls, top_k: int) -> int:
        """Validate top_k parameter."""
        if not isinstance(top_k, int):
            try:
                top_k = int(top_k)
            except (ValueError, TypeError):
                top_k = 5
        
        return max(1, min(top_k, cls.MAX_TOP_K))
    
    @classmethod
    def sanitize_for_bm25(cls, query: str) -> str:
        """Delegate to QueryEscaper."""
        return QueryEscaper.sanitize_for_bm25(query)
