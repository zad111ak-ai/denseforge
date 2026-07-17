"""DenseForge Robustness — Input Validation, Error Handling, Rate Limiting.

This module provides:
1. Input validation & sanitization (empty, length, special chars)
2. Graceful degradation (fallbacks on errors)
3. Rate limiting (protection against overload)
4. Error recovery (retry logic)
"""
import re
import time
from typing import Any, Callable, Optional
from collections import deque
from loguru import logger


# ============================================================================
# 1. INPUT VALIDATION & SANITIZATION
# ============================================================================

class InputValidator:
    """Validate and sanitize inputs before processing."""
    
    # Limits
    MAX_TEXT_LENGTH = 50000  # ~10K words
    MIN_TEXT_LENGTH = 1      # At least 1 char
    MAX_QUERY_LENGTH = 10000
    MIN_QUERY_LENGTH = 1
    
    # Patterns
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    WHITESPACE_PATTERN = re.compile(r'\s+')
    INJECTION_PATTERNS = [
        re.compile(r'ignore\s+(all\s+)?previous\s+instructions', re.IGNORECASE),
        re.compile(r'you\s+are\s+now', re.IGNORECASE),
        re.compile(r'system\s*prompt', re.IGNORECASE),
        re.compile(r'<\|system\|>', re.IGNORECASE),
    ]
    
    @classmethod
    def validate_text(cls, text: Any, field_name: str = "text") -> str:
        """Validate and sanitize text input.
        
        Args:
            text: Input text
            field_name: Name for error messages
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If validation fails
        """
        # Type check
        if not isinstance(text, str):
            raise ValueError(f"{field_name}: Expected string, got {type(text).__name__}")
        
        # Empty check
        if not text or not text.strip():
            raise ValueError(f"{field_name}: Empty or whitespace-only input")
        
        # Sanitize
        text = cls.sanitize(text)
        
        # Length check
        if len(text) < cls.MIN_TEXT_LENGTH:
            raise ValueError(f"{field_name}: Too short ({len(text)} < {cls.MIN_TEXT_LENGTH})")
        
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise ValueError(f"{field_name}: Too long ({len(text)} > {cls.MAX_TEXT_LENGTH})")
        
        return text
    
    @classmethod
    def validate_query(cls, query: Any) -> str:
        """Validate query input."""
        return cls.validate_text(query, field_name="query")
    
    @classmethod
    def sanitize(cls, text: str) -> str:
        """Sanitize text input."""
        # Remove control characters
        text = cls.CONTROL_CHAR_PATTERN.sub('', text)
        
        # Normalize whitespace
        text = cls.WHITESPACE_PATTERN.sub(' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    @classmethod
    def detect_injection(cls, text: str) -> bool:
        """Detect potential prompt injection attempts."""
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(text):
                return True
        return False
    
    @classmethod
    def safe_validate(cls, text: Any, field_name: str = "text") -> tuple[bool, str, Optional[str]]:
        """Safe validation that returns (is_valid, text, error_message)."""
        try:
            validated = cls.validate_text(text, field_name)
            return True, validated, None
        except ValueError as e:
            return False, "", str(e)


# ============================================================================
# 2. ERROR HANDLING & GRACEFUL DEGRADATION
# ============================================================================

class ErrorRecovery:
    """Handle errors gracefully with fallbacks."""
    
    @staticmethod
    def with_fallback(
        primary_fn: Callable,
        fallback_fn: Callable,
        error_types: tuple = (Exception,),
        log_error: bool = True
    ) -> Any:
        """Execute primary function with fallback on error.
        
        Args:
            primary_fn: Main function to try
            fallback_fn: Fallback function if primary fails
            error_types: Tuple of exception types to catch
            log_error: Whether to log the error
            
        Returns:
            Result from primary or fallback
        """
        try:
            return primary_fn()
        except error_types as e:
            if log_error:
                logger.warning(f"Primary failed: {e}, using fallback")
            return fallback_fn()
    
    @staticmethod
    def with_retry(
        fn: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff: float = 2.0,
        error_types: tuple = (Exception,),
    ) -> Any:
        """Execute function with retry logic.
        
        Args:
            fn: Function to execute
            max_retries: Maximum retry attempts
            delay: Initial delay between retries
            backoff: Backoff multiplier
            error_types: Tuple of exception types to catch
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_error = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except error_types as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")
        
        raise last_error


class GracefulDegradation:
    """Provide degraded functionality when components fail."""
    
    def __init__(self):
        self._degraded = False
        self._degraded_components: list[str] = []
    
    def mark_degraded(self, component: str):
        """Mark a component as degraded."""
        self._degraded = True
        if component not in self._degraded_components:
            self._degraded_components.append(component)
            logger.warning(f"Component degraded: {component}")
    
    def is_degraded(self) -> bool:
        """Check if system is in degraded mode."""
        return self._degraded
    
    def get_degraded_components(self) -> list[str]:
        """Get list of degraded components."""
        return self._degraded_components.copy()
    
    def query_degraded(self, query: str, forge) -> dict:
        """Execute query in degraded mode (simplified pipeline)."""
        try:
            # Try basic retrieval without advanced features
            results = forge.triple_store.search(
                query_embedding=forge.embedder.encode(query),
                top_k=5
            )
            return {
                "results": results,
                "mode": "degraded",
                "warning": "Running in degraded mode — some features unavailable"
            }
        except Exception as e:
            logger.error(f"Degraded query also failed: {e}")
            return {
                "results": [],
                "mode": "failed",
                "error": str(e)
            }


# ============================================================================
# 3. RATE LIMITING
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60,
        burst_size: int = 10
    ):
        self.max_requests = max_requests
        self.window = window_seconds
        self.burst_size = burst_size
        self._requests: deque = deque()
        self._burst_count = 0
        self._burst_start = time.time()
    
    def allow(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        # Clean old requests
        while self._requests and self._requests[0] < now - self.window:
            self._requests.popleft()
        
        # Check window limit
        if len(self._requests) >= self.max_requests:
            return False
        
        # Check burst limit
        if now - self._burst_start > 1.0:  # Reset burst every second
            self._burst_count = 0
            self._burst_start = now
        
        if self._burst_count >= self.burst_size:
            return False
        
        # Allow
        self._requests.append(now)
        self._burst_count += 1
        return True
    
    def wait_time(self) -> float:
        """Get time to wait before next request is allowed."""
        if self.allow():
            return 0.0
        
        now = time.time()
        if self._requests:
            oldest = self._requests[0]
            return max(0, oldest + self.window - now)
        return 0.0
    
    def stats(self) -> dict:
        """Get rate limiter stats."""
        return {
            "requests_in_window": len(self._requests),
            "max_requests": self.max_requests,
            "window_seconds": self.window,
            "burst_count": self._burst_count,
            "burst_size": self.burst_size,
        }


# ============================================================================
# 4. QUERY EXPANSION (SYNONYMS)
# ============================================================================

class QueryExpander:
    """Expand queries with synonyms and related terms."""
    
    def __init__(self):
        # Synonym dictionary (English + Russian)
        self._synonyms: dict[str, list[str]] = {
            # English
            "ml": ["machine learning", "машинное обучение"],
            "ai": ["artificial intelligence", "искусственный интеллект"],
            "dl": ["deep learning", "глубокое обучение"],
            "nlp": ["natural language processing", "обработка естественного языка"],
            "cv": ["computer vision", "компьютерное зрение"],
            "nn": ["neural network", "нейронная сеть"],
            "dnn": ["deep neural network", "глубокая нейронная сеть"],
            "cnn": ["convolutional neural network", "сверточная нейронная сеть"],
            "rnn": ["recurrent neural network", "рекуррентная нейронная сеть"],
            "lstm": ["long short-term memory", "долгая краткосрочная память"],
            "gan": ["generative adversarial network", "генеративно-состязательная сеть"],
            "rl": ["reinforcement learning", "обучение с подкреплением"],
            "bert": ["bidirectional encoder representations", "двунаправленные представления"],
            "gpt": ["generative pre-trained transformer", "генеративный трансформер"],
            "llm": ["large language model", "большая языковая модель"],
            "rag": ["retrieval augmented generation", "генерация с извлечением"],
            "api": ["application programming interface", "интерфейс программирования приложений"],
            "db": ["database", "база данных"],
            "sql": ["structured query language", "язык структурированных запросов"],
            "nosql": ["not only sql", "не только sql"],
            "etl": ["extract transform load", "извлечение трансформация загрузка"],
            "ci": ["continuous integration", "непрерывная интеграция"],
            "cd": ["continuous delivery", "непрерывная доставка"],
            "k8s": ["kubernetes", "кубернетес"],
            "docker": ["container", "контейнер"],
            "aws": ["amazon web services", "amazon веб сервисы"],
            "gcp": ["google cloud platform", "google cloud"],
            "azure": ["microsoft azure", "майкрософт ажур"],
            
            # Russian
            "ии": ["искусственный интеллект", "artificial intelligence"],
            "мо": ["машинное обучение", "machine learning"],
            "го": ["глубокое обучение", "deep learning"],
            "оё": ["обработка естественного языка", "nlp"],
            "кз": ["компьютерное зрение", "computer vision"],
            "нс": ["нейронная сеть", "neural network"],
            "бд": ["база данных", "database"],
            "си": ["сигналы и извещения", "signals"],
            "сб": ["система безопасности", "security system"],
        }
        
        # Build reverse index
        self._reverse_index: dict[str, str] = {}
        for key, synonyms in self._synonyms.items():
            for syn in synonyms:
                self._reverse_index[syn.lower()] = key
    
    def expand(self, query: str, max_expansions: int = 3) -> list[str]:
        """Expand query with synonyms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries (including original)
        """
        queries = [query]
        
        # Find acronyms in query (strip punctuation for matching)
        words = query.split()
        for word in words:
            # Strip punctuation for matching
            clean = re.sub(r'[^\w]', '', word).lower()
            if clean in self._synonyms and len(queries) < max_expansions + 1:
                synonyms = self._synonyms[clean]
                if synonyms:
                    expanded = query.replace(word, synonyms[0])
                    if expanded not in queries:
                        queries.append(expanded)
        
        return queries
    
    def add_synonym(self, key: str, synonyms: list[str]):
        """Add custom synonyms."""
        self._synonyms[key.lower()] = synonyms
        for syn in synonyms:
            self._reverse_index[syn.lower()] = key.lower()
    
    def get_synonyms(self, term: str) -> list[str]:
        """Get synonyms for a term."""
        term_lower = term.lower()
        if term_lower in self._synonyms:
            return self._synonyms[term_lower]
        
        # Check reverse index
        if term_lower in self._reverse_index:
            key = self._reverse_index[term_lower]
            return self._synonyms.get(key, [])
        
        return []


# ============================================================================
# 5. COMBINED ROBUSTNESS MANAGER
# ============================================================================

class RobustnessManager:
    """Combined manager for all robustness features."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        burst_size: int = 10,
    ):
        self.validator = InputValidator()
        self.recovery = ErrorRecovery()
        self.degradation = GracefulDegradation()
        self.rate_limiter = RateLimiter(max_requests, window_seconds, burst_size)
        self.query_expander = QueryExpander()
    
    def safe_query(
        self,
        query: str,
        forge,
        top_k: int = 5,
        use_expansion: bool = True,
    ) -> dict:
        """Execute query with full robustness.
        
        Pipeline:
        1. Rate limit check
        2. Input validation
        3. Query expansion (optional)
        4. Execute with fallback
        5. Return results
        """
        # 1. Rate limit
        if not self.rate_limiter.allow():
            wait = self.rate_limiter.wait_time()
            return {
                "results": [],
                "error": f"Rate limited. Wait {wait:.1f}s",
                "rate_limited": True,
            }
        
        # 2. Validate input
        is_valid, validated_query, error = self.validator.safe_validate(query)
        if not is_valid:
            return {
                "results": [],
                "error": error,
                "validation_error": True,
            }
        
        # 3. Expand query
        if use_expansion:
            queries = self.query_expander.expand(validated_query)
        else:
            queries = [validated_query]
        
        # 4. Execute with fallback
        try:
            results = forge.query(validated_query, top_k=top_k)
            return {
                "results": results,
                "expanded_queries": queries,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
            # Try degraded mode
            if not self.degradation.is_degraded():
                self.degradation.mark_degraded("query_pipeline")
            
            return self.degradation.query_degraded(validated_query, forge)
    
    def safe_ingest(
        self,
        text: str,
        forge,
        **kwargs,
    ) -> dict:
        """Execute ingest with full robustness."""
        # 1. Validate
        is_valid, validated_text, error = self.validator.safe_validate(text)
        if not is_valid:
            return {
                "success": False,
                "error": error,
            }
        
        # 2. Detect injection
        if self.validator.detect_injection(validated_text):
            logger.warning(f"Potential injection detected in text")
            return {
                "success": False,
                "error": "Potential injection detected",
            }
        
        # 3. Execute
        try:
            forge.ingest(validated_text, **kwargs)
            return {"success": True}
        except Exception as e:
            logger.error(f"Ingest failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def stats(self) -> dict:
        """Get robustness stats."""
        return {
            "rate_limiter": self.rate_limiter.stats(),
            "degraded": self.degradation.is_degraded(),
            "degraded_components": self.degradation.get_degraded_components(),
        }
