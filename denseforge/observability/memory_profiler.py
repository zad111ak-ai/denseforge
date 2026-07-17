"""Memory Profiler — monitor and prevent memory leaks.

Features:
- Track memory usage over time
- Detect memory leaks
- Force garbage collection
- Memory limits and alerts
"""
import gc
import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    objects_count: int
    heap_mb: Optional[float] = None


@dataclass
class MemoryLeak:
    """Detected memory leak."""
    component: str
    growth_rate: float  # MB per minute
    duration: float  # seconds
    severity: str  # low, medium, high, critical


class MemoryProfiler:
    """Monitor memory usage and detect leaks.
    
    Features:
    - Periodic snapshots
    - Leak detection
    - Automatic GC hints
    - Component-level tracking
    """
    
    def __init__(
        self,
        alert_threshold_mb: float = 1000,  # 1GB
        leak_detection_window: float = 300,  # 5 minutes
        snapshot_interval: float = 60,  # 1 minute
    ):
        self.alert_threshold_mb = alert_threshold_mb
        self.leak_detection_window = leak_detection_window
        self.snapshot_interval = snapshot_interval
        
        self._snapshots: list[MemorySnapshot] = []
        self._component_sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Stats
        self._stats = {
            "snapshots": 0,
            "gc_collections": 0,
            "leaks_detected": 0,
            "alerts": 0,
        }
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        try:
            import psutil
            process = psutil.Process()
            rss_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # Fallback without psutil
            rss_mb = 0.0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            objects_count=len(gc.get_objects()),
        )
        
        with self._lock:
            self._snapshots.append(snapshot)
            self._stats["snapshots"] += 1
        
        # Check threshold
        if rss_mb > self.alert_threshold_mb:
            logger.warning(f"Memory alert: {rss_mb:.1f}MB > {self.alert_threshold_mb}MB threshold")
            self._stats["alerts"] += 1
        
        return snapshot
    
    def detect_leaks(self) -> list[MemoryLeak]:
        """Detect memory leaks from recent snapshots."""
        leaks = []
        
        with self._lock:
            if len(self._snapshots) < 2:
                return leaks
            
            # Get recent snapshots
            now = time.time()
            recent = [
                s for s in self._snapshots
                if now - s.timestamp < self.leak_detection_window
            ]
            
            if len(recent) < 2:
                return leaks
            
            # Calculate growth rate
            time_span = recent[-1].timestamp - recent[0].timestamp
            if time_span < 60:  # Need at least 1 minute
                return leaks
            
            memory_growth = recent[-1].rss_mb - recent[0].rss_mb
            growth_rate = memory_growth / (time_span / 60)  # MB per minute
            
            # Detect leak
            if growth_rate > 1.0:  # More than 1MB per minute
                severity = "low"
                if growth_rate > 10:
                    severity = "medium"
                if growth_rate > 50:
                    severity = "high"
                if growth_rate > 100:
                    severity = "critical"
                
                leak = MemoryLeak(
                    component="overall",
                    growth_rate=growth_rate,
                    duration=time_span,
                    severity=severity,
                )
                leaks.append(leak)
                self._stats["leaks_detected"] += 1
                logger.warning(f"Memory leak detected: {growth_rate:.1f}MB/min ({severity})")
        
        return leaks
    
    def force_gc(self) -> dict:
        """Force garbage collection and measure freed memory."""
        try:
            import psutil
            process = psutil.Process()
            before_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            before_mb = 0.0
        
        # Force GC
        collected = gc.collect()
        self._stats["gc_collections"] += 1
        
        try:
            import psutil
            after_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            after_mb = 0.0
        
        freed_mb = before_mb - after_mb
        
        return {
            "objects_collected": collected,
            "memory_freed_mb": freed_mb,
            "before_mb": before_mb,
            "after_mb": after_mb,
        }
    
    def track_component(self, name: str, size: int):
        """Track memory usage of a component."""
        with self._lock:
            self._component_sizes[name] = size
    
    def get_component_sizes(self) -> dict[str, int]:
        """Get tracked component sizes."""
        with self._lock:
            return self._component_sizes.copy()
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self.take_snapshot()
                
                # Check for leaks every minute
                leaks = self.detect_leaks()
                if leaks:
                    # Force GC if leak detected
                    self.force_gc()
                
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                time.sleep(self.snapshot_interval)
    
    def get_history(self, last_n: int = 10) -> list[dict]:
        """Get recent memory history."""
        with self._lock:
            snapshots = self._snapshots[-last_n:]
        
        return [
            {
                "timestamp": s.timestamp,
                "rss_mb": s.rss_mb,
                "objects": s.objects_count,
            }
            for s in snapshots
        ]
    
    def stats(self) -> dict:
        """Get profiler statistics."""
        with self._lock:
            current_rss = self._snapshots[-1].rss_mb if self._snapshots else 0
        
        return {
            **self._stats,
            "current_rss_mb": current_rss,
            "threshold_mb": self.alert_threshold_mb,
            "monitoring": self._running,
        }
