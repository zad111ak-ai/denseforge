"""Observability — metrics collection, structured logging, and optional OpenTelemetry tracing."""
from __future__ import annotations

import json
import time
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger

from denseforge.config import ObservabilityConfig


class _Counter:
    """Thread-safe counter."""

    def __init__(self, name: str, labels: dict | None = None):
        self.name = name
        self.labels = labels or {}
        self.value: float = 0.0
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0):
        with self._lock:
            self.value += amount

    def set(self, value: float):
        with self._lock:
            self.value = value


class _Histogram:
    """Thread-safe histogram with fixed buckets."""

    BUCKETS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]

    def __init__(self, name: str, labels: dict | None = None):
        self.name = name
        self.labels = labels or {}
        self._lock = threading.Lock()
        self._observations: list[float] = []
        self._bucket_counts = [0] * (len(self.BUCKETS) + 1)  # +Inf bucket
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float):
        with self._lock:
            self._observations.append(value)
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self.BUCKETS):
                if value <= bound:
                    self._bucket_counts[i] += 1
                    break
            else:
                self._bucket_counts[-1] += 1  # +Inf

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count > 0 else 0.0

    @property
    def p50(self) -> float:
        return self._percentile(50)

    @property
    def p95(self) -> float:
        return self._percentile(95)

    @property
    def p99(self) -> float:
        return self._percentile(99)

    def _percentile(self, pct: float) -> float:
        with self._lock:
            if not self._observations:
                return 0.0
            sorted_obs = sorted(self._observations)
            idx = int(len(sorted_obs) * pct / 100)
            return sorted_obs[min(idx, len(sorted_obs) - 1)]


class MetricsCollector:
    """Collect metrics for the DenseForge pipeline.

    Tracks counters for documents/queries, histograms for latency,
    and gauges for cache/index sizes.
    """

    def __init__(self, config: ObservabilityConfig | None = None):
        self.config = config or ObservabilityConfig()
        self._counters: dict[str, _Counter] = {}
        self._histograms: dict[str, _Histogram] = {}
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()
        self._query_log: list[dict] = []

        logger.level(self.config.log_level)

    # ------------------------------------------------------------------
    # Counter API
    # ------------------------------------------------------------------
    def counter(self, name: str, labels: dict | None = None) -> _Counter:
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        with self._lock:
            if key not in self._counters:
                self._counters[key] = _Counter(name, labels)
            return self._counters[key]

    def inc(self, name: str, amount: float = 1.0, **labels):
        self.counter(name, labels).inc(amount)

    # ------------------------------------------------------------------
    # Histogram API
    # ------------------------------------------------------------------
    def histogram(self, name: str, labels: dict | None = None) -> _Histogram:
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = _Histogram(name, labels)
            return self._histograms[key]

    def observe(self, name: str, value: float, **labels):
        self.histogram(name, labels).observe(value)

    # ------------------------------------------------------------------
    # Gauge API
    # ------------------------------------------------------------------
    def gauge(self, name: str, value: float):
        with self._lock:
            self._gauges[name] = value

    # ------------------------------------------------------------------
    # Query log (structured)
    # ------------------------------------------------------------------
    def log_query(self, query: str, latency_ms: float, cache_hit: bool,
                  n_sources: int, tokens_used: int = 0):
        entry = {
            "ts": time.time(),
            "query": query[:200],
            "latency_ms": round(latency_ms, 1),
            "cache_hit": cache_hit,
            "n_sources": n_sources,
            "tokens": tokens_used,
        }
        self._query_log.append(entry)
        logger.info(
            "query latency={}ms cache_hit={} sources={}",
            entry["latency_ms"], entry["cache_hit"], entry["n_sources"],
        )

    # ------------------------------------------------------------------
    # Context-manager for timing
    # ------------------------------------------------------------------
    def timed(self, metric_name: str, **labels):
        """Return a context-manager that times a block and records it."""
        return _TimedBlock(self, metric_name, labels)

    # ------------------------------------------------------------------
    # Aggregated snapshot
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Return a full metrics snapshot."""
        counters = {}
        for c in self._counters.values():
            counters[c.name] = {"value": c.value, "labels": c.labels}

        histograms = {}
        for h in self._histograms.values():
            histograms[h.name] = {
                "count": h._count,
                "sum": round(h._sum, 6),
                "mean": round(h.mean, 6),
                "p50": round(h.p50, 6),
                "p95": round(h.p95, 6),
                "p99": round(h.p99, 6),
                "labels": h.labels,
            }

        return {
            "counters": counters,
            "histograms": histograms,
            "gauges": dict(self._gauges),
            "total_queries_logged": len(self._query_log),
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: list[str] = []
        for c in self._counters.values():
            labels = self._fmt_labels(c.labels)
            lines.append(f"denseforge_{c.name}{labels} {c.value}")
        for h in self._histograms.values():
            labels = self._fmt_labels(h.labels)
            lines.append(f"denseforge_{h.name}_count{labels} {h._count}")
            lines.append(f"denseforge_{h.name}_sum{labels} {round(h._sum, 6)}")
            for bound, count in zip(h.BUCKETS + [float("inf")], h._bucket_counts):
                le = "+Inf" if bound == float("inf") else bound
                lines.append(
                    f"denseforge_{h.name}_bucket{labels}{{le=\"{le}\"}} {count}"
                )
        for name, val in self._gauges.items():
            lines.append(f"denseforge_{name} {val}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _fmt_labels(labels: dict) -> str:
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(parts) + "}"


class _TimedBlock:
    """Context-manager that records elapsed time."""

    def __init__(self, collector: MetricsCollector, name: str, labels: dict):
        self._collector = collector
        self._name = name
        self._labels = labels
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self._collector.observe(self._name, elapsed, **self._labels)
