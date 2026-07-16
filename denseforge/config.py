"""DenseForge Configuration."""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class EmbeddingConfig:
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    device: str = "cpu"
    default_dim: int = 512
    supported_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 768])
    batch_size: int = 64
    normalize: bool = True


@dataclass
class RetrievalConfig:
    use_bm25: bool = True
    use_dense: bool = True
    use_splade: bool = False
    use_raptor: bool = True
    use_hippo: bool = True
    use_cag: bool = True
    top_k: int = 10
    retrieve_k: int = 50
    max_tokens: int = 8000
    cag_threshold_tokens: int = 100_000
    raptor_max_levels: int = 3
    raptor_cluster_ratio: float = 0.3


@dataclass
class RerankerConfig:
    model_name: str = "BAAI/bge-reranker-v2-m3"
    device: str = "cpu"
    rerank_weight: float = 0.7
    use_isolated: bool = True


@dataclass
class GenerationConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 2000
    use_attribution: bool = True
    speculative_drafts: int = 3


@dataclass
class MemoryConfig:
    enable_episodic: bool = True
    enable_semantic: bool = True
    enable_procedural: bool = True
    consolidation_interval_hours: int = 24
    max_episodic_age_days: int = 30
    ewc_lambda: float = 1000.0
    replay_buffer_size: int = 1000


@dataclass
class CacheConfig:
    enable_semantic_cache: bool = True
    similarity_threshold: float = 0.92
    max_size: int = 10000
    default_ttl_seconds: float = 3600
    persist_path: Optional[str] = "./cache/semantic_cache.pkl"


@dataclass
class AgentConfig:
    enable_multi_agent: bool = True
    dynamic_spawning: bool = True
    max_agents: int = 10
    coordination_threshold: float = 0.7


@dataclass
class CarbonConfig:
    enable_carbon_aware: bool = False
    carbon_api_key: Optional[str] = None
    region: str = "EU"
    daily_budget_wh: float = 1000.0
    defer_threshold_intensity: float = 500.0


@dataclass
class SecurityConfig:
    enable_encryption: bool = False
    enable_pii_detection: bool = True
    enable_adversarial_defense: bool = True
    audit_log_path: Optional[str] = "./logs/audit.jsonl"


@dataclass
class ObservabilityConfig:
    enable_tracing: bool = False
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"


@dataclass
class DenseForgeConfig:
    data_dir: Path = field(default_factory=lambda: Path("./denseforge_data"))
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    carbon: CarbonConfig = field(default_factory=CarbonConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    llm_provider: str = "omniroute"
    llm_api_key: Optional[str] = None
    llm_base_url: str = "http://localhost:3000/v1"

    def post_init(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.cache.persist_path:
            Path(self.cache.persist_path).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "DenseForgeConfig":
        import os
        return cls(
            llm_provider=os.getenv("DENSEFORGE_LLM_PROVIDER", "omniroute"),
            llm_api_key=os.getenv("DENSEFORGE_LLM_API_KEY"),
            llm_base_url=os.getenv("DENSEFORGE_LLM_BASE_URL", "http://localhost:3000/v1"),
            embedding=EmbeddingConfig(device=os.getenv("DENSEFORGE_DEVICE", "cpu")),
        )
