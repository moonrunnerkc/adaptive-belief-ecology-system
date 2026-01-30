"""
Configuration for ABES using Pydantic settings.
Reads from environment variables or .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ABESSettings(BaseSettings):
    """Global config for the belief ecology system."""

    # environment
    environment: str = "development"

    # storage
    database_url: str = "sqlite+aiosqlite:///./data/abes.db"

    # --- Embedding Configuration (spec 3.5) ---
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64

    # --- Decay Parameters (spec 3.4.1) ---
    decay_rate: float = 0.995  # per-hour multiplier
    # half-life ~138 hours at 0.995

    # --- Status Thresholds (spec 3.4.2) ---
    confidence_threshold_decaying: float = 0.3
    confidence_threshold_deprecated: float = 0.1
    stale_days_deprecated: int = 30  # unused beliefs older than this get deprecated

    # --- Contradiction & Tension (spec 3.4.3, 3.4.4) ---
    similarity_threshold_contradiction: float = 0.5  # skip pairs below this
    tension_threshold_high: float = 0.7  # flags for resolution
    tension_cap: float = 10.0

    # --- Relevance (spec 3.4.5) ---
    relevance_threshold_min: float = 0.1  # lowered for chat - be more inclusive

    # --- Ranking Weights (spec 3.4.6) ---
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "relevance": 0.4,
            "confidence": 0.3,
            "recency": 0.2,
            "tension": 0.1,
        }
    )
    ranking_weight_bounds: tuple[float, float] = (0.1, 0.6)  # RL tuning limits
    recency_window_hours: int = 168  # 1 week for recency normalization

    # --- Mutation Triggers (spec 3.4.7) ---
    tension_threshold_mutation: float = 0.6
    confidence_threshold_mutation: float = 0.5
    mutation_strategy: str = "rule"  # "rule" or "llm"

    # --- Resolution Triggers (spec 3.4.8) ---
    tension_threshold_resolution: float = 0.7
    confidence_threshold_resolution: float = 0.6

    # --- Clustering (spec 3.6) ---
    cluster_similarity_threshold: float = 0.7  # min similarity to join cluster
    cluster_merge_threshold: float = 0.85
    cluster_maintenance_interval: int = 100  # iterations between maintenance

    # --- Safety Limits (spec 3.8) ---
    max_belief_content_length: int = 2000
    max_active_beliefs: int = 10000
    max_beliefs_per_cluster: int = 500
    max_mutation_depth: int = 5
    max_contradiction_pairs_per_iteration: int = 50000
    max_snapshot_size_mb: int = 50

    # --- Reinforcement (agent-level) ---
    reinforcement_similarity_threshold: float = 0.7
    reinforcement_confidence_boost: float = 0.1
    reinforcement_cooldown_seconds: int = 60
    max_reinforced_confidence: float = 0.95

    # --- Deduplication ---
    dedupe_similarity_threshold: float = 0.95

    # --- LLM / Ollama Configuration ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct-q4_0"
    ollama_timeout: float = 120.0
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_context_beliefs: int = 15  # max beliefs to include in LLM context

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# singleton instance
settings = ABESSettings()


__all__ = ["ABESSettings", "settings"]
