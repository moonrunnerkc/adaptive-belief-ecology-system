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

    # embedding model for semantic search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # belief ecology loop parameters
    decay_half_life_hours: float = 24.0
    tension_threshold_high: float = 0.7
    tension_threshold_low: float = 0.3
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "confidence": 0.4,
            "relevance": 0.4,
            "recency": 0.1,
            "tension": 0.1,
        }
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# singleton instance
settings = ABESSettings()


__all__ = ["ABESSettings", "settings"]
