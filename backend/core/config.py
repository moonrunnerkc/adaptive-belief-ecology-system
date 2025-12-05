"""
Configuration for ABES using Pydantic settings.
Reads from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class ABESSettings(BaseSettings):
    """Global config for the belief ecology system."""

    # environment
    environment: str = "development"

    # storage
    database_url: str = "sqlite+aiosqlite:///./data/abes.db"

    # embedding model for semantic search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# singleton instance
settings = ABESSettings()


__all__ = ["ABESSettings", "settings"]
