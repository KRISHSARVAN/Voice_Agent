"""Application configuration (environment variables)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "suvit-help-chat-api"
    environment: str = "development"

    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="Path to persisted Chroma (same as ingest).",
    )
    chroma_collection_name: str = Field(default="suvit_help")
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5")

    rag_top_k: int = 6
    rag_max_context_chars: int = 12000

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    openai_timeout_s: int = 120

    mongodb_uri: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI for chat history.",
    )
    mongodb_db: str = Field(default="suvit_voice", description="Database name for application data.")
    mongodb_chat_collection: str = Field(
        default="chat_turns",
        description="Collection name for stored user/bot turns.",
    )

    cartesia_api_key: str = Field(
        default="",
        description="Cartesia API key used for Ink Whisper STT and Sonic TTS.",
    )

    cors_origins_raw: str = Field(
        default="",
        description="Comma-separated allowed origins; empty disables CORS middleware.",
    )

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_origins_raw.strip()
        if not raw:
            return []
        return [o.strip() for o in raw.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
