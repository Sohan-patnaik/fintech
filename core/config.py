from functools import lru_cache
from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AI Financial Copilot"

    ENV: str = "development"
    DEBUG: bool = False

    SUPABASE_URL: str
    SUPABASE_KEY: SecretStr

    NVIDIA_API_KEY: SecretStr
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"

    CHROMA_PATH: str = "/data/chroma_db"
    CHROMA_COLLECTION: str = "news_embeddings"

    SECRET_KEY: SecretStr
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24

    LANGSMITH_TRACING: str | None = None
    LANGSMITH_ENDPOINT: str | None = None
    LANGSMITH_API_KEY: SecretStr | None = None
    LANGSMITH_PROJECT: str | None = None

    FINNHUB_API_KEY: SecretStr | None = None

    CORS_ORIGINS: list[str]
    ALLOWED_HOSTS: list[str]

    HTTP_TIMEOUT: int = 30
    MAX_RETRIES: int = 3

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @field_validator("SUPABASE_URL")
    @classmethod
    def validate_url(cls, v):
        if not v.startswith("https://"):
            raise ValueError("SUPABASE_URL must use HTTPS")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()