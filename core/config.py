from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "AI Financial Copilot"
    DEBUG: bool = False
    ENV: str = "development"
    DEMO_MODE: bool = True

    SUPABASE_URL: str = "https://demo.supabase.co"
    SUPABASE_KEY: str = "demo-key"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "*"] 
    ALLOWED_HOSTS: list[str] = ["*"]
    

    NVIDIA_API_KEY: str = "demo-nvidia-key"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: str = "nvidia/llama-3.1-nemotron-70b-instruct"

    HF_API_KEY: str = ""
    EMBEDDING_MODEL: str = "nvidia/nv-embedqa-e5-v5"

    CHROMA_PATH: str = "./chroma_db"
    CHROMA_COLLECTION: str = "news_embeddings"

    SECRET_KEY: str = "demo-secret-key-1234567890-super-secure-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24
    LANGSMITH_TRACING: str | None = None
    LANGSMITH_ENDPOINT: str | None = None
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str | None = None
    FINNHUB_API_KEY: str = ""  

    model_config = {"env_file": ".env", "extra": "ignore"}

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()