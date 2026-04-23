from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    USE_OPENAI: bool = False
    OPENAI_API_KEY: str = "FILL_ME_IN"
    SERP_API_KEY: str = "FILL_ME_IN"
    
    # Prefer REDIS_URL in cloud, fallback to local filesystem/sqlite setup.
    BROKER_URL: str = os.getenv("REDIS_URL", "filesystem://")
    RESULT_BACKEND: str = os.getenv("REDIS_URL", "db+sqlite:///./celery_results.db")
    DATABASE_URL: str = "sqlite:///./jobs.db"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
