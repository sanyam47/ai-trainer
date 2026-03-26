from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "FILL_ME_IN"
    SERP_API_KEY: str = "FILL_ME_IN"
    
    # Use filesystem for local broker and sqlite for result backend
    BROKER_URL: str = "filesystem://"
    RESULT_BACKEND: str = "db+sqlite:///./celery_results.db"
    DATABASE_URL: str = "sqlite:///./jobs.db"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
