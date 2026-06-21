from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Astram AI Command Center"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development" # 'development' or 'production'

    # AI & Cloud API Keys (Pydantic will throw an error on startup if these are missing!)
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    
    # Optional Database URL (if you use PostgreSQL later)
    DATABASE_URL: str | None = None

    # This tells Pydantic to look for a .env file in the root backend directory
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # Ignore extra variables in .env that aren't defined here
    )

# @lru_cache ensures we only read the .env file once from the hard drive
# and then cache the settings in RAM for ultra-fast access across the app.
@lru_cache
def get_settings() -> Settings:
    return Settings()

# Instantiate it so you can import it easily anywhere in your app
settings = get_settings()