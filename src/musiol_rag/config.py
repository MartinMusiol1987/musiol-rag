"""
Configuration settings for Musiol-RAG.
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Retrieval settings
    top_k: int = 3
    chunk_size: int = 200
    chunk_overlap: int = 50
    
    # Database settings
    database_url: Optional[str] = None
    database_type: str = "postgresql"  # or "sqlite"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # FAISS settings
    faiss_index_path: Optional[str] = "faiss_index.bin"

    class Config:
        env_file = ".env"

# Create global settings instance
settings = Settings() 