"""Configuration"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_env(key: str, default=None, type_func=str):
    """Get environment variable with type conversion."""
    value = os.getenv(key, default)
    if value is None:
        return None
    try:
        return type_func(value)
    except (ValueError, TypeError):
        return default


def str_to_bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ('true', '1', 'yes', 'on')


class Config:
    """Main configuration class."""
    
    def __init__(self):
        # Server configuration
        self.server_host = get_env("RAG_PROXY_HOST", "0.0.0.0")
        self.server_port = get_env("RAG_PROXY_PORT", 8081, int)
        self.server_debug = get_env("RAG_PROXY_DEBUG", False, str_to_bool)
        self.server_workers = get_env("RAG_PROXY_WORKERS", 1, int)
        
        # Backend configuration
        self.backend_url = get_env("BACKEND_URL")
        self.backend_timeout = get_env("BACKEND_TIMEOUT", 120, int)
        self.backend_max_retries = get_env("BACKEND_MAX_RETRIES", 3, int)
        
        # RAG configuration
        self.rag_db_path = get_env("RAG_DB_PATH", "rag_index.db")
        self.rag_embedding_type = get_env("RAG_EMBEDDING_TYPE", "simple")
        self.rag_chunk_size = get_env("RAG_CHUNK_SIZE", 500, int)
        self.rag_chunk_overlap = get_env("RAG_CHUNK_OVERLAP", 50, int)
        self.rag_enabled = get_env("RAG_ENABLED", True, str_to_bool)
        self.rag_top_k = get_env("RAG_TOP_K", 5, int)
        self.rag_similarity_threshold = get_env("RAG_SIMILARITY_THRESHOLD", 0.1, float)
        self.rag_max_context_length = get_env("RAG_MAX_CONTEXT_LENGTH", 4000, int)
        
        # Logging configuration
        self.log_level = get_env("LOG_LEVEL", "INFO")
        self.log_file = get_env("LOG_FILE")
        self.log_format = get_env("LOG_FORMAT", 
                                 "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.backend_url:
            raise ValueError("BACKEND_URL must be set")
        
        if self.rag_chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be positive")
        
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be less than RAG_CHUNK_SIZE")
        
        if self.rag_top_k <= 0:
            raise ValueError("RAG_TOP_K must be positive")
        
        if self.server_port < 1 or self.server_port > 65535:
            raise ValueError("RAG_PROXY_PORT must be between 1 and 65535")


# Global configuration instance
config = Config()
