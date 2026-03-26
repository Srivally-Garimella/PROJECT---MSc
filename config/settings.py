"""
Configuration for TemporalGuard-RAG

Central configuration management using Pydantic settings.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # ═══════════════════════════════════════════════════════════════
    # Project Settings
    # ═══════════════════════════════════════════════════════════════
    
    PROJECT_NAME: str = "TemporalGuard-RAG"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # ═══════════════════════════════════════════════════════════════
    # API Keys
    # ═══════════════════════════════════════════════════════════════
    
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM access"
    )
    
    SEC_API_EMAIL: str = Field(
        default="your-email@example.com",
        description="Email for SEC EDGAR API User-Agent"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Model Configuration
    # ═══════════════════════════════════════════════════════════════
    
    LLM_MODEL: str = Field(
        default="gpt-4",
        description="OpenAI model for agents"
    )
    
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="HuggingFace embedding model"
    )
    
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Embedding vector dimension"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # ChromaDB Settings
    # ═══════════════════════════════════════════════════════════════
    
    CHROMA_PERSIST_DIR: str = Field(
        default="data/chroma_db",
        description="ChromaDB persistence directory"
    )
    
    CHROMA_COLLECTION_NAME: str = Field(
        default="temporal_financials",
        description="ChromaDB collection name"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Data Directories
    # ═══════════════════════════════════════════════════════════════
    
    DATA_DIR: str = Field(
        default="data",
        description="Base data directory"
    )
    
    RAW_DATA_DIR: str = Field(
        default="data/raw",
        description="Raw data directory"
    )
    
    SEC_FILINGS_DIR: str = Field(
        default="data/raw/sec_filings",
        description="SEC filings directory"
    )
    
    XBRL_DATA_DIR: str = Field(
        default="data/raw/xbrl_structured",
        description="XBRL structured data directory"
    )
    
    PROCESSED_DIR: str = Field(
        default="data/processed",
        description="Processed data directory"
    )
    
    CHUNKS_DIR: str = Field(
        default="data/processed/chunks",
        description="Text chunks directory"
    )
    
    EMBEDDINGS_DIR: str = Field(
        default="data/processed/embeddings",
        description="Embeddings directory"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Processing Settings
    # ═══════════════════════════════════════════════════════════════
    
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Text chunk size in characters"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Chunk overlap in characters"
    )
    
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Temporal Settings
    # ═══════════════════════════════════════════════════════════════
    
    FILING_LAG_10K: int = Field(
        default=60,
        description="10-K filing lag in days"
    )
    
    FILING_LAG_10Q: int = Field(
        default=40,
        description="10-Q filing lag in days"
    )
    
    FILING_LAG_8K: int = Field(
        default=4,
        description="8-K filing lag in days"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Security Settings
    # ═══════════════════════════════════════════════════════════════
    
    ENABLE_AUDIT_LOGGING: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    
    AUDIT_LOG_DIR: str = Field(
        default="data/audit",
        description="Audit log directory"
    )
    
    AUDIT_RETENTION_DAYS: int = Field(
        default=90,
        description="Days to retain audit logs"
    )
    
    ENABLE_POISONING_DETECTION: bool = Field(
        default=True,
        description="Enable data poisoning detection"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # API Settings
    # ═══════════════════════════════════════════════════════════════
    
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    
    API_PORT: int = Field(
        default=8000,
        description="API port"
    )
    
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    
    # ═══════════════════════════════════════════════════════════════
    # Streamlit Settings
    # ═══════════════════════════════════════════════════════════════
    
    STREAMLIT_PORT: int = Field(
        default=8501,
        description="Streamlit port"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def get_path(self, attribute: str) -> Path:
        """Get path attribute as Path object."""
        value = getattr(self, attribute)
        return Path(value)
        
    def ensure_directories(self):
        """Create all required directories."""
        directories = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.SEC_FILINGS_DIR,
            self.XBRL_DATA_DIR,
            self.PROCESSED_DIR,
            self.CHUNKS_DIR,
            self.EMBEDDINGS_DIR,
            self.CHROMA_PERSIST_DIR,
            self.AUDIT_LOG_DIR,
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get settings instance."""
    return settings


# Ensure directories exist on import
if __name__ != "__main__":
    try:
        settings.ensure_directories()
    except Exception as e:
        print(f"Warning: Could not create directories: {e}")
