"""Configuration module for the agentic RAG agent."""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the agent."""
    
    # Google Gemini API / Vertex AI
    # Set USE_VERTEX_AI=true to use Vertex AI, otherwise uses Gemini API
    USE_VERTEX_AI: bool = os.getenv("USE_VERTEX_AI", "").lower() in ("true", "1", "yes")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
    MAX_TOKENS: Optional[int] = int(os.getenv("MAX_TOKENS", "8192")) if os.getenv("MAX_TOKENS") else None
    
    @classmethod
    def get_max_retries(cls) -> int:
        """Get max retries based on whether using Vertex AI or Gemini API."""
        max_retries_env = os.getenv("MAX_RETRIES")
        if max_retries_env:
            return int(max_retries_env)
        return 6 if cls.USE_VERTEX_AI else 2
    
    # Qdrant Configuration
    # For Qdrant Cloud: Use your cloud cluster URL (e.g., https://xxxxx-xxxxx-xxxxx.qdrant.io)
    # For local: Use http://localhost:6333
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Required for Qdrant Cloud, optional for local
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "384"))  # Default for all-MiniLM-L6-v2
    
    # Embedding Model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Google Search (Optional)
    GOOGLE_SEARCH_API_KEY: Optional[str] = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_SEARCH_ENGINE_ID: Optional[str] = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    
    # RAG Configuration
    TOP_K: int = int(os.getenv("TOP_K", "5"))  # Number of documents to retrieve
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.7"))  # Weight for vector search (0.7) vs keyword (0.3)
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.USE_VERTEX_AI and not cls.GOOGLE_API_KEY:
            # For Vertex AI, credentials are handled via gcloud/auth
            raise ValueError("GOOGLE_API_KEY environment variable is required when using Gemini API (not Vertex AI)")
        
        # Validate Qdrant Cloud setup
        if cls.QDRANT_URL.startswith("https://"):
            # Qdrant Cloud requires API key
            if not cls.QDRANT_API_KEY:
                raise ValueError(
                    "QDRANT_API_KEY is required when using Qdrant Cloud (https:// URL). "
                    "Get your API key from https://cloud.qdrant.io"
                )
        
        return True

