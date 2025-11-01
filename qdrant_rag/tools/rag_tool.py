"""RAG tool with hybrid search for Qdrant."""
import sys
from pathlib import Path
from typing import List, Dict, Any
from langchain.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


# Initialize embedding model (loaded once)
_embedding_model: SentenceTransformer = None
_qdrant_client: QdrantClient = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy load embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _embedding_model


def _get_qdrant_client() -> QdrantClient:
    """Initialize and return Qdrant client (supports both local and cloud)."""
    global _qdrant_client
    if _qdrant_client is None:
        Config._normalize_qdrant_url()
        # Qdrant Cloud requires API key, local doesn't
        if Config.QDRANT_API_KEY:
            _qdrant_client = QdrantClient(
                url=Config.QDRANT_URL,
                api_key=Config.QDRANT_API_KEY
            )
        else:
            # Local Qdrant instance
            _qdrant_client = QdrantClient(url=Config.QDRANT_URL)
    return _qdrant_client


def _ensure_collection_exists(client: QdrantClient, collection_name: str):
    """Ensure the Qdrant collection exists, create if it doesn't."""
    try:
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=Config.VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
    except Exception as e:
        # Collection might already exist, which is fine
        pass


def _hybrid_search(
    query: str,
    collection_name: str,
    top_k: int = 5,
    alpha: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining dense vector search and keyword matching.
    
    Args:
        query: The search query
        collection_name: Name of the Qdrant collection
        top_k: Number of results to return
        alpha: Weight for vector search (1-alpha for keyword search)
    
    Returns:
        List of search results with scores and metadata
    """
    client = _get_qdrant_client()
    _ensure_collection_exists(client, collection_name)
    
    model = _get_embedding_model()
    query_vector = model.encode(query).tolist()
    
    # Dense vector search
    vector_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k * 2  # Get more results for fusion
    )
    
    # Keyword-based search using Qdrant filters
    # Extract keywords from query (simple approach)
    query_lower = query.lower()
    keywords = [word for word in query_lower.split() if len(word) > 2]
    
    # Perform keyword search by filtering on text fields in payload
    # This is a simplified version - in production, you'd use BM25 or proper sparse vectors
    keyword_results = []
    if keywords:
        try:
            # Search with keyword matching on text field
            # Note: This assumes documents have a "text" field in payload
            all_points = client.scroll(
                collection_name=collection_name,
                limit=1000,  # Adjust based on your collection size
                with_payload=True,
                with_vectors=False
            )
            
            keyword_matches = []
            for point in all_points[0]:
                if point.payload and "text" in point.payload:
                    text = str(point.payload["text"]).lower()
                    match_score = sum(1 for keyword in keywords if keyword in text)
                    if match_score > 0:
                        keyword_matches.append({
                            "id": point.id,
                            "score": match_score / len(keywords),
                            "payload": point.payload
                        })
            
            # Sort by keyword match score
            keyword_matches.sort(key=lambda x: x["score"], reverse=True)
            keyword_results = keyword_matches[:top_k * 2]
        except Exception as e:
            # If keyword search fails, fall back to vector search only
            pass
    
    # Fusion: Combine results using weighted Reciprocal Rank Fusion
    # Create a combined result dictionary
    combined_scores = {}
    
    # Add vector search results
    for rank, result in enumerate(vector_results, 1):
        doc_id = result.id
        vector_score = result.score
        # Convert similarity to rank-based score (higher is better)
        rrf_score = 1.0 / (rank + 60)  # RRF constant = 60
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "id": doc_id,
                "vector_score": vector_score,
                "keyword_score": 0.0,
                "payload": result.payload,
                "combined_score": 0.0
            }
        combined_scores[doc_id]["combined_score"] += alpha * rrf_score
    
    # Add keyword search results
    for rank, result in enumerate(keyword_results, 1):
        doc_id = result["id"]
        keyword_score = result["score"]
        rrf_score = 1.0 / (rank + 60)
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "id": doc_id,
                "vector_score": 0.0,
                "keyword_score": keyword_score,
                "payload": result["payload"],
                "combined_score": 0.0
            }
        combined_scores[doc_id]["combined_score"] += (1 - alpha) * rrf_score
    
    # Sort by combined score and return top_k
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:top_k]
    
    return sorted_results


@tool
def rag_search_tool(query: str) -> str:
    """Search the knowledge base using hybrid search (vector + keyword).
    
    Use this tool when you need to retrieve information from the knowledge base
    or when the query requires domain-specific knowledge. This tool searches
    both by semantic similarity (vector search) and keyword matching.
    
    Args:
        query: The search query string
    
    Returns:
        Relevant document chunks from the knowledge base
    """
    try:
        results = _hybrid_search(
            query=query,
            collection_name=Config.COLLECTION_NAME,
            top_k=Config.TOP_K,
            alpha=Config.HYBRID_ALPHA
        )
        
        if not results:
            return "No relevant documents found in the knowledge base."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            payload = result.get("payload", {})
            text = payload.get("text", payload.get("content", "No content available"))
            source = payload.get("source", payload.get("url", "Unknown source"))
            
            formatted_results.append(
                f"[Document {i}]\n"
                f"Source: {source}\n"
                f"Content: {text}\n"
                f"Relevance Score: {result['combined_score']:.4f}\n"
            )
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

