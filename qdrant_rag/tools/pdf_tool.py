"""PDF processing and ingestion tools for RAG."""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain.tools import tool
from pypdf import PdfReader
from qdrant_client.models import PointStruct
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config
from tools.rag_tool import _get_embedding_model, _get_qdrant_client, _ensure_collection_exists


def _extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF file, preserving page information.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of dicts with 'text', 'page', and 'pdf_path'
    """
    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        pages_data = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                pages_data.append({
                    "text": text,
                    "page": page_num,
                    "pdf_path": str(pdf_path),
                    "pdf_name": pdf_path_obj.name
                })
        
        return pages_data
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the end
            for punct in ['. ', '.\n', '!\n', '?\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > chunk_size * 0.7:  # If found in last 30%
                    chunk = chunk[:last_punct + 1]
                    end = start + last_punct + 1
                    break
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


def _generate_chunk_id(pdf_path: str, page: int, chunk_index: int) -> int:
    """Generate a unique ID for a chunk."""
    content = f"{pdf_path}_{page}_{chunk_index}"
    # Use hash to generate consistent integer ID
    return int(hashlib.md5(content.encode()).hexdigest()[:15], 16)


@tool
def ingest_pdf_tool(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> str:
    """Ingest a PDF file into the knowledge base for querying.
    
    This tool extracts text from a PDF, splits it into chunks, generates embeddings,
    and stores them in Qdrant. After ingestion, you can query the PDF content using rag_search_tool.
    
    Args:
        pdf_path: Path to the PDF file to ingest
        chunk_size: Size of text chunks in characters (default: 500)
        chunk_overlap: Overlap between chunks in characters (default: 50)
    
    Returns:
        Status message indicating success or failure
    """
    try:
        # Extract text from PDF
        pages_data = _extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            return f"No text found in PDF: {pdf_path}"
        
        # Get embedding model and Qdrant client
        model = _get_embedding_model()
        client = _get_qdrant_client()
        
        # Ensure collection exists
        _ensure_collection_exists(client, Config.COLLECTION_NAME)
        
        # Process each page
        total_chunks = 0
        points_to_insert = []
        
        for page_data in pages_data:
            text = page_data["text"]
            page_num = page_data["page"]
            pdf_path_str = page_data["pdf_path"]
            pdf_name = page_data["pdf_name"]
            
            # Chunk the text
            chunks = _chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Generate embedding
                vector = model.encode(chunk).tolist()
                
                # Generate unique ID
                point_id = _generate_chunk_id(pdf_path_str, page_num, chunk_idx)
                
                # Create point with metadata
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": pdf_path_str,
                        "pdf_name": pdf_name,
                        "page": page_num,
                        "chunk_index": chunk_idx,
                        "type": "pdf"
                    }
                )
                points_to_insert.append(point)
                total_chunks += 1
        
        # Batch insert into Qdrant
        if points_to_insert:
            client.upsert(
                collection_name=Config.COLLECTION_NAME,
                points=points_to_insert
            )
            return (
                f"Successfully ingested PDF: {pdf_path}\n"
                f"- Pages processed: {len(pages_data)}\n"
                f"- Total chunks: {total_chunks}\n"
                f"- Collection: {Config.COLLECTION_NAME}"
            )
        else:
            return f"No content to ingest from PDF: {pdf_path}"
            
    except FileNotFoundError as e:
        return f"Error: {str(e)}"
    except Exception as e:
        return f"Error ingesting PDF: {str(e)}"


@tool
def query_pdf_tool(query: str, pdf_path: Optional[str] = None) -> str:
    """Query PDF content from the knowledge base.
    
    This tool searches for information in ingested PDFs. You can search across
    all PDFs or specify a particular PDF path.
    
    Args:
        query: The search query
        pdf_path: Optional - Specific PDF path to search (if None, searches all PDFs)
    
    Returns:
        Relevant excerpts from PDF(s) matching the query
    """
    try:
        from tools.rag_tool import _hybrid_search
        
        # Perform hybrid search
        results = _hybrid_search(
            query=query,
            collection_name=Config.COLLECTION_NAME,
            top_k=Config.TOP_K,
            alpha=Config.HYBRID_ALPHA
        )
        
        # Filter by PDF path if specified
        if pdf_path:
            results = [
                r for r in results 
                if r.get("payload", {}).get("source") == pdf_path
            ]
        
        # Filter to only PDF documents
        results = [
            r for r in results 
            if r.get("payload", {}).get("type") == "pdf"
        ]
        
        if not results:
            if pdf_path:
                return f"No relevant content found in PDF: {pdf_path}. Make sure the PDF has been ingested using ingest_pdf_tool."
            return "No relevant PDF content found. Make sure PDFs have been ingested using ingest_pdf_tool."
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            payload = result.get("payload", {})
            text = payload.get("text", "No content available")
            source = payload.get("source", "Unknown")
            pdf_name = payload.get("pdf_name", Path(source).name if source else "Unknown")
            page = payload.get("page", "?")
            
            formatted_results.append(
                f"[Result {i}]\n"
                f"PDF: {pdf_name}\n"
                f"Page: {page}\n"
                f"Content: {text}\n"
                f"Relevance Score: {result['combined_score']:.4f}\n"
            )
        
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error querying PDF: {str(e)}"

