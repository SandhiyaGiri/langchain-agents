# Agentic RAG Agent

A Python-based agentic RAG (Retrieval-Augmented Generation) agent using Gemini API, Qdrant vector store, and LangGraph.

## Features

- **PDF Processing**: Ingest and query PDF documents with automatic text extraction, chunking, and embedding
- **LLM**: Gemini API (free tier) via `langchain-google-genai`
- **Vector Store**: Qdrant with hybrid search (dense vector + keyword matching)
- **Agent Framework**: LangGraph for agentic RAG workflow
- **Tools**: 
  - PDF ingestion and querying
  - Google search (with DuckDuckGo fallback)
  - URL content fetching
  - RAG search with hybrid search
- **Memory**: InMemorySaver for short-term conversation state
- **Guardrails**: Human-in-the-Loop middleware for sensitive operations (if available)

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```
   
   **Option 1: Using Gemini API** (Default):
   ```env
   USE_VERTEX_AI=false
   GOOGLE_API_KEY=your_google_api_key_here
   GEMINI_MODEL=gemini-2.5-flash
   TEMPERATURE=0
   QDRANT_URL=https://your-cluster-id.qdrant.io
   QDRANT_API_KEY=your_qdrant_cloud_api_key
   ```
   
   **Option 2: Using Vertex AI**:
   ```env
   USE_VERTEX_AI=true
   GEMINI_MODEL=gemini-2.5-flash
   TEMPERATURE=0
   # GOOGLE_API_KEY not needed - uses gcloud credentials
   QDRANT_URL=https://your-cluster-id.qdrant.io
   QDRANT_API_KEY=your_qdrant_cloud_api_key
   ```
   
   For Vertex AI, make sure you're authenticated:
   ```bash
   gcloud auth application-default login
   ```
   
   Get your Qdrant Cloud credentials from [https://cloud.qdrant.io](https://cloud.qdrant.io)
   
   **For Local Qdrant** (Alternative):
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   QDRANT_URL=http://localhost:6333
   # QDRANT_API_KEY not needed for local
   ```
   
   Then start Qdrant locally:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

## Usage

### PDF Querying (Main Feature)

**Option 1: Using convenience functions (Recommended)**

```python
from agentic_arg import ingest_pdf, query_pdf

# Step 1: Ingest a PDF
pdf_path = "/path/to/your/document.pdf"
result = ingest_pdf(pdf_path)
print(result)  # Shows ingestion status

# Step 2: Query the PDF
result = query_pdf("What are the key points?", pdf_path=pdf_path)
print(result)

# Or query all ingested PDFs
result = query_pdf("What does the document say about machine learning?")
print(result)
```

**Option 2: Using the agent**

```python
from agentic_arg import invoke_agent

# Step 1: Ingest a PDF
pdf_path = "/path/to/your/document.pdf"
result = invoke_agent(
    f"Ingest this PDF: {pdf_path}",
    thread_id="user1"
)
print(result)  # Shows ingestion status

# Step 2: Query the PDF
result = invoke_agent(
    f"What are the key points in {pdf_path}?",
    thread_id="user1"
)
print(result)
```

### Basic Usage

```python
from agentic_arg import invoke_agent, resume_agent

# Simple query
result = invoke_agent("What is Python?", thread_id="user1")
print(result)

# Query with URL (may trigger HIL)
result = invoke_agent(
    "What is in this link: https://www.python.org/about/",
    thread_id="user1"
)

# If HIL interrupts, resume with approval
if isinstance(result, Command):
    result = resume_agent(
        [{"type": "approve"}],
        thread_id="user1"
    )
```

### Advanced Usage

```python
from agentic_arg import create_agentic_rag_agent
from langgraph.types import Command

# Create custom agent
agent = create_agentic_rag_agent(
    model_name="gemini-2.0-flash-exp",
    temperature=0.7,
    enable_hil=True
)

# Use agent
config = {"configurable": {"thread_id": "session1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello!"}]},
    config=config
)
```

## Configuration

All configuration is managed through environment variables. See `.env.example` for available options:

- `USE_VERTEX_AI`: Set to `true` to use Vertex AI, otherwise uses Gemini API (default: `false`)
- `GOOGLE_API_KEY`: Required for Gemini API (not needed for Vertex AI if using gcloud auth)
- `GEMINI_MODEL`: Model name (default: `gemini-2.5-flash`)
- `TEMPERATURE`: Model temperature (default: `0`)
- `MAX_TOKENS`: Maximum output tokens (default: `8192`, set to empty for no limit)
- `MAX_RETRIES`: Maximum retries (default: `6` for Vertex AI, `2` for Gemini API)
- `QDRANT_URL`: 
  - **For Qdrant Cloud**: Your cluster URL (e.g., `https://xxxxx-xxxxx-xxxxx.qdrant.io`)
  - **For Local**: `http://localhost:6333` (default)
- `QDRANT_API_KEY`: **Required for Qdrant Cloud** - Get from [https://cloud.qdrant.io](https://cloud.qdrant.io)
- `COLLECTION_NAME`: Qdrant collection name (default: `documents`)
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)

## Tools

### 1. `ingest_pdf_tool`
Ingest PDF files into the knowledge base. Extracts text, chunks it, generates embeddings, and stores in Qdrant.

**Usage**: `ingest_pdf_tool(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50)`

### 2. `query_pdf_tool`
Query content from ingested PDFs using hybrid search. Can search all PDFs or a specific one.

**Usage**: `query_pdf_tool(query: str, pdf_path: Optional[str] = None)`

### 3. `google_search_tool`
Web search using Google Custom Search API or DuckDuckGo fallback.

### 4. `fetch_url_tool`
Fetch and extract text content from URLs. **Requires HIL approval** by default.

### 5. `rag_search_tool`
Hybrid search in the knowledge base (excluding PDFs) combining:
- Dense vector search (semantic similarity)
- Keyword matching (BM25-style)

## Adding Documents to Knowledge Base

### PDF Ingestion (Recommended)

The easiest way is to use the `ingest_pdf_tool`:

```python
from agentic_arg import invoke_agent

# Ingest a PDF through the agent
result = invoke_agent(
    "Ingest this PDF: /path/to/document.pdf",
    thread_id="ingestion"
)
print(result)
```

Or use the tool directly:

```python
from tools.pdf_tool import ingest_pdf_tool

result = ingest_pdf_tool.invoke({
    "pdf_path": "/path/to/document.pdf",
    "chunk_size": 500,
    "chunk_overlap": 50
})
print(result)
```

### Manual Document Addition

To add custom documents to the Qdrant knowledge base (works with both Cloud and Local):

1. Generate embeddings for your documents
2. Store them in Qdrant with appropriate metadata

Example:
```python
from tools.rag_tool import _get_embedding_model, _get_qdrant_client
from qdrant_client.models import PointStruct

model = _get_embedding_model()
client = _get_qdrant_client()

# Embed your document
text = "Your document text here"
vector = model.encode(text).tolist()

# Store in Qdrant (works with both cloud and local)
client.upsert(
    collection_name="documents",  # or Config.COLLECTION_NAME
    points=[PointStruct(
        id=1,
        vector=vector,
        payload={"text": text, "source": "example.txt", "type": "text"}
    )]
)
```

**Note**: Make sure your `.env` file is properly configured with `QDRANT_URL` and `QDRANT_API_KEY` (for cloud) before running this code.

## Architecture

- `agentic_arg.py`: Main agent implementation with convenience functions (`ingest_pdf`, `query_pdf`)
- `config.py`: Configuration management
- `tools/`: Tool implementations
  - `pdf_tool.py`: PDF ingestion and querying
  - `search_tool.py`: Web search and URL fetching
  - `rag_tool.py`: Hybrid RAG search
- `memory/`: Memory utilities

## Notes

- HumanInTheLoopMiddleware may not be available in all LangChain versions. The agent will gracefully degrade if it's not available.
- Qdrant collection is automatically created if it doesn't exist.
- Embedding model is lazily loaded on first use.

