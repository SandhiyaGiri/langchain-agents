"""Main agentic RAG agent implementation."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from tools import (
    google_search_tool, 
    fetch_url_tool, 
    rag_search_tool,
    ingest_pdf_tool,
    query_pdf_tool
)


# System prompt for agentic decision-making
SYSTEM_PROMPT = """You are an intelligent assistant with access to multiple information sources:
1. PDF Querying: Use query_pdf_tool when the user asks about content from a specific PDF or provides a PDF path
2. PDF Ingestion: Use ingest_pdf_tool when the user provides a PDF path that needs to be processed and added to the knowledge base
3. Knowledge Base (RAG): Use rag_search_tool when you need domain-specific knowledge or information from the knowledge base
4. Web Search: Use google_search_tool for current events, news, or general web information
5. URL Fetching: Use fetch_url_tool when the user provides a specific URL or link to read

Decision guidelines:
- If user provides a PDF path and asks to ingest it: Use ingest_pdf_tool first, then use query_pdf_tool to answer questions
- If user asks about a PDF or provides a PDF path with a query: Use query_pdf_tool (it will search ingested PDFs)
- Use RAG (rag_search_tool) when the query requires domain knowledge, specific documentation, or internal information (excluding PDFs)
- Use web search (google_search_tool) for current events, recent news, or information that changes frequently
- Use URL fetching (fetch_url_tool) when the user explicitly provides a URL or asks about content from a specific link
- Answer directly for simple questions, greetings, or when you have sufficient knowledge
- You can use multiple tools in sequence if needed (e.g., ingest PDF, then query it)

Always provide accurate, helpful, and well-structured responses. When using tools, quote relevant snippets and cite sources (including PDF names and page numbers) when possible."""


def create_agentic_rag_agent(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    enable_hil: bool = True
) -> Any:
    """
    Create the agentic RAG agent with all tools and middleware.
    
    Args:
        model_name: Gemini model name (defaults to Config.GEMINI_MODEL)
        temperature: Model temperature (defaults to Config.TEMPERATURE)
        enable_hil: Whether to enable Human-in-the-Loop middleware
    
    Returns:
        Configured agent with tools, memory, and middleware
    """
    # Validate configuration
    Config.validate()
    
    # Initialize LLM - support both Vertex AI and Gemini API
    if Config.USE_VERTEX_AI:
        try:
            from langchain_google_vertexai import ChatVertexAI
            
            llm = ChatVertexAI(
                model=model_name or Config.GEMINI_MODEL,
                temperature=temperature if temperature is not None else Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                max_retries=Config.get_max_retries(),
            )
        except ImportError:
            raise ImportError(
                "langchain-google-vertexai is required when USE_VERTEX_AI=true. "
                "Install it with: pip install langchain-google-vertexai"
            )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model=model_name or Config.GEMINI_MODEL,
            temperature=temperature if temperature is not None else Config.TEMPERATURE,
            max_output_tokens=Config.MAX_TOKENS,
            max_retries=Config.get_max_retries(),
            google_api_key=Config.GOOGLE_API_KEY,
        )
    
    # Prepare tools
    tools = [
        ingest_pdf_tool,
        query_pdf_tool,
        google_search_tool, 
        fetch_url_tool, 
        rag_search_tool
    ]
    
    # Bind tools to the model (required for tool calling)
    llm_with_tools = llm.bind_tools(tools)
    
    # Setup interrupts for HIL if enabled
    interrupt_before = None
    if enable_hil:
        # For LangGraph, we use interrupt_before to pause before tool execution
        # We can specify which tools need approval
        # Note: LangGraph doesn't support tool-specific interrupts directly,
        # so we'll interrupt before all tool calls if HIL is enabled
        interrupt_before = ["tools"]  # Interrupt before executing tools
    
    # Create agent using LangGraph's create_react_agent
    agent = create_react_agent(
        model=llm_with_tools,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
        interrupt_before=interrupt_before,
    )
    
    return agent


# Default agent instance (lazy initialization)
_agent = None

def get_agent():
    """Get or create the default agent instance."""
    global _agent
    if _agent is None:
        _agent = create_agentic_rag_agent()
    return _agent


def invoke_agent(
    message: str,
    thread_id: str = "default",
    config: Optional[Dict[str, Any]] = None,
    agent_instance: Optional[Any] = None
) -> Any:
    """
    Invoke the agent with a message.
    
    Args:
        message: User message
        thread_id: Thread ID for conversation persistence
        config: Optional additional configuration
        agent_instance: Optional agent instance (uses default if None)
    
    Returns:
        Agent response or interrupt state if HIL interrupt occurred
    """
    if agent_instance is None:
        agent_instance = get_agent()
    
    if config is None:
        config = {"configurable": {"thread_id": thread_id}}
    else:
        config["configurable"] = config.get("configurable", {})
        config["configurable"]["thread_id"] = thread_id
    
    # LangGraph expects messages in a specific format
    from langchain_core.messages import HumanMessage
    
    result = agent_instance.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    return result


def resume_agent(
    decisions: list = None,
    thread_id: str = "default",
    config: Optional[Dict[str, Any]] = None,
    agent_instance: Optional[Any] = None
) -> Any:
    """
    Resume agent after HIL interrupt.
    
    For LangGraph, interrupting returns a state that can be resumed by calling invoke again.
    The agent will automatically continue from where it paused.
    
    Args:
        decisions: Not used for LangGraph (agent auto-continues), kept for API compatibility
        thread_id: Thread ID for conversation persistence
        config: Optional additional configuration
        agent_instance: Optional agent instance (uses default if None)
    
    Returns:
        Agent response
    """
    if agent_instance is None:
        agent_instance = get_agent()
    
    if config is None:
        config = {"configurable": {"thread_id": thread_id}}
    else:
        config["configurable"] = config.get("configurable", {})
        config["configurable"]["thread_id"] = thread_id
    
    # For LangGraph, when interrupted, invoke returns a state with '__interrupt__' key
    # To resume, we pass Command(resume="value") where value is what to pass to the interrupt
    # If decisions is provided, use it; otherwise default to "continue"
    if decisions and len(decisions) > 0:
        # Extract the decision type, defaulting to "approve" if available
        resume_value = str(decisions[0].get("type", "approve"))
    else:
        resume_value = "continue"
    
    # Use Command to resume from interrupt
    result = agent_instance.invoke(
        Command(resume=resume_value),
        config=config
    )
    
    return result


def ingest_pdf(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> str:
    """
    Convenience function to ingest a PDF directly without going through the agent.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of text chunks in characters
        chunk_overlap: Overlap between chunks in characters
    
    Returns:
        Status message
    """
    return ingest_pdf_tool.invoke({
        "pdf_path": pdf_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    })


def query_pdf(query: str, pdf_path: Optional[str] = None) -> str:
    """
    Convenience function to query PDF content directly without going through the agent.
    
    Args:
        query: The search query
        pdf_path: Optional specific PDF path to search
    
    Returns:
        Relevant excerpts from PDF(s)
    """
    return query_pdf_tool.invoke({
        "query": query,
        "pdf_path": pdf_path
    })


# Example usage
if __name__ == "__main__":
    # Example 1: Ingest and query a PDF
    print("Example 1: PDF ingestion and querying")
    pdf_path = "/path/to/your/document.pdf"  # Replace with actual PDF path
    
    # First, ingest the PDF
    print(f"Ingesting PDF: {pdf_path}")
    result = invoke_agent(
        f"Ingest this PDF: {pdf_path}",
        thread_id="pdf_example"
    )
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Then query it
    print("Querying the PDF")
    result = invoke_agent(
        f"What is the main topic discussed in {pdf_path}?",
        thread_id="pdf_example"
    )
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Simple query
    print("Example 2: Simple query")
    result = invoke_agent("What is Python?", thread_id="example2")
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Query with URL (will trigger HIL)
    print("Example 3: Query with URL (HIL interrupt)")
    result = invoke_agent(
        "What is in this link: https://www.python.org/about/",
        thread_id="example3"
    )
    
    # Check if HIL interrupted
    if isinstance(result, Command):
        print("HIL interrupt occurred. Resuming with approval...")
        result = resume_agent(
            [{"type": "approve"}],
            thread_id="example3"
        )
    
    print(result)

