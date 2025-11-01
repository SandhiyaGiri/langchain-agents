"""Main agentic RAG agent implementation."""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
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


def _extract_messages_from_chunk(chunk: Dict[str, Any]) -> Optional[List[Any]]:
    """Extract messages list from a stream chunk."""
    if "messages" in chunk:
        return chunk["messages"]
    for value in chunk.values():
        if isinstance(value, dict) and "messages" in value:
            return value["messages"]
    return None


# System prompt for simple agent behavior
SYSTEM_PROMPT = """You are a helpful assistant. You should:

1. **Greet the user** when starting a conversation
2. **Answer questions directly** when you have sufficient knowledge - avoid unnecessary tool calls
3. **Only use tools when absolutely necessary** and the user explicitly requests or it's clearly needed

**PDF Tools** - Only use when:
- User explicitly provides a PDF file path AND asks you to refer to it or query it
- Example: "Look at /path/to/file.pdf and tell me..." or "Ingest this PDF: /path/to/file.pdf"
- Do NOT use PDF tools for general queries unless explicitly requested

**Other Tools** - Only use when:
- User explicitly asks you to search the web or fetch a URL
- You cannot answer the question from your knowledge

**Important**: Before using ANY tool, you must ask for the user's permission. All tool calls require approval.

Keep your responses concise, helpful, and natural. Avoid using tools unless truly necessary."""


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
    
    # Setup interrupts for HIL - ALL tools require approval
    interrupt_before = ["tools"]  # Always interrupt before executing ANY tool
    
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
    message: Optional[str] = None,
    thread_id: str = "default",
    config: Optional[Dict[str, Any]] = None,
    agent_instance: Optional[Any] = None,
    greet: bool = True
) -> Any:
    """
    Invoke the agent with a message.
    
    Args:
        message: User message (optional, if None and greet=True, will just greet)
        thread_id: Thread ID for conversation persistence
        config: Optional additional configuration
        agent_instance: Optional agent instance (uses default if None)
        greet: If True and this is a new conversation, agent will greet the user
    
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
    
    # Build messages - agent will greet naturally based on system prompt
    messages = []
    
    # If no message provided and greet=True, send minimal message to trigger greeting
    if not message or (greet and not message.strip()):
        # Send a simple message that will trigger the agent to greet
        # The system prompt instructs the agent to greet, so it will respond with a greeting
        messages.append(HumanMessage(content="Hi"))
    elif message:
        messages.append(HumanMessage(content=message))
    
    last_state: Dict[str, Any] | None = None
    interrupt_payload = None
    for chunk in agent_instance.stream(
        {"messages": messages},
        config=config
    ):
        extracted = _extract_messages_from_chunk(chunk)
        if extracted is not None:
            last_state = {"messages": extracted}
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"]
    
    if last_state is None:
        last_state = {}
    else:
        last_state = dict(last_state)
    state_snapshot = agent_instance.get_state(config)
    if state_snapshot and state_snapshot.values:
        last_state["messages"] = state_snapshot.values.get("messages", [])
    if interrupt_payload is not None:
        last_state["__interrupt__"] = interrupt_payload
    
    return last_state


def resume_agent(
    decisions: list = None,
    thread_id: str = "default",
    config: Optional[Dict[str, Any]] = None,
    agent_instance: Optional[Any] = None
) -> Any:
    """
    Resume agent after HIL interrupt.
    
    When approving: Resumes execution and tools are executed.
    When rejecting: Adds rejection ToolMessages so the LLM knows tools were rejected.
    
    Args:
        decisions: List with decision, e.g., [{"type": "approve"}] or [{"type": "reject"}]
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
    
    # Get current state to check for tool calls
    state = agent_instance.get_state(config)
    
    # Accept common approval / rejection variants
    approval_decisions = {
        "approve",
        "approved",
        "accept",
        "accepted",
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "yep",
        "yup",
    }
    rejection_decisions = {
        "reject",
        "rejected",
        "no",
        "n",
        "deny",
        "denied",
        "decline",
        "declined",
        "cancel",
        "cancelled",
        "c",
        "nope",
    }

    # Extract decision type (default to approve/continue)
    decision_type = "approve"
    if decisions and len(decisions) > 0:
        decision_type = str(decisions[0].get("type", "approve")).lower()

    # If rejecting, we need to add ToolMessages for rejected tool calls
    # This completes the tool call chain so the LLM can process the rejection
    if decision_type in rejection_decisions:
        from langchain_core.messages import ToolMessage, AIMessage
        
        if state and state.values:
            messages = state.values.get("messages", [])
            
            # Find the last AIMessage with tool_calls
            last_ai_msg = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    last_ai_msg = msg
                    break
            
            if last_ai_msg and last_ai_msg.tool_calls:
                # Create rejection ToolMessages for each tool call
                # This tells the LLM that the tools were rejected
                rejection_messages = []
                for tool_call in last_ai_msg.tool_calls:
                    tool_call_id = tool_call.get("id", "")
                    tool_name = tool_call.get("name", "unknown")
                    tool_msg = ToolMessage(
                        content="This tool call was rejected by the user. Please inform the user that you cannot proceed without tool execution.",
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )
                    rejection_messages.append(tool_msg)
                
                # Add rejection messages to continue the conversation
                # The LLM will receive these and can respond appropriately
                last_state = None
                interrupt_payload = None
                for chunk in agent_instance.stream(
                    {"messages": rejection_messages},
                    config=config
                ):
                    extracted = _extract_messages_from_chunk(chunk)
                    if extracted is not None:
                        last_state = {"messages": extracted}
                    if "__interrupt__" in chunk:
                        interrupt_payload = chunk["__interrupt__"]
                if last_state is None:
                    last_state = {}
                else:
                    last_state = dict(last_state)
                state_snapshot = agent_instance.get_state(config)
                if state_snapshot and state_snapshot.values:
                    last_state["messages"] = state_snapshot.values.get("messages", [])
                if interrupt_payload is not None:
                    last_state["__interrupt__"] = interrupt_payload
                return last_state
    
    # For approval, resume execution so the tools can run
    resume_value = "continue"
    if decision_type in approval_decisions:
        resume_value = "approve"
    else:
        resume_value = "continue"
    try:
        stream_iter = agent_instance.stream(
            Command(resume=resume_value),
            config=config
        )
    except Exception:
        stream_iter = agent_instance.stream(
            {},
            config=config
        )

    last_state = None
    interrupt_payload = None
    for chunk in stream_iter:
        extracted = _extract_messages_from_chunk(chunk)
        if extracted is not None:
            last_state = {"messages": extracted}
        if "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"]

    if last_state is None:
        last_state = {}
    else:
        last_state = dict(last_state)
    state_snapshot = agent_instance.get_state(config)
    if state_snapshot and state_snapshot.values:
        last_state["messages"] = state_snapshot.values.get("messages", [])
    if interrupt_payload is not None:
        last_state["__interrupt__"] = interrupt_payload

    return last_state


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
    # Example 1: Start conversation (agent greets)
    print("Example 1: Starting conversation (agent greets)")
    result = invoke_agent(thread_id="example1", greet=True)
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            print("Agent:", messages[-1].content)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Simple query (no tools needed)
    print("Example 2: Simple query (no tools)")
    result = invoke_agent("What is Python?", thread_id="example2")
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            print("Agent:", messages[-1].content)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Query with PDF (will trigger HIL for tool approval)
    print("Example 3: Query with PDF (requires approval)")
    pdf_path = "/Users/sandhiya.cv/Downloads/multi-agent-architecture/Aleena_Joseph.pdf"
    result = invoke_agent(
        f"Please look at this PDF: {pdf_path} and tell me what it contains",
        thread_id="example3"
    )
    
    # Check if HIL interrupted (agent will pause before using tools)
    if isinstance(result, dict) and "__interrupt__" in result:
        print("\n[INTERRUPT] Tool approval needed before accessing PDF.")
        print("Resuming with approval...")
        result = resume_agent(
            [{"type": "approve"}],
            thread_id="example3"
        )
        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                print("Agent:", messages[-1].content)
    elif isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            print("Agent:", messages[-1].content)

