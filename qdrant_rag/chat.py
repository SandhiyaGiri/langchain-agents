"""Interactive chat interface for the agentic RAG agent."""
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

try:
    from InquirerPy import inquirer
except ImportError:  # pragma: no cover - optional dependency
    inquirer = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from agentic_arg import get_agent, invoke_agent, resume_agent


def format_message(msg) -> str:
    """Format a message for display."""
    if isinstance(msg, str):
        return msg
    if hasattr(msg, 'content'):
        return msg.content
    return str(msg)


def extract_tool_calls(messages):
    """Extract tool calls from messages."""
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_calls.append({
                    'name': tool_call.get('name', 'unknown'),
                    'args': tool_call.get('args', {}),
                    'id': tool_call.get('id', '')
                })
    return tool_calls


def _get_recent_tool_activity(messages):
    """Return the most recent tool call and its tool messages (if any)."""
    recent_tool_messages = []
    recent_tool_call = None

    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            recent_tool_messages.append(msg)
            continue
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            recent_tool_call = msg
            break
        if isinstance(msg, HumanMessage):
            # We've reached the user again without finding a tool call
            break

    if recent_tool_call is None:
        return None, []

    recent_tool_messages.reverse()
    return recent_tool_call, recent_tool_messages


def print_latest_tool_activity(messages):
    """Print only the latest tool call and its corresponding results."""
    tool_call_msg, tool_messages = _get_recent_tool_activity(messages)

    if tool_call_msg:
        print("\n" + "=" * 70)
        print("🔧 TOOL CALLS DETECTED:")
        print("=" * 70)
        for tc in tool_call_msg.tool_calls:
            print(f"\n  Tool Name: {tc.get('name', 'unknown')}")
            args = tc.get('args', {})
            if args:
                print("      Arguments:")
                for key, value in args.items():
                    val_str = str(value)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"        • {key}: {val_str}")
        print("=" * 70 + "\n")

    if tool_messages:
        print("\n" + "=" * 70)
        print("📊 TOOL RESULTS:")
        print("=" * 70)
        for msg in tool_messages:
            content = format_message(msg)
            if len(content) > 300:
                content = content[:300] + "..."
            print(f"\n  Tool: {getattr(msg, 'name', 'unknown')}")
            print(f"      Result: {content}")
        print("=" * 70 + "\n")


def prompt_yes_no(message: str, default: str = "yes") -> str:
    """Prompt the user with a yes/no question."""
    if inquirer is not None:
        return inquirer.select(
            message=message,
            choices=["yes", "no"],
            default=default
        ).execute().lower()

    valid = {"yes", "y", "no", "n"}
    while True:
        choice = input(f"{message} [yes/no]: ").strip().lower()
        if choice in valid:
            return "yes" if choice in {"yes", "y"} else "no"
        print("Please respond with 'yes' or 'no'.")


def print_messages(result: Dict[str, Any], show_all: bool = False):
    """Print messages from agent result."""
    if not isinstance(result, dict):
        print(f"Result: {result}")
        return
    
    messages = result.get("messages", [])
    if not messages:
        print("No messages in response")
        return
    
    # Show agent info
    from agentic_arg import Config
    agent_type = "Vertex AI" if Config.USE_VERTEX_AI else "Gemini API"
    model_name = Config.GEMINI_MODEL
    print(f"\n🤖 Agent: {agent_type} ({model_name})")
    print("-" * 70)
    
    if show_all:
        print("\n📝 CONVERSATION HISTORY:")
        print("="*70)
        for i, msg in enumerate(messages, 1):
            if isinstance(msg, HumanMessage):
                role = "👤 USER"
                icon = ""
            elif isinstance(msg, AIMessage):
                role = "🤖 AGENT"
                icon = ""
            elif isinstance(msg, ToolMessage):
                role = "🔧 TOOL"
                icon = ""
            else:
                role = "📄 SYSTEM"
                icon = ""
            
            content = format_message(msg)
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "\n    ... (truncated)"
            
            print(f"\n[{i}] {role}:")
            # Indent content
            for line in content.split('\n'):
                print(f"    {line}")
        print("="*70 + "\n")
    else:
        # Show only the last message
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            content = format_message(last_msg)
            print(f"\n💬 Response:")
            # Indent content
            for line in content.split('\n'):
                print(f"    {line}")
        else:
            content = format_message(last_msg)
            print(f"\n📄 {content}")
    
    # Show the latest tool activity (if any)
    print_latest_tool_activity(messages)


def chat_interactive():
    """Interactive chat interface with the agent."""
    from agentic_arg import Config
    
    agent_type = "Vertex AI" if Config.USE_VERTEX_AI else "Gemini API"
    model_name = Config.GEMINI_MODEL
    
    print("="*70)
    print("🤖 Agentic RAG Agent - Interactive Chat")
    print("="*70)
    print(f"\nAgent Info:")
    print(f"  • Type: {agent_type}")
    print(f"  • Model: {model_name}")
    print(f"  • All tool calls require your approval")
    print("\nCommands:")
    print("  - Type your message and press Enter to chat")
    print("  - Type 'exit' or 'quit' to end the chat")
    print("  - Type 'show' to see all messages in current thread")
    print("  - Type 'new' to start a new conversation thread")
    print("="*70 + "\n")
    
    thread_id = "chat_session_1"
    agent = get_agent()
    
    # Start conversation with greeting
    print("Starting conversation...\n")
    result = invoke_agent(
        message=None,
        thread_id=thread_id,
        agent_instance=agent,
        greet=True
    )
    
    print_messages(result)
    
    pending_interrupt = None
    
    while True:
        try:
            if pending_interrupt:
                messages = pending_interrupt.get("messages", [])
                tool_calls = extract_tool_calls(messages)

                print("\n" + "="*70)
                print("⚠️  TOOL APPROVAL REQUIRED")
                print("="*70)
                if tool_calls:
                    print(f"\nThe agent wants to use {len(tool_calls)} tool(s):")
                    for i, tc in enumerate(tool_calls, 1):
                        print(f"  [{i}] Tool: {tc['name']}")
                        args = tc.get('args', {})
                        if 'query' in args:
                            print(f"       Query: {args['query']}")
                        if 'pdf_path' in args:
                            print(f"       PDF: {args['pdf_path']}")
                        if 'url' in args:
                            print(f"       URL: {args['url']}")
                decision = prompt_yes_no("Allow tool execution?", default="yes")

                if decision == "yes":
                    print("\n✅ Approving tool calls...\n")
                    result = resume_agent(
                        decisions=[{"type": "approve"}],
                        thread_id=thread_id,
                        agent_instance=agent
                    )
                    print_messages(result)
                    if isinstance(result, dict) and "__interrupt__" in result:
                        pending_interrupt = result
                    else:
                        pending_interrupt = None
                else:
                    print("\n❌ Rejecting tool calls...\n")
                    result = resume_agent(
                        decisions=[{"type": "reject"}],
                        thread_id=thread_id,
                        agent_instance=agent
                    )
                    print_messages(result)
                    pending_interrupt = None
                continue

            user_input = input("User: ").strip()

            if not user_input:
                continue

            lowered = user_input.lower()

            if lowered in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            if lowered == 'show':
                from langgraph.checkpoint.base import BaseCheckpointSaver
                config = {"configurable": {"thread_id": thread_id}}
                state = agent.get_state(config)
                if state and state.values:
                    print_messages(state.values, show_all=True)
                continue

            if lowered == 'new':
                import uuid
                thread_id = f"chat_session_{uuid.uuid4().hex[:8]}"
                print(f"\n🆕 New conversation thread: {thread_id}\n")
                result = invoke_agent(
                    message=None,
                    thread_id=thread_id,
                    agent_instance=agent,
                    greet=True
                )
                print_messages(result)
                pending_interrupt = None
                continue

            print("\n🔄 Processing...\n")
            result = invoke_agent(
                message=user_input,
                thread_id=thread_id,
                agent_instance=agent
            )

            if isinstance(result, dict) and "__interrupt__" in result:
                messages = result.get("messages", [])
                tool_calls = extract_tool_calls(messages)

                if tool_calls:
                    print_messages(result, show_all=False)
                    pending_interrupt = result
                else:
                    print_messages(result)
                    pending_interrupt = None
            else:
                print_messages(result)
                pending_interrupt = None
                
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            pending_interrupt = None


if __name__ == "__main__":
    chat_interactive()

