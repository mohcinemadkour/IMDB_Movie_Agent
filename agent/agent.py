"""
agent/agent.py
--------------
Builds and runs the LangChain conversational agent.

Uses langchain.agents.create_agent (LangChain 1.x / LangGraph-based).

LLM selection is controlled by environment variables:
  LLM_PROVIDER  = "openai" (default) | "gemini"
  OPENAI_MODEL  = "gpt-4o" (default)
  GEMINI_MODEL  = "gemini-1.5-pro" (default)
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from agent.prompts import SYSTEM_PROMPT
from agent.tools import get_tools

# Maximum LangGraph recursion steps (each tool call + LLM call = 2 steps).
# Prevents runaway agents from generating unbounded API spend.
MAX_AGENT_ITERATIONS = 10


def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # Default: OpenAI
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_agent_executor():
    """Instantiate the LLM, bind tools, and return a compiled agent graph."""
    llm = _build_llm()
    tools = get_tools()
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def format_chat_history(messages: list[dict]) -> list:
    """Convert Streamlit-style message dicts to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history


def run_agent(
    executor,
    user_input: str,
    chat_history: list[dict],
) -> tuple[str, dict]:
    """Invoke the agent and return (response_text, usage_dict).

    usage_dict keys: prompt_tokens, completion_tokens, total_tokens (ints).
    Values are 0 when the provider does not return usage metadata.
    """
    history = format_chat_history(chat_history)
    messages = history + [HumanMessage(content=user_input)]
    result = executor.invoke(
        {"messages": messages},
        config={"recursion_limit": MAX_AGENT_ITERATIONS},
    )
    last_msg = result["messages"][-1]
    response_text = last_msg.content

    # OpenAI responses carry token counts in response_metadata.
    usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    meta = getattr(last_msg, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {})
    if token_usage:
        usage = {
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens": token_usage.get("total_tokens", 0),
        }

    return response_text, usage
