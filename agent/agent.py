"""
agent/agent.py
--------------
Builds and runs the LangChain conversational agent.

Uses langchain.agents.create_agent (LangChain 1.x / LangGraph-based).

LLM selection is controlled by environment variables:
  LLM_PROVIDER  = "openai" (default) | "gemini"
  OPENAI_MODEL  = "gpt-4o" (default)
  GEMINI_MODEL  = "gemini-1.5-pro" (default)

Caching:
  LLM_CACHE     = "sqlite" (default) | "memory" | "none"
  LLM_CACHE_DB  = path to the SQLite cache file (default: .cache/llm_cache.db)
                  Only used when LLM_CACHE=sqlite.
"""

import logging
import os
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from agent.prompts import SYSTEM_PROMPT
from agent.tools import get_tools

_LOG = logging.getLogger(__name__)

# Maximum LangGraph recursion steps (each tool call + LLM call = 2 steps).
# Prevents runaway agents from generating unbounded API spend.
MAX_AGENT_ITERATIONS = 10


def _setup_llm_cache() -> None:
    """Configure LangChain's global LLM response cache.

    Cache type is selected via LLM_CACHE env var:
      sqlite  (default) — persists across restarts; stored in LLM_CACHE_DB
      memory            — in-process only; cleared on restart
      none              — disables caching entirely

    SQLiteCache is effective for repeated identical queries (same prompt + model).
    It reduces latency and API spend for deterministic look-ups (exact-match title,
    "how many" counts, director aggregations, etc.).
    """
    from pathlib import Path

    cache_type = os.getenv("LLM_CACHE", "sqlite").lower()

    if cache_type == "none":
        _LOG.info("LLM response cache disabled.")
        return

    if cache_type == "memory":
        from langchain_community.cache import InMemoryCache
        from langchain_core.globals import set_llm_cache

        set_llm_cache(InMemoryCache())
        _LOG.info("LLM cache: InMemoryCache (in-process, cleared on restart)")
        return

    # Default: SQLiteCache
    try:
        from langchain_community.cache import SQLiteCache
        from langchain_core.globals import set_llm_cache

        db_path = Path(os.getenv("LLM_CACHE_DB", ".cache/llm_cache.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(db_path)))
        _LOG.info("LLM cache: SQLiteCache at %s", db_path)
    except Exception as exc:
        # Graceful fallback — caching failure must never break the agent
        _LOG.warning("SQLiteCache setup failed (%s); falling back to InMemoryCache.", exc)
        from langchain_community.cache import InMemoryCache
        from langchain_core.globals import set_llm_cache

        set_llm_cache(InMemoryCache())


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
    _setup_llm_cache()
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

    _LOG.info(
        "agent_run_start",
        extra={
            "query_length": len(user_input),
            "query_preview": user_input[:120],
            "history_turns": len(chat_history),
        },
    )
    t0 = time.perf_counter()

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

    _LOG.info(
        "agent_run_complete",
        extra={
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        },
    )

    return response_text, usage
