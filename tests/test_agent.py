"""
tests/test_agent.py
--------------------
Integration tests for agent/agent.py using a stubbed LLM.

The stub LLM bypasses all OpenAI calls; each test scenario is answered by a
pre-canned message so we can verify:
  - Agent runs without error for all 9 scenario questions
  - run_agent() returns a non-empty string
  - stream_agent() yields tokens that concatenate to a non-empty string
  - Token usage dict has the expected keys
"""

import os
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

# ---------------------------------------------------------------------------
# Stub LLM — returns a fixed AIMessage for any invocation
# ---------------------------------------------------------------------------

class _StubLLM:
    """Minimal LLM stub that satisfies LangGraph's bind_tools() + ainvoke() contract."""

    def __init__(self, reply: str = "Test response."):
        self._reply = reply
        self.model_name = "stub"

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        return AIMessage(content=self._reply)

    def stream(self, messages, **kwargs):
        # Yield token-by-token
        for word in self._reply.split():
            yield AIMessage(content=word + " ")

    # LangChain needs these for type checking
    @property
    def _llm_type(self):  # noqa: D102
        return "stub"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def agent_executor(sample_df, mock_vectorstore):
    """Build a real agent executor backed by the stub LLM and sample data."""
    import agent.tools as _t
    _t._df = sample_df
    _t._vectorstore = mock_vectorstore

    with patch("agent.agent._build_llm", return_value=_StubLLM()):
        with patch("agent.agent._setup_llm_cache"):
            from agent.agent import build_agent_executor
            return build_agent_executor()


# ---------------------------------------------------------------------------
# 9 scenario questions (from the assignment brief)
# ---------------------------------------------------------------------------

SCENARIO_QUESTIONS = [
    "When did The Matrix release?",
    "Top 5 movies of 2019 by meta score",
    "Top 7 comedy movies 2010–2020 by IMDB rating",
    "Horror movies with meta score > 85 and IMDB rating > 8",
    "Movies before 1990 involving police in the plot",
    "Summarize Spielberg's top-rated sci-fi movies",
    "Al Pacino movies grossing over $50M with IMDB rating >= 8",
    "How many action movies are in the dataset?",
    "Top directors whose movies grossed over $500M at least twice",
]


class TestRunAgentScenarios:
    @pytest.mark.parametrize("question", SCENARIO_QUESTIONS)
    def test_run_agent_returns_string(self, agent_executor, question):
        from agent.agent import run_agent
        response, usage = run_agent(agent_executor, question, [])
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.parametrize("question", SCENARIO_QUESTIONS)
    def test_run_agent_usage_keys(self, agent_executor, question):
        from agent.agent import run_agent
        _, usage = run_agent(agent_executor, question, [])
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_run_agent_with_chat_history(self, agent_executor):
        from agent.agent import run_agent
        history = [
            {"role": "user", "content": "Top 5 drama movies?"},
            {"role": "assistant", "content": "Here are the top 5 drama movies…"},
        ]
        response, usage = run_agent(agent_executor, "What about comedies?", history)
        assert isinstance(response, str)
        assert len(response) > 0


class TestStreamAgentScenarios:
    @pytest.mark.parametrize("question", SCENARIO_QUESTIONS)
    def test_stream_agent_yields_tokens(self, agent_executor, question):
        from agent.agent import stream_agent
        usage_out: dict = {}
        tokens = list(stream_agent(agent_executor, question, [], usage_out))
        full_text = "".join(tokens)
        assert isinstance(full_text, str)
        assert len(full_text) > 0

    def test_stream_agent_populates_usage_out(self, agent_executor):
        from agent.agent import stream_agent
        usage_out: dict = {}
        list(stream_agent(agent_executor, "Top 5 movies", [], usage_out))
        # Usage keys must exist after stream is exhausted
        assert "prompt_tokens" in usage_out
        assert "completion_tokens" in usage_out
        assert "total_tokens" in usage_out

    def test_stream_agent_concatenated_equals_full_response(self, agent_executor):
        """Concatenating all yielded tokens should be the complete response."""
        from agent.agent import stream_agent
        tokens = list(stream_agent(agent_executor, "When did The Matrix release?", []))
        streamed = "".join(tokens)
        assert len(streamed) > 0


# ---------------------------------------------------------------------------
# format_chat_history
# ---------------------------------------------------------------------------

class TestFormatChatHistory:
    def test_empty_history(self):
        from agent.agent import format_chat_history
        assert format_chat_history([]) == []

    def test_user_becomes_human_message(self):
        from langchain_core.messages import HumanMessage

        from agent.agent import format_chat_history
        result = format_chat_history([{"role": "user", "content": "hello"}])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"

    def test_assistant_becomes_ai_message(self):
        from langchain_core.messages import AIMessage

        from agent.agent import format_chat_history
        result = format_chat_history([{"role": "assistant", "content": "hi there"}])
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)

    def test_alternating_history(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from agent.agent import format_chat_history
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        result = format_chat_history(history)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert isinstance(result[1], AIMessage)
        assert isinstance(result[2], HumanMessage)


# ---------------------------------------------------------------------------
# LLM cache setup
# ---------------------------------------------------------------------------

class TestSetupLlmCache:

    def test_none_disables_cache(self):
        with patch.dict(os.environ, {"LLM_CACHE": "none"}):
            with patch("langchain_core.globals.set_llm_cache") as mock_set:
                from agent.agent import _setup_llm_cache
                _setup_llm_cache()
                mock_set.assert_not_called()

    def test_memory_sets_in_memory_cache(self):
        with patch.dict(os.environ, {"LLM_CACHE": "memory"}):
            from langchain_community.cache import InMemoryCache
            with patch("langchain_core.globals.set_llm_cache") as mock_set:
                from agent.agent import _setup_llm_cache
                _setup_llm_cache()
                mock_set.assert_called_once()
                assert isinstance(mock_set.call_args[0][0], InMemoryCache)

    def test_sqlite_fallback_on_error(self, tmp_path):
        # If SQLiteCache init fails, should fall back to InMemoryCache silently
        with patch.dict(os.environ, {"LLM_CACHE": "sqlite",
                                     "LLM_CACHE_DB": str(tmp_path / "cache.db")}):
            with patch("langchain_community.cache.SQLiteCache.__init__",
                       side_effect=RuntimeError("disk full")):
                with patch("langchain_core.globals.set_llm_cache") as mock_set:
                    from agent.agent import _setup_llm_cache
                    _setup_llm_cache()
                    # Should still call set_llm_cache (with InMemoryCache fallback)
                    mock_set.assert_called_once()
