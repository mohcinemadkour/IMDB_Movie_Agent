git revert 88c943d 4f7d80b --no-edit
git push origin master# IMDB Movie Agent — Architecture

## Overview

A GenAI-powered conversational voice agent built with **Streamlit**. Users ask natural-language questions about the IMDB Top 1000 dataset and receive streamed answers from a **LangGraph** agent that routes each query to one or more specialised tools (structured pandas query, FAISS semantic search, or director aggregation). Voice input (Whisper STT) and voice output (OpenAI TTS / gTTS) are optional and toggleable per session.

---

## High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (User)                        │
│   chat input │ voice recorder │ sidebar settings            │
└────────────────────────────┬────────────────────────────────┘
                             │  HTTP (Streamlit WebSocket)
┌────────────────────────────▼────────────────────────────────┐
│                        app.py  (Streamlit)                   │
│                                                             │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ _speak()     │  │_transcribe_  │  │ st.write_stream │  │
│  │ OpenAI TTS   │  │ audio()      │  │ (streaming UI)  │  │
│  │ / gTTS       │  │ Whisper STT  │  └────────┬────────┘  │
│  └──────────────┘  └───────────────┘           │            │
└────────────────────────────────────────────────┼────────────┘
                                                 │
                             ┌───────────────────▼──────────────────────┐
                             │          agent/agent.py                   │
                             │                                           │
                             │  build_agent_executor()                   │
                             │    └─ _build_llm()    (OpenAI / Gemini)   │
                             │    └─ _setup_llm_cache()  (SQLite/memory) │
                             │    └─ create_agent()  (LangGraph)         │
                             │                                           │
                             │  stream_agent()  ─── yields tokens ──►   │
                             │  run_agent()     (testing / CLI)          │
                             └─────────────────┬─────────────────────────┘
                                               │  tool calls
                          ┌────────────────────┼──────────────────────┐
                          │                    │                       │
              ┌───────────▼──────┐  ┌──────────▼──────┐  ┌───────────▼──────────┐
              │ structured_query │  │ semantic_search  │  │ director_gross_      │
              │ (pandas filters) │  │ (FAISS + OpenAI  │  │ summary              │
              │                  │  │  embeddings)     │  │ (pandas aggregation) │
              └───────────┬──────┘  └──────────┬───────┘  └──────────────────────┘
                          │                    │
              ┌───────────▼──────┐  ┌──────────▼──────────────────────┐
              │  data/loader.py  │  │  data/vectorstore.py             │
              │  (lru_cache)     │  │  FAISS / Pinecone / ChromaDB     │
              └───────────┬──────┘  └──────────┬──────────────────────┘
                          │                    │
              ┌───────────▼────────────────────▼──────────────────────┐
              │           imdb_dataset/imdb_top_1000.csv               │
              │           data/faiss_index/  (committed)               │
              └───────────────────────────────────────────────────────┘
```

---

## Layers

### 1. Presentation Layer — `app.py`

| Responsibility | Detail |
|---|---|
| Streamlit page config | Wide layout, page title, icon |
| API key guard | Stops the app with a clear error if neither `OPENAI_API_KEY` nor `GOOGLE_API_KEY` is set |
| Shared resource loading | `@st.cache_resource` loads the DataFrame and FAISS index **once per worker process**, shared across all browser tabs |
| Session state | `_SESSION_DEFAULTS` + `_init_session_state()` seed `messages` and `agent_executor` per browser session |
| Streaming UI | `st.write_stream(stream_agent(...))` renders LLM tokens progressively as they arrive |
| Voice input | `audio_recorder` widget → `_transcribe_audio()` → Whisper API → `pending_input` → `st.rerun()` |
| Voice output | `_speak()` calls OpenAI TTS (`tts-1`) or gTTS fallback → `st.audio()` player |
| Sidebar | Voice toggle, TTS engine/voice selector, Clear Chat button, example question buttons |
| Input guard | Messages truncated to `MAX_INPUT_CHARS = 2000` before reaching the agent |

### 2. Agent Layer — `agent/`

#### `agent/agent.py`

| Function | Purpose |
|---|---|
| `_setup_llm_cache()` | Configures LangChain's global LLM cache (SQLite default, in-memory, or off) via `LLM_CACHE` env var |
| `_build_llm()` | Instantiates `ChatOpenAI` (default) or `ChatGoogleGenerativeAI` based on `LLM_PROVIDER` env var |
| `build_agent_executor()` | Composes LLM + tools + system prompt into a compiled LangGraph agent graph |
| `stream_agent()` | Sync generator — uses `stream_mode="messages"` to yield prose tokens, filters out tool-dispatch chunks |
| `run_agent()` | Blocking invocation for CLI / tests; returns `(response_text, usage_dict)` |
| `format_chat_history()` | Converts Streamlit `messages` dicts → `HumanMessage` / `AIMessage` objects |

The agent is built with **LangGraph's `create_agent()`** (ReAct-style) with a recursion limit of 10 steps (`MAX_AGENT_ITERATIONS`) to prevent runaway API spend.

#### `agent/prompts.py`

Defines `SYSTEM_PROMPT` — a detailed instruction set that controls:
- When to route to each tool
- How to handle count queries (`count_only: true`)
- How to handle "all / every" queries (set `limit: 500`)
- Response formatting (monetary values, numbered lists)
- Clarifying questions for ambiguous actor queries
- Mandatory 2–3 movie recommendations after every answer

#### `agent/tools.py`

Three LangChain `@tool` functions exposed to the agent:

| Tool | Input | When the agent uses it |
|---|---|---|
| `structured_query` | JSON string with optional filter keys | Numeric filters, sort, genre, director, actor, count queries |
| `semantic_search` | Natural-language plot description | Thematic / conceptual queries referencing story elements |
| `director_gross_summary` | `gross_threshold` float | "Directors with N+ blockbusters" aggregation |

Module-level constants keep magic numbers in one place:
- `_MAX_RESULTS = 500` — hard cap on rows returned; used in the tool doc string and the safety clamp
- `_DISPLAY_COLS` — ordered list of columns shown in structured results

### 3. Data Layer — `data/`

#### `data/loader.py`

Loads and cleans `imdb_top_1000.csv` once, cached with `@lru_cache(maxsize=1)`:

| Column | Cleaning applied |
|---|---|
| `Runtime` | Strip `" min"`, cast to `int` |
| `Gross` | Remove commas, cast to `float` |
| `No_of_Votes` | Remove commas, cast to `Int64` |
| `Released_Year` | Coerce to numeric, drop non-integer rows, cast to `int` |

#### `data/vectorstore.py`

Provider-agnostic vector store, selected via `VECTOR_STORE` env var:

| Backend | Default | Config |
|---|---|---|
| **FAISS** | ✅ yes | Local flat file at `data/faiss_index/` — committed to repo |
| **Pinecone** | — | `VECTOR_STORE=pinecone`, requires `PINECONE_API_KEY` |
| **ChromaDB** | — | `VECTOR_STORE=chroma`, local or remote (`CHROMA_HOST`) |

All backends embed the `Overview` (plot summary) column using OpenAI `text-embedding-3-small` (1536 dimensions) and expose `.similarity_search(query, k=10)`.

The FAISS index is pre-built and committed to the repo at `data/faiss_index/` so the app works on first run without re-embedding.

### 4. Infrastructure Layer

#### Logging — `logging_config.py`

- Structured JSON logs via `python-json-logger`
- Two handlers: stdout (always) + rotating file (`logs/app.log`, 10 MB × 5 files)
- Level controlled by `LOG_LEVEL` env var (default: `INFO`)
- Every tool call and agent invocation logs structured fields (query, latency_ms, rows_returned, etc.)

#### LLM Caching

| Setting | Behaviour |
|---|---|
| `LLM_CACHE=sqlite` (default) | Persists identical (prompt, model) pairs to `.cache/llm_cache.db` across restarts |
| `LLM_CACHE=memory` | In-process cache, cleared on restart (used on Render free tier) |
| `LLM_CACHE=none` | Disabled |

#### Containerisation — `Dockerfile` + `docker-compose.yml`

- **Multi-stage build**: builder stage installs wheels into `/install`; runtime stage is a minimal `python:3.11-slim` with no build tools
- Runs as a **non-root user** (`appuser`)
- Health check: polls `/_stcore/health` every 30 s
- Docker Compose mounts `./logs`, `./data/faiss_index`, `./data/faiss_index`, and `./.cache` as host volumes so data survives container restarts

#### CI/CD — `.github/workflows/ci.yml`

Triggered on every push and pull request to `master`:

```
ruff check agent/ data/ tests/ app.py
  └─ mypy agent/ data/
       └─ pytest tests/ -q   (114 tests)
```

---

## Data Flow — Single Query

```
User types: "Top 5 horror movies of 2020 with a dark plot"
       │
       ▼
app.py: truncate to 2000 chars → append to messages → st.write_stream()
       │
       ▼
stream_agent(): format_chat_history() → HumanMessage + history → executor.stream()
       │
       ▼
LangGraph ReAct loop:
  Step 1: LLM decides to call BOTH tools
  Step 2: structured_query({"genre":"Horror","year_min":2020,"year_max":2020,"limit":5})
            └─ pandas filter → 5 rows → formatted table string
  Step 3: semantic_search("dark disturbing horror plot")
            └─ FAISS similarity_search(k=10) → 10 docs with metadata
  Step 4: LLM synthesises both results → prose response
       │
       ▼  (token stream)
st.write_stream(): renders tokens progressively in the chat bubble
       │
       ▼
_speak(): OpenAI TTS → st.audio() player rendered below the response
```

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API access |
| `GOOGLE_API_KEY` | — | Required only when `LLM_PROVIDER=gemini` |
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_MODEL` | `gpt-4o` | Any OpenAI chat model |
| `GEMINI_MODEL` | `gemini-1.5-pro` | Any Gemini chat model |
| `LLM_CACHE` | `sqlite` | `sqlite`, `memory`, or `none` |
| `LLM_CACHE_DB` | `.cache/llm_cache.db` | Path to SQLite cache file |
| `VECTOR_STORE` | `faiss` | `faiss`, `pinecone`, or `chroma` |
| `PINECONE_API_KEY` | — | Required when `VECTOR_STORE=pinecone` |
| `PINECONE_INDEX_NAME` | `imdb-movies` | Pinecone index name |
| `CHROMA_HOST` | — | ChromaDB HTTP host (local persistent if unset) |
| `CHROMA_PERSIST_DIR` | `.chroma` | ChromaDB local persistence dir |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE` | `logs/app.log` | Set to `""` to disable file logging |
| `LANGCHAIN_TRACING_V2` | — | Set `true` to enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `LANGCHAIN_PROJECT` | — | LangSmith project name |

---

## File Structure

```
.
├── app.py                        # Streamlit entry point
├── logging_config.py             # Structured JSON logging setup
├── requirements.txt              # Pinned direct dependencies
├── requirements.lock             # Full transitive lock
├── pyproject.toml                # ruff + mypy config
├── pytest.ini                    # pytest configuration
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # Local / self-hosted deployment
├── render.yaml                   # Render.com deployment config
│
├── agent/
│   ├── agent.py                  # LangGraph agent: build, stream, run
│   ├── tools.py                  # structured_query, semantic_search, director_gross_summary
│   └── prompts.py                # System prompt + ChatPromptTemplate
│
├── data/
│   ├── loader.py                 # CSV loader with cleaning (lru_cache)
│   ├── vectorstore.py            # FAISS / Pinecone / ChromaDB dispatch
│   └── faiss_index/              # Pre-built FAISS index (committed)
│       ├── index.faiss
│       └── index.pkl
│
├── imdb_dataset/
│   └── imdb_top_1000.csv         # Source data — do not modify
│
├── tests/
│   ├── conftest.py               # Shared fixtures (sample_df, mock_vectorstore)
│   ├── test_loader.py            # 27 tests — cleaning rules, types, edge cases
│   ├── test_tools.py             # 36 tests — all three tools
│   ├── test_vectorstore.py       # 12 tests — provider dispatch, rebuild
│   └── test_agent.py             # 39 tests — agent scenarios, streaming, caching
│
├── .github/
│   └── workflows/
│       └── ci.yml                # ruff → mypy → pytest on push/PR to master
│
├── .streamlit/
│   └── config.toml               # Headless server config for deployment
│
├── .env.example                  # Template for required environment variables
└── .dockerignore                 # Excludes secrets, venv, logs from Docker context
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **LangGraph ReAct agent** | Supports multi-turn memory, multi-tool chaining, and controllable recursion depth |
| **FAISS committed to repo** | Zero-config startup — no re-embedding required on first run or deployment |
| **`@st.cache_resource` for DF + vector store** | Single load per worker process; all browser sessions share the same read-only objects |
| **Per-session `agent_executor` in `st.session_state`** | Each browser tab gets isolated conversation history |
| **`stream_mode="messages"` in LangGraph** | Token-by-token streaming eliminates long blank waits in the UI |
| **Tool-dispatch chunk filtering** | Function-call JSON tokens are suppressed; only final prose is rendered to the user |
| **SQLite LLM cache** | Exact-match queries (title lookups, counts) are served instantly on repeat without API calls |
| **Non-root Docker user** | Defence-in-depth; container cannot write to system paths even if exploited |
| **Multi-stage Docker build** | Runtime image has no compilers or build tools, reducing attack surface and image size |
