# IMDB Movie Agent

A GenAI-powered conversational voice agent built with Streamlit that answers questions about the **IMDB Top 1000** dataset. It combines structured pandas queries, semantic vector search (FAISS + RAG), and an LLM to handle everything from exact lookups to plot-based similarity searches.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Git | any recent version |
| OpenAI API key | [platform.openai.com](https://platform.openai.com) |

> **Optional:** A Google API key if you want to use Gemini as the LLM provider instead of OpenAI.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/mohcinemadkour/IMDB_Movie_Agent.git
cd IMDB_Movie_Agent
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and add your OpenAI API key:

```dotenv
OPENAI_API_KEY=sk-your-key-here
```

**How to get your API key:**
1. Go to [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Paste it into `.env` after `OPENAI_API_KEY=`

All available variables are documented in [.env.example](.env.example). The only **required** value is `OPENAI_API_KEY`.

> **Security note:** `.env` is listed in `.gitignore` and must never be committed to version control.

**Alternative:** If you don't have an OpenAI key yet, you can also use Google Gemini by setting `GOOGLE_API_KEY` and `LLM_PROVIDER=gemini` in `.env` (see [.env.example](.env.example) for details).

### 5. Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Features

- **Streaming responses** — LLM tokens are streamed token-by-token into the chat bubble via `st.write_stream()`, eliminating long blank waits.
- **Voice input** — Click the microphone in the sidebar to record a question; OpenAI Whisper transcribes it automatically.
- **Voice output** — Toggle the 🔊 Voice Output switch in the sidebar to hear responses read aloud. Choose between OpenAI TTS (`tts-1`, six voices) or the free gTTS fallback, selectable per session.
- **Shared resource caching** — The IMDB DataFrame and FAISS vector index are loaded once per Streamlit worker process with `@st.cache_resource` and shared across all browser tabs.
- **LLM response caching** — Repeated identical queries are served from a SQLite cache (`LLM_CACHE=sqlite`, default) to reduce latency and API cost.
- **Input guard** — User messages are capped at 2,000 characters before being sent to the LLM.
- **Example questions** — One-click example queries in the sidebar cover the full range of supported query types.

---

## Model

| Component | Default | Override |
|-----------|---------|----------|
| LLM (chat + reasoning) | `gpt-4o` | Set `OPENAI_MODEL=gpt-4-turbo` in `.env` |
| LLM provider | OpenAI | Set `LLM_PROVIDER=gemini` + `GOOGLE_API_KEY` in `.env` |
| LLM cache | SQLite (`.cache/llm_cache.db`) | Set `LLM_CACHE=memory` or `LLM_CACHE=none` in `.env` |
| Speech-to-text | `whisper-1` (OpenAI) | — |
| Text-to-speech | `tts-1` / `gTTS` (selectable in sidebar) | — |
| Embeddings | `text-embedding-3-small` (OpenAI) | — |
| Vector store | FAISS (local) | Set `VECTOR_STORE=pinecone` or `VECTOR_STORE=chroma` in `.env` |

---

## Agent Tools

The LangGraph agent has access to three tools, routed automatically based on the query:

| Tool | When used |
|------|-----------|
| `structured_query` | Numeric filters, sorting, director/actor lookups, genre filtering |
| `semantic_search` | Plot/theme/concept queries (searches the `Overview` column via FAISS) |
| `director_gross_summary` | "Directors with multiple blockbusters" style aggregation queries |

---

## FAISS Vector Index

The repository includes a pre-built FAISS index at `data/faiss_index/`. This index is built on the `Overview` (plot summary) column of the IMDB Top 1000 dataset using OpenAI `text-embedding-3-small` embeddings and enables semantic / plot-based searches without requiring a rebuild on first run.

If you ever need to rebuild the index (e.g., after modifying the dataset):

```python
from data.vectorstore import get_vectorstore
from data.loader import load_data
get_vectorstore(load_data(), force_rebuild=True)
```

---

## Project Structure

```
app.py                  # Streamlit entry point — chat UI + voice I/O
agent/
  agent.py              # LangGraph agent: build_agent_executor(), stream_agent(), run_agent()
  tools.py              # structured_query, semantic_search, director_gross_summary
  prompts.py            # System prompt and few-shot rules
data/
  loader.py             # CSV loader with data cleaning (lru_cache)
  vectorstore.py        # Provider-agnostic vector store (FAISS / Pinecone / ChromaDB)
  faiss_index/          # Pre-built FAISS index (committed)
imdb_dataset/
  imdb_top_1000.csv     # Source data — do not modify
logging_config.py       # Structured JSON logging setup
tests/                  # pytest test suite (114 tests)
.github/workflows/
  ci.yml                # CI pipeline: ruff → mypy → pytest on push/PR to master
pyproject.toml          # ruff + mypy configuration
requirements.txt
.env.example
```

---

## CI / CD

A GitHub Actions pipeline runs on every push and pull request to `master`:

1. **Lint** — `ruff check` across `agent/`, `data/`, `tests/`, `app.py`
2. **Type check** — `mypy agent/ data/`
3. **Tests** — `pytest tests/ -q` (114 tests)

---

## Docker

### Build and run with Docker Compose (recommended)

```bash
# 1. Copy env file and add your API key
cp .env.example .env
# edit .env — set OPENAI_API_KEY

# 2. Build and start
docker compose up --build

# 3. Open http://localhost:8501
```

Logs are persisted to `./logs/app.log` on the host via a volume mount.

### Build and run manually

```bash
docker build -t imdb-movie-agent .
docker run -p 8501:8501 --env-file .env imdb-movie-agent
```

---

## Observability (LangSmith)

LangChain and LangGraph emit traces automatically when the following variables are set in `.env`:

```dotenv
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=imdb-movie-agent
```

Sign up for a free account at [smith.langchain.com](https://smith.langchain.com). Once enabled, every agent invocation appears in the LangSmith UI with:
- Full reasoning chain (LLM calls + tool calls in order)
- Per-step latency and token counts
- Input/output payloads for debugging

No code changes are required — tracing is activated purely by the environment variables.

---

## Example Queries

- *"When did The Matrix release?"*
- *"Top 5 movies of 2019 by meta score"*
- *"Top 7 comedy movies 2010–2020 by IMDB rating"*
- *"Horror movies with meta score > 85 and IMDB rating > 8"*
- *"Directors with 2 or more movies grossing over $500M"*
- *"Top 10 movies with over 1M votes but lower gross"*
- *"Comedy movies involving death or dead people"*
- *"Movies before 1990 involving police in the plot"*
- *"Summarize Spielberg's top-rated sci-fi movies"*
- *"Al Pacino movies grossing over $50M with IMDB ≥ 8"*
