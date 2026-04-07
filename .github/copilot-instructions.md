# RealPage IMDB Agent — Project Guidelines

## Project Overview

A GenAI-powered conversational voice agent built with Streamlit that answers questions about the IMDB Top 1000 dataset. The agent combines structured pandas/SQL queries with semantic vector search (RAG) and an LLM (OpenAI or Gemini).

## Architecture

```
app.py                  # Streamlit entry point; chat UI + voice input
agent/
  agent.py              # LangChain (or similar) agent orchestration
  tools.py              # Tool definitions: structured query, semantic search, summarize
  prompts.py            # System prompt and few-shot examples
data/
  loader.py             # Load & clean imdb_top_1000.csv
  vectorstore.py        # Build / load FAISS (or ChromaDB) index on Overview column
imdb_dataset/
  imdb_top_1000.csv     # Source data — do not modify
```

## Dataset Schema (`imdb_top_1000.csv`)

| Column | Type | Notes |
|--------|------|-------|
| `Poster_Link` | str | IMDB poster URL |
| `Series_Title` | str | Movie name |
| `Released_Year` | int | Release year (some values may be strings — clean on load) |
| `Certificate` | str | Age certificate |
| `Runtime` | str | e.g. "142 min" — parse to int when needed |
| `Genre` | str | Comma-separated genres |
| `IMDB_Rating` | float | 0–10 |
| `Overview` | str | Plot summary — used for semantic/similarity search |
| `Meta_score` | float | Metacritic score 0–100 (may have NaN) |
| `Director` | str | |
| `Star1`–`Star4` | str | Cast members |
| `No_of_Votes` | int | Stored with commas — parse on load |
| `Gross` | str | Dollar amount with commas — parse to float on load |

**Data cleaning rules:**
- Strip `" min"` from `Runtime`, cast to int
- Remove commas from `Gross` and `No_of_Votes`, cast to float/int
- Drop or flag rows where `Released_Year` is not a valid integer

## Query Strategy

Use **two tool types** routed by the LLM agent:

1. **Structured tool** — pandas filtering/sorting for any query involving numeric columns (`IMDB_Rating`, `Meta_score`, `Gross`, `Released_Year`, `No_of_Votes`), genre filtering, or director/actor lookup.
2. **Semantic search tool** — FAISS/Chroma vector search over the `Overview` column for conceptual queries (e.g., "movies involving police", "movies with death in comedy").

Combine both tools for hybrid queries (e.g., filter by year *and* search by concept).

## LLM & API

- **Primary**: OpenAI (`gpt-4o` or `gpt-4-turbo`) via `openai` Python SDK
- **Fallback**: Google Gemini (`gemini-1.5-pro`) via `google-generativeai`
- Store the API key in a `.env` file (`OPENAI_API_KEY` / `GOOGLE_API_KEY`) — never hardcode keys
- Use `python-dotenv` to load `.env`

## Voice Feature

- **Input**: `streamlit-audio-recorder` or `st.audio` + OpenAI Whisper (`whisper-1`) for STT
- **Output**: OpenAI TTS (`tts-1`) or `gTTS` for text-to-speech playback in Streamlit

## Key Behaviors

- **Clarifying questions**: When a query is ambiguous (e.g., actor role), the agent should ask before answering. Example: Al Pacino query → ask if user means Star1 only or any star column.
- **Reasoning**: Always include a brief explanation of *how* the result was derived (which filters/search terms were used).
- **Recommendations**: After answering, suggest 2–3 similar movies based on comparable `IMDB_Rating` and `Meta_score` ranges.

## Build & Run

```bash
pip install -r requirements.txt
# Add OPENAI_API_KEY=sk-... to .env
streamlit run app.py
```

## Testing Key Scenarios

Before considering a feature done, verify these manually:
- Exact match lookup: "When did The Matrix release?"
- Numeric filter + sort: "Top 5 movies of 2019 by meta score"
- Genre + year range + rating: "Top 7 comedy movies 2010–2020 by IMDB rating"
- Combined numeric threshold: "Horror movies with meta score > 85 and IMDB rating > 8"
- Semantic/overview search: "Movies before 1990 involving police in the plot"
- Summarization: "Summarize Spielberg's top-rated sci-fi movies"
- Ambiguous actor query: "Al Pacino movies grossing over $50M with IMDB rating ≥ 8"

## Conventions

- Use `pandas` for all structured data operations; avoid raw SQL unless using DuckDB explicitly
- Vector store index files (`.faiss` / `chroma.sqlite3`) live in `data/` and must be committed/shared
- All monetary values are in USD; display with `$` and comma formatting in responses
- Genres are comma-separated strings — use `str.contains()` for genre filtering (case-insensitive)
- `No_of_Votes` threshold for "popular" movies: ≥ 1,000,000
