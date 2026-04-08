# Next Steps — Enterprise & Production Readiness

This document captures the improvements needed to take the IMDB Movie Agent from a working prototype to a production-grade, enterprise-ready application. Items are grouped by priority.

---

## 🔴 Critical (Do First)

### 1. Secrets Management
- **Problem**: `OPENAI_API_KEY` and `GOOGLE_API_KEY` are loaded from a plain `.env` file, which is easy to accidentally commit.
- **Fix**: Add `.env` to `.gitignore` (verify it's excluded), provide `.env.example` with placeholder values, and document the setup in `README.md`. For production deployments, use a secrets manager (AWS Secrets Manager, Azure Key Vault, or Streamlit Community Cloud Secrets).

### 2. README.md (Missing)
- **Problem**: There is no `README.md` with quickstart instructions — flagged as outstanding in `checklist.md`.
- **Fix**: Create `README.md` covering: prerequisites, virtual env setup, `pip install -r requirements.txt`, copying `.env.example` → `.env`, adding the API key, and running `streamlit run app.py`. Include the model used (GPT-4o by default) and a note on the FAISS index.

### 3. Input Validation & Cost Caps
- **Problem**: There is no cap on prompt length or the number of LLM calls per session. A large paste or a looping agent could generate unbounded spend.
- **Fix**: Truncate user input to a maximum character length (e.g., 2000 chars) before sending to the agent. Add a max-iterations guard to the LangGraph agent. Optionally, add usage tracking per session.

### 4. Error Handling on API Failures
- **Problem**: OpenAI `RateLimitError`, `APIConnectionError`, and `AuthenticationError` propagate as unhandled exceptions and crash the Streamlit app.
- **Fix**: Wrap `run_agent()` and TTS calls in try/except blocks that display a user-friendly `st.error()` message and log the full traceback server-side.

---

## 🟠 High Priority

### 5. Pin Dependency Versions
- **Problem**: `requirements.txt` uses unpinned (or loosely pinned) versions. A future `pip install` could pull breaking changes.
- **Fix**: Run `pip freeze > requirements.txt` (or use `pip-compile`) after the final working state to lock all transitive dependencies.

### 6. LangSmith / Observability Tracing
- **Problem**: There is no visibility into agent reasoning chains, tool call latencies, or failure modes in production.
- **Fix**: Add `LANGSMITH_API_KEY` and `LANGCHAIN_TRACING_V2=true` to `.env`. This gives a full trace UI for every agent run with zero code changes, enabling debugging and performance monitoring.

### 7. Structured Logging
- **Problem**: The application has no logging beyond Streamlit's console output.
- **Fix**: Add Python `logging` with a JSON formatter (e.g., `python-json-logger`). Log: query received, tool invoked, result row count, latency, and errors. For production, ship logs to a centralised sink (CloudWatch, Datadog, etc.).

### 8. Merge `mic_fix` Branch to `master`
- **Problem**: The single-click microphone fix (stable key, unconditional render position) lives on the `mic_fix` branch but has not been merged to `master`.
- **Fix**: Test the recording flow on `mic_fix`, then `git merge mic_fix` into `master` and push.

---

## 🟡 Medium Priority

### 9. Dockerfile + Containerisation
- **Problem**: The app runs only via a local Python virtual environment. Sharing or deploying it requires manual environment setup.
- **Fix**: Add a `Dockerfile` (Python 3.11-slim base, `COPY requirements.txt`, `RUN pip install`, `CMD streamlit run app.py --server.port 8501`). Add a `docker-compose.yml` for local development with environment variable injection.

### 10. Caching Layer for LLM Responses
- **Problem**: Identical or near-identical queries hit the OpenAI API every time, increasing cost and latency.
- **Fix**: Use LangChain's built-in `SQLiteCache` or `InMemoryCache` for deterministic structured queries. For semantic queries, consider a short TTL Redis cache keyed on the embedding vector.

### 11. Managed / Persistent Vector Store
- **Problem**: The FAISS index is a flat file committed to Git. This works for 1000 movies but is not scalable and conflates code with data.
- **Fix**: Migrate to a managed vector DB (Pinecone free tier, ChromaDB with a persistent remote backend, or pgvector on Postgres). Keep the FAISS approach as a local-only fallback.

### 12. Async / Streaming Responses
- **Problem**: The agent blocks Streamlit while waiting for the full LLM response, giving a poor UX for longer answers.
- **Fix**: Use `run_agent` with `astream_events` and `st.write_stream` to stream tokens progressively into the chat bubble. This is supported by LangGraph out of the box.

### 13. Session Isolation
- **Problem**: `st.session_state` binds state to a browser tab, but the FAISS index and DataFrame are loaded once per process and shared. If multiple users hit the same process, there is no isolation risk for read-only data — but the agent executor is recreated per session, which could be expensive.
- **Fix**: Cache the heavy objects (`load_data`, `load_vectorstore`) with `@st.cache_resource`. The agent executor (which holds conversation state) should remain per-session in `st.session_state`.

---

## 🟢 Nice to Have

### 14. Test Suite
- **Problem**: There are no automated tests. Regressions are caught manually.
- **Fix**: Add `tests/` with `pytest`. Key test targets:
  - `data/loader.py`: verify column types, no NaN in `Runtime`, commas stripped from `Gross`
  - `agent/tools.py`: unit-test `structured_query` with known filter combinations
  - `agent/agent.py`: integration test for all 9 scenario questions using a stubbed LLM

### 15. CI/CD Pipeline
- **Problem**: There is no automated quality gate before merging to `master`.
- **Fix**: Add `.github/workflows/ci.yml` running `pytest`, `ruff` (linting), and `mypy` (type checking) on every pull request.

### 16. Role-Based Access / Authentication
- **Problem**: The Streamlit app is open to anyone with the URL.
- **Fix**: For internal deployment, add Streamlit's built-in [authentication](https://docs.streamlit.io/develop/concepts/connections/authentication) or wrap with an OAuth2 proxy. For public demos, at minimum add a password gate via `st.secrets`.

### 17. Cost Monitoring Dashboard
- **Problem**: There is no visibility into API spend per user or per session.
- **Fix**: Track `usage.total_tokens` from each OpenAI response and accumulate in session state. Display a small cost indicator in the sidebar (estimated cost = tokens × rate). Optionally log to a database for cross-session analytics.

---

## Summary Table

| # | Item | Priority | Effort |
|---|------|----------|--------|
| 1 | Secrets management + `.env.example` | 🔴 Critical | Low |
| 2 | `README.md` quickstart | 🔴 Critical | Low |
| 3 | Input validation + cost caps | 🔴 Critical | Low |
| 4 | API error handling | 🔴 Critical | Low |
| 5 | Pin dependency versions | 🟠 High | Low |
| 6 | LangSmith tracing | 🟠 High | Low |
| 7 | Structured logging | 🟠 High | Medium |
| 8 | Merge `mic_fix` branch | 🟠 High | Low |
| 9 | Dockerfile + docker-compose | 🟡 Medium | Medium |
| 10 | LLM response caching | 🟡 Medium | Medium |
| 11 | Managed vector store | 🟡 Medium | High |
| 12 | Streaming responses | 🟡 Medium | Medium |
| 13 | `@st.cache_resource` for heavy objects | 🟡 Medium | Low |
| 14 | Test suite (pytest) | 🟢 Nice | High |
| 15 | CI/CD pipeline | 🟢 Nice | Medium |
| 16 | Authentication | 🟢 Nice | Medium |
| 17 | Cost monitoring dashboard | 🟢 Nice | Medium |
