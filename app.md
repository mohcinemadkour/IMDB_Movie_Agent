# app.py — Streamlit UI Documentation

## Overview

`app.py` is the Streamlit entry point for the IMDB Movie Agent. It wires together the conversational chat UI, voice input pipeline, text-to-speech output, and the LangChain/LangGraph agent backend.

---

## Startup Sequence

1. **Environment loading** — `python-dotenv` loads `.env` (OpenAI / Google API keys).
2. **Page config** — `st.set_page_config` sets the browser title, icon, and wide layout. This must be the first Streamlit call.
3. **API key guard** — If neither `OPENAI_API_KEY` nor `GOOGLE_API_KEY` is set, an error is shown and the app halts via `st.stop()`.
4. **Session state init** — `st.session_state.messages` (chat history list) and `st.session_state.agent_executor` (the LangGraph agent) are initialized once per browser session. The agent build triggers the FAISS vector index construction on the first run.

---

## Module Layout

```
app.py
│
├── _speak()                  # TTS helper
├── Sidebar                   # Settings + example questions
├── Main heading              # Title + caption
├── Voice input expander      # Mic recording + Whisper transcription
├── Chat history display      # Renders all past messages
└── Input handling            # chat_input + pending_input → agent → response
```

---

## Key Functions

### `_speak(text, engine, voice) → str`

Generates and plays TTS audio. Returns the engine name used (`"openai"` / `"gtts"`) or `"error: <message>"`.

| Parameter | Description |
|-----------|-------------|
| `text` | The string to speak (truncated to 4096 chars for OpenAI, 500 for gTTS) |
| `engine` | `"auto"` (OpenAI → gTTS fallback), `"openai"`, or `"gtts"` |
| `voice` | OpenAI voice name: `nova`, `alloy`, `echo`, `fable`, `onyx`, `shimmer` |

- **OpenAI TTS** (`tts-1` model): higher quality, requires `OPENAI_API_KEY`.
- **gTTS** (Google, free): used as fallback or explicit selection; limited to 500 characters.
- `st.audio(..., autoplay=True)` plays the audio inline in the browser.

---

## Sidebar

| Control | Purpose |
|---------|---------|
| **Voice Output toggle** | Enables/disables TTS after each agent response |
| **TTS Engine radio** | Select Auto / OpenAI / gTTS |
| **OpenAI Voice selectbox** | Choose from 6 OpenAI voices (shown only when engine ≠ gTTS) |
| **Clear Chat button** | Resets `st.session_state.messages` and reruns |
| **Example question buttons** | Sets `st.session_state["pending_input"]` and reruns to submit the question automatically |

---

## Voice Input Pipeline

Rendered inside a collapsible `st.expander("🎤 Voice Input")`.

```
audio_recorder()
    │
    ▼ (bytes > 1000)
MD5 deduplication  ──── already seen? skip
    │
    ▼
Whisper API (whisper-1, verbose_json, language="en", IMDB domain prompt)
    │
    ├── Low confidence? (avg no_speech_prob ≥ 0.6 OR empty text)
    │       └── ⚠️ Warning shown, audio hash reset (allow re-record)
    │
    └── High confidence
            ├── 📊 Per-segment confidence expander (🟢/🟡/🔴)
            └── Store in session_state["_pending_transcription"]
                    │
                    ▼
            Confirmation UI
              - st.info() shows transcribed text
              - st.text_input() lets user edit
              - ✅ Send  → sets pending_input, reruns
              - 🗑️ Discard → clears transcription + audio hash, reruns
```

### Whisper call details

- `response_format="verbose_json"` — returns segments with per-chunk metadata.
- `prompt` — domain hint: `"IMDB, movie titles, director names, genres like Horror, Comedy, Sci-Fi, actor names"` improves accuracy on movie vocabulary.
- `no_speech_prob` per segment — accessed via `getattr(seg, "no_speech_prob", 0)` (Pydantic object, not dict).

### Deduplication

`audio_recorder()` retains the same bytes across Streamlit reruns. An MD5 hash stored in `st.session_state["_last_audio_hash"]` prevents the same audio clip from being transcribed multiple times.

---

## Chat History

All messages are stored as dicts `{"role": "user"|"assistant", "content": "..."}` in `st.session_state.messages`. They are rendered with `st.chat_message` on every rerun, preserving the full conversation context.

---

## Input Handling

Two input sources are merged:

| Source | Mechanism |
|--------|-----------|
| **Keyboard** | `st.chat_input("Ask about movies…")` |
| **Voice / Example button** | `st.session_state["pending_input"]` set elsewhere, consumed here |

Processing flow:
1. User message is appended to history and displayed.
2. `run_agent(executor, user_input, history)` is called inside a spinner.
3. Exceptions are caught and shown as a user-friendly error message instead of crashing.
4. The assistant response is displayed with `st.markdown`.
5. If **Voice Output** is on, `_speak()` is called and a status caption is shown beneath the response indicating which TTS engine/voice was used.
6. The assistant message is appended to history.

---

## Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `messages` | `list[dict]` | Full chat history |
| `agent_executor` | `CompiledStateGraph` | Cached LangGraph agent |
| `_last_audio_hash` | `str` | MD5 of last processed audio (dedup) |
| `_pending_transcription` | `str` | Whisper output awaiting user confirmation |
| `pending_input` | `str` | Question ready to submit (from voice or example button) |

---

## Dependencies

| Package | Usage |
|---------|-------|
| `streamlit` | UI framework |
| `openai` | Whisper STT, TTS-1 |
| `audio-recorder-streamlit` | Microphone recording widget |
| `gtts` | TTS fallback |
| `python-dotenv` | `.env` loading |
| `agent.agent` | `build_agent_executor`, `run_agent` |

---

## Agent Architecture

### Framework: LangGraph (`create_agent`)

`app.py` uses `langchain.agents.create_agent` (LangChain 1.x), which returns a **`CompiledStateGraph`** — a LangGraph-based reactive agent, **not** the old `AgentExecutor`. The agent runs as a stateful graph where each node can call tools and loop until a final answer is produced.

### Component Breakdown

```
app.py
  └─ build_agent_executor()
        ├─ _build_llm()              → ChatOpenAI(gpt-4o) or ChatGoogleGenerativeAI
        ├─ get_tools()               → [structured_query, semantic_search, director_gross_summary]
        └─ create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)
              └─ returns CompiledStateGraph
```

### Invocation

```
run_agent(executor, user_input, chat_history)
  │
  ├─ format_chat_history()  → converts Streamlit dicts to LangChain HumanMessage/AIMessage
  ├─ executor.invoke({"messages": history + [HumanMessage(user_input)]})
  └─ returns result["messages"][-1].content   (last message = final answer)
```

### The 3 Tools

| Tool | When used |
|------|-----------|
| `structured_query` | Numeric filters (rating, year, gross, votes), genre, director/actor lookups; supports `count_only`, `sort_ascending` |
| `semantic_search` | Thematic/plot queries via FAISS similarity search over the `Overview` column |
| `director_gross_summary` | Aggregation queries like "directors with 2+ films grossing >$500M" |

For **hybrid queries** (e.g., "comedy movies before 1990 with police in the plot"), the LLM calls **both** `structured_query` and `semantic_search` and merges the results.

### LLM Selection

Controlled by `.env` — no code changes needed to switch:

| `LLM_PROVIDER` | Model |
|----------------|-------|
| `openai` (default) | `gpt-4o` (or `OPENAI_MODEL` env var) |
| `gemini` | `gemini-1.5-pro` (or `GEMINI_MODEL` env var) |

### System Prompt Key Behaviours

- `count_only: true` for "how many" queries — prevents returning a 10-row sample as the total count
- Clarifying question asked before answering ambiguous actor queries (e.g. Star1 vs any star column)
- Every answer includes reasoning about which filters/search terms were used
- Every answer ends with 2–3 similar movie recommendations based on comparable `IMDB_Rating` and `Meta_score`

---

## Model Selection (`OPENAI_MODEL`)

The active model is set via `OPENAI_MODEL` in `.env` (default: `gpt-4o`).

| Model | Speed | Cost | Intelligence | Best for |
|-------|-------|------|--------------|---------|
| `gpt-4o` ✅ | Fast | Mid | High | Balanced — current choice |
| `gpt-4o-mini` | Fastest | Cheapest | Good | Simple lookups, high-volume |
| `o3` / `o4-mini` | Slower | Higher | Highest | Complex multi-step reasoning |
| `gpt-4.1` | Fast | Mid | High | Coding-heavy tasks |

**Recommendation: keep `gpt-4o`.**

- Handles tool-calling (structured + semantic search routing) reliably at low latency — well-suited for a chat UI.
- `gpt-4o-mini` is cheaper but occasionally misroutes hybrid queries or misses the `count_only` flag.
- `o3`/`o4-mini` would be overkill — reasoning overhead adds latency with no meaningful benefit for structured data + FAISS queries.

To switch models, update `.env`:

```dotenv
OPENAI_MODEL=gpt-4o-mini
```

---

## Conversational Context — How Chat History is Maintained

The agent maintains full conversation context through two cooperating mechanisms.

### 1. Streamlit session state stores the full chat history

Every message is appended to `st.session_state.messages` — a plain list of dicts that persists for the lifetime of the browser session:

```python
# After user sends a message:
st.session_state.messages.append({"role": "user", "content": user_input})

# After agent replies:
st.session_state.messages.append({"role": "assistant", "content": response})
```

On every Streamlit rerun, the full list is iterated and rendered with `st.chat_message`, so the UI always shows the complete conversation.

### 2. The full history is passed to the agent on every call

`run_agent` receives `messages[:-1]` — everything except the current message:

```python
response = run_agent(
    st.session_state.agent_executor,
    user_input,
    st.session_state.messages[:-1],  # full prior history
)
```

In `agent/agent.py`, `format_chat_history` converts those dicts into LangChain message objects, then prepends them to the new message:

```python
def run_agent(executor, user_input, chat_history):
    history = format_chat_history(chat_history)   # → [HumanMessage, AIMessage, ...]
    messages = history + [HumanMessage(content=user_input)]
    result = executor.invoke({"messages": messages})
    return result["messages"][-1].content
```

The LangGraph agent receives the **entire conversation thread** as its `messages` input on every turn, giving GPT-4o full context of what was asked and answered before.

### Key Limitation

There is no summarization or truncation — the full raw history is sent to the LLM every time. This works fine for typical sessions, but for very long conversations it will eventually hit the model's context window limit (~128k tokens for GPT-4o, so not a practical concern for this use case).

---

## Voice Architecture

### Voice Input Pipeline

```
Browser mic (WebRTC)
        │
        ▼
audio-recorder-streamlit widget
  → Returns raw WAV bytes to Python on each Streamlit rerun
        │
        ▼
Guard: len(bytes) > 1000  (skip WAV-header-only clips)
        │
        ▼
MD5 deduplication  (skip if same clip already processed)
        │
        ▼
OpenAI Whisper API  (whisper-1)
  - format: verbose_json  → returns TranscriptionSegment objects
  - language: "en"
  - prompt: IMDB domain hint (improves accuracy on movie vocab)
        │
        ├── avg no_speech_prob ≥ 0.6  →  ⚠️ warning, hash reset
        │
        └── Confidence OK
              → st.session_state["pending_input"] = transcribed text
              → st.rerun()  triggers agent like a typed message
```

**Key components:**

| Component | Role |
|-----------|------|
| `audio-recorder-streamlit` | Browser mic widget — captures via WebRTC, returns WAV bytes |
| `hashlib.md5` | Dedup — prevents the same clip re-transcribing on every Streamlit rerun |
| `openai.audio.transcriptions.create` | Whisper STT — converts WAV bytes to text |
| `verbose_json` response format | Returns per-segment `no_speech_prob` for confidence scoring |
| `pending_input` session key | Bridge between voice and the normal text input handler |

---

### Voice Output Pipeline

```
Agent response text
        │
        ▼
_speak(text, engine, voice)
        │
        ├── engine = "openai" or "auto"
        │       → openai.audio.speech.create(model="tts-1", voice=voice, input=text[:4096])
        │       → returns MP3 bytes
        │       → st.audio(bytes, format="audio/mp3", autoplay=True)
        │
        └── engine = "gtts" (or OpenAI failed in "auto" mode)
                → gTTS(text[:500], lang="en")
                → writes to io.BytesIO buffer
                → st.audio(buffer, format="audio/mp3", autoplay=True)
```

**Key components:**

| Component | Role |
|-----------|------|
| `openai.audio.speech` (`tts-1`) | High-quality neural TTS; 6 voices; 4096 char limit |
| `gTTS` | Google TTS fallback; free, no key needed; 500 char limit |
| `st.audio(..., autoplay=True)` | Plays audio inline in the browser without a page reload |
| Sidebar engine radio | User picks: Auto (OpenAI → gTTS fallback) / OpenAI / gTTS |
| Sidebar voice selector | Picks OpenAI voice: nova / alloy / echo / fable / onyx / shimmer |

---

### End-to-End Voice Flow

```
[User speaks]
     ↓
audio-recorder-streamlit (WebRTC capture)
     ↓
OpenAI Whisper whisper-1 (STT → text)
     ↓
pending_input → st.chat_input handler
     ↓
LangGraph agent (GPT-4o + tools)
     ↓
response text → st.markdown (displayed)
     ↓
openai tts-1 / gTTS (TTS → MP3)
     ↓
st.audio autoplay (user hears the answer)
```

---

### Confidence Scoring

`verbose_json` returns `TranscriptionSegment` Pydantic objects. The average `no_speech_prob` across all segments is computed:

- `< 0.6` → proceed, show `🎙️ transcribed text` in sidebar
- `≥ 0.6` → warn user, reset hash so they can re-record

This prevents hallucinated transcriptions from background noise or accidental clicks being sent to the agent.
