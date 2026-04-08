"""
app.py
------
Streamlit entry point for the IMDB Movie Agent.

Features:
  - Conversational chat UI with persistent message history
  - Voice input via streamlit-audio-recorder + OpenAI Whisper (optional)
  - Voice output via OpenAI TTS / gTTS (optional, toggle in sidebar)
  - Example question shortcuts in the sidebar
"""

import io
import logging
import os
import traceback

import openai
import streamlit as st
from dotenv import load_dotenv

from logging_config import setup_logging

load_dotenv()
setup_logging()

_LOG = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IMDB Movie Agent",
    page_icon="🎬",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────────────────
MAX_INPUT_CHARS = 2000  # truncate user input before sending to the LLM

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "**No API key found.**  "
        "Create a `.env` file in the project root with `OPENAI_API_KEY=sk-...` "
        "and restart the app.\n\n"
        "See `.env.example` for the full template."
    )
    st.stop()


# ── Imports (deferred agent import after session-state init) ─────────────────
from agent.agent import stream_agent  # noqa: E402  (module loaded at first import)
from agent.tools import init_tool_singletons  # noqa: E402

# ── Process-level shared resource cache ──────────────────────────────────────
# @st.cache_resource loads each object exactly ONCE per Streamlit worker process
# and shares it across all browser sessions.  The DataFrame and vector store are
# read-only after startup so sharing is safe.  The agent executor (which holds
# per-session conversation state) must NOT be cached here — it lives in
# st.session_state so each browser tab gets its own isolated copy.

@st.cache_resource(show_spinner="📊 Loading IMDB dataset…")
def _load_shared_df():
    from data.loader import load_data
    return load_data()


@st.cache_resource(show_spinner="🔍 Loading vector index…")
def _load_shared_vectorstore():
    from data.vectorstore import get_vectorstore
    # _load_shared_df() is already cached; calling it here is a cache hit.
    return get_vectorstore(_load_shared_df())


# Populate tool module globals with the shared objects.
# Subsequent calls are instant (singletons already set).
_shared_df = _load_shared_df()
_shared_vs = _load_shared_vectorstore()
init_tool_singletons(_shared_df, _shared_vs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _speak(text: str, engine: str = "auto", voice: str = "nova") -> str:
    """Generate TTS audio and render a player widget the user can click to play."""
    if engine in ("openai", "auto"):
        try:
            client = openai.OpenAI()
            audio = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text[:4096],
            )
            st.audio(audio.content, format="audio/mp3")
            return "openai"
        except openai.AuthenticationError:
            _LOG.error("OpenAI TTS authentication error.\n%s", traceback.format_exc())
            return "error: invalid API key"
        except openai.RateLimitError:
            _LOG.warning("OpenAI TTS rate limit hit.\n%s", traceback.format_exc())
            if engine == "openai":
                return "error: rate limit — try again shortly"
            # engine == "auto" — fall through to gTTS
        except Exception as exc:
            _LOG.error("OpenAI TTS error: %s\n%s", exc, traceback.format_exc())
            if engine == "openai":
                return f"error: {exc}"
            # engine == "auto" — fall through to gTTS

    # gTTS (fallback or explicit selection)
    try:
        from gtts import gTTS

        tts = gTTS(text=text[:500], lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3")
        return "gtts"
    except Exception as exc:
        _LOG.error("gTTS error: %s\n%s", exc, traceback.format_exc())
        return f"error: {exc}"


def _transcribe_audio(audio_data: bytes) -> None:
    """Transcribe audio via Whisper and store result in pending_input.

    On success sets st.session_state["pending_input"] and calls st.rerun().
    On low confidence or failure shows a warning inline.
    """
    try:
        with st.spinner("Transcribing…"):
            transcript = openai.OpenAI().audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", io.BytesIO(audio_data), "audio/wav"),
                language="en",
                prompt="IMDB, movie titles, director names, genres like Horror, Comedy, Sci-Fi, actor names",
                response_format="verbose_json",
            )
        segments = getattr(transcript, "segments", []) or []
        avg_no_speech = (
            sum(getattr(s, "no_speech_prob", 0) for s in segments) / len(segments)
            if segments else 0.0
        )
        transcribed = (transcript.text or "").strip()
        if avg_no_speech >= 0.6 or not transcribed:
            st.warning(f"⚠️ Low confidence ({avg_no_speech:.0%}). Please re-record.")
            st.session_state.pop("_last_audio_hash", None)
        else:
            st.success(f"🎙️ *{transcribed}*")
            st.session_state["pending_input"] = transcribed
            st.rerun()
    except openai.AuthenticationError:
        _LOG.error("Whisper authentication error.\n%s", traceback.format_exc())
        st.error("**Authentication error:** Your OpenAI API key is invalid or expired.")
    except openai.RateLimitError:
        _LOG.warning("Whisper rate limit hit.\n%s", traceback.format_exc())
        st.warning("⚠️ Rate limit reached. Please wait a moment, then try recording again.")
        st.session_state.pop("_last_audio_hash", None)
    except openai.APIConnectionError as exc:
        _LOG.error("Whisper connection error: %s\n%s", exc, traceback.format_exc())
        st.warning("⚠️ Could not reach OpenAI. Check your network and try again.")
        st.session_state.pop("_last_audio_hash", None)
    except Exception as exc:
        _LOG.error("Transcription failed: %s\n%s", exc, traceback.format_exc())
        st.warning(f"Transcription failed: {exc}")


# ── Session state initialisation ──────────────────────────────────────────────

_SESSION_DEFAULTS: dict = {
    "messages": [],
}


def _init_session_state() -> None:
    """Seed all session-state keys with their default values on first run."""
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if "agent_executor" not in st.session_state:
        from agent.agent import build_agent_executor
        with st.spinner("🔧 Building agent executor…"):
            st.session_state.agent_executor = build_agent_executor()


_init_session_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    voice_output = st.toggle("🔊 Voice Output", value=True)

    if voice_output:
        tts_engine = st.radio(
            "TTS Engine",
            options=["auto", "openai", "gtts"],
            format_func=lambda x: {
                "auto": "🤖 Auto (OpenAI → gTTS fallback)",
                "openai": "✨ OpenAI TTS · tts-1 (higher quality)",
                "gtts": "🔡 gTTS · Google (free, 500 char limit)",
            }[x],
            index=0,
        )
        if tts_engine in ("auto", "openai"):
            tts_voice = st.selectbox(
                "OpenAI Voice",
                options=["nova", "alloy", "echo", "fable", "onyx", "shimmer"],
                index=0,
            )
        else:
            tts_voice = "nova"
    else:
        tts_engine = "auto"
        tts_voice = "nova"

    if st.button("🗑️ Clear Chat", use_container_width=True):
        for _k, _v in _SESSION_DEFAULTS.items():
            st.session_state[_k] = _v
        st.rerun()

    st.divider()
    st.markdown("**🎤 Voice Input**")
    try:
        import hashlib

        from audio_recorder_streamlit import audio_recorder  # type: ignore

        audio_data = audio_recorder(
            text="Click to record",
            icon_size="2x",
            neutral_color="#6C757D",
            recording_color="#E74C3C",
            key="mic_sidebar",
        )
        if audio_data is not None and len(audio_data) > 1000:
            audio_hash = hashlib.md5(audio_data).hexdigest()
            if audio_hash != st.session_state.get("_last_audio_hash"):
                st.session_state["_last_audio_hash"] = audio_hash
                _transcribe_audio(audio_data)
    except ImportError:
        st.caption("Install `audio-recorder-streamlit` to enable voice input.")

    st.divider()
    st.markdown("**📋 Example Questions**")

    _examples = [
        "When did The Matrix release?",
        "Top 5 movies of 2019 by meta score",
        "Top 7 comedy movies 2010–2020 by IMDB rating",
        "Horror movies with meta score > 85 and IMDB rating > 8",
        "Top directors with gross > $500M at least twice",
        "Top 10 movies with over 1M votes but lower gross",
        "Comedy movies involving death or dead people",
        "Summarize Spielberg's top-rated sci-fi movies",
        "Movies before 1990 involving police in the plot",
        "Al Pacino movies grossing over $50M with IMDB ≥ 8",
    ]
    for _q in _examples:
        if st.button(_q, key=f"ex_{_q}", use_container_width=True):
            st.session_state["pending_input"] = _q
            st.rerun()


# ── Main heading ──────────────────────────────────────────────────────────────

st.title("🎬 IMDB Movie Agent")
st.caption(
    "Ask anything about the **IMDB Top 1000** movies — "
    "powered by GPT-4o · FAISS semantic search · LangChain"
)


# ── Chat history ──────────────────────────────────────────────────────────────

for _msg in st.session_state.messages:
    with st.chat_message(_msg["role"]):
        st.markdown(_msg["content"])


# ── Input handling ────────────────────────────────────────────────────────────

_pending = st.session_state.pop("pending_input", None)
user_input: str | None = st.chat_input("Ask about movies…") or _pending

if user_input:
    # Enforce input length cap
    if len(user_input) > MAX_INPUT_CHARS:
        st.warning(
            f"⚠️ Your message was truncated to {MAX_INPUT_CHARS} characters "
            f"(original: {len(user_input):,} chars)."
        )
        user_input = user_input[:MAX_INPUT_CHARS]

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent and stream response token-by-token into the chat bubble.
    # st.write_stream() drives the sync generator and returns the full text.
    with st.chat_message("assistant"):
        response: str = ""
        try:
            response = st.write_stream(
                stream_agent(
                    st.session_state.agent_executor,
                    user_input,
                    st.session_state.messages[:-1],  # history excludes current message
                )
            )
        except openai.AuthenticationError:
            _LOG.error("OpenAI authentication error.\n%s", traceback.format_exc())
            st.error(
                "**Authentication error:** Your OpenAI API key is invalid or expired. "
                "Update `OPENAI_API_KEY` in `.env` and restart the app."
            )
            st.stop()
        except openai.RateLimitError:
            _LOG.warning("OpenAI rate limit hit.\n%s", traceback.format_exc())
            response = "⚠️ Rate limit reached. Please wait a moment and try your question again."
            st.markdown(response)
        except openai.APIConnectionError as exc:
            _LOG.error("OpenAI connection error: %s\n%s", exc, traceback.format_exc())
            response = "⚠️ Could not reach the OpenAI API. Check your network connection and try again."
            st.markdown(response)
        except Exception as exc:
            _LOG.error("Unexpected agent error: %s\n%s", exc, traceback.format_exc())
            response = f"⚠️ Something went wrong: {exc}\n\nPlease try rephrasing your question."
            st.markdown(response)

        if voice_output and response:
            with st.spinner("🔊 Generating audio…"):
                used = _speak(response, engine=tts_engine, voice=tts_voice)
            if used.startswith("error"):
                st.caption(f"⚠️ TTS failed: {used[7:]}")
            elif used == "openai":
                st.caption(f"🔊 Spoken via **OpenAI TTS** · voice: `{tts_voice}`")
            elif used == "gtts":
                st.caption("🔊 Spoken via **gTTS** (first 500 chars)")

    st.session_state.messages.append({"role": "assistant", "content": response})
