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
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="IMDB Movie Agent",
    page_icon="🎬",
    layout="wide",
)

# ── API key guard ─────────────────────────────────────────────────────────────
if not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "**No API key found.**  "
        "Create a `.env` file in the project root with `OPENAI_API_KEY=sk-...` "
        "and restart the app.\n\n"
        "See `.env.example` for the full template."
    )
    st.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _speak(text: str, engine: str = "auto", voice: str = "nova") -> str:
    """Play TTS audio and return the engine name used ('openai'/'gtts') or 'error: ...'."""
    if engine in ("openai", "auto"):
        try:
            import openai

            client = openai.OpenAI()
            audio = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text[:4096],
            )
            st.audio(audio.content, format="audio/mp3", autoplay=True)
            return "openai"
        except Exception as exc:
            if engine == "openai":  # hard-selected — do not fall through
                return f"error: {exc}"
            # engine == "auto" — fall through to gTTS

    # gTTS (fallback or explicit selection)
    try:
        from gtts import gTTS

        tts = gTTS(text=text[:500], lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3", autoplay=True)
        return "gtts"
    except Exception as exc:
        return f"error: {exc}"


# ── Session state initialisation ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_executor" not in st.session_state:
    from agent.agent import build_agent_executor

    with st.spinner("🔧 Loading IMDB Agent (first run may build the vector index)…"):
        st.session_state.agent_executor = build_agent_executor()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")
    voice_output = st.toggle("🔊 Voice Output", value=False)

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
        st.session_state.messages = []
        st.rerun()

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


# ── Voice input (optional) ────────────────────────────────────────────────────

with st.expander("🎤 Voice Input", expanded=False):
    try:
        from audio_recorder_streamlit import audio_recorder  # type: ignore
        import hashlib

        audio_data = audio_recorder()
        # WAV header is ~44 bytes; anything shorter means no real audio was recorded.
        if audio_data is not None and len(audio_data) > 1000:
            # Deduplicate: only transcribe audio that hasn't been processed yet.
            audio_hash = hashlib.md5(audio_data).hexdigest()
            if audio_hash != st.session_state.get("_last_audio_hash"):
                st.session_state["_last_audio_hash"] = audio_hash
                import openai

                client = openai.OpenAI()
                try:
                    with st.spinner("Transcribing…"):
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=("audio.wav", io.BytesIO(audio_data), "audio/wav"),
                            language="en",
                        )
                    transcribed = transcript.text.strip()
                    if transcribed:
                        st.info(f"Transcribed: **{transcribed}**")
                        st.session_state["pending_input"] = transcribed
                        st.rerun()
                except Exception as exc:
                    st.warning(f"Voice transcription failed: {exc}")
    except ImportError:
        st.info(
            "Voice input is optional.  "
            "Install `audio-recorder-streamlit` to enable it: "
            "`pip install audio-recorder-streamlit`"
        )


# ── Chat history ──────────────────────────────────────────────────────────────

for _msg in st.session_state.messages:
    with st.chat_message(_msg["role"]):
        st.markdown(_msg["content"])


# ── Input handling ────────────────────────────────────────────────────────────

_pending = st.session_state.get("pending_input")
if _pending:
    del st.session_state["pending_input"]

user_input: str | None = st.chat_input("Ask about movies…") or _pending

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            from agent.agent import run_agent

            try:
                response = run_agent(
                    st.session_state.agent_executor,
                    user_input,
                    st.session_state.messages[:-1],  # history excludes current message
                )
            except Exception as exc:
                response = f"⚠️ Something went wrong: {exc}\n\nPlease try rephrasing your question."
        st.markdown(response)

        if voice_output:
            with st.spinner("🔊 Generating audio…"):
                used = _speak(response, engine=tts_engine, voice=tts_voice)
            if used.startswith("error"):
                st.caption(f"⚠️ TTS failed: {used[7:]}")
            elif used == "openai":
                st.caption(f"🔊 Spoken via **OpenAI TTS** · voice: `{tts_voice}`")
            elif used == "gtts":
                st.caption("🔊 Spoken via **gTTS** (first 500 chars)")

    st.session_state.messages.append({"role": "assistant", "content": response})
