"""
data/usage_db.py
----------------
Lightweight SQLite logger for per-turn API usage.

Each call to log_turn() inserts one row, enabling:
  - Live per-session cost dashboard in the sidebar
  - Cross-session analytics aggregated from the DB

The DB file lives alongside the LLM cache:  .cache/usage.db

Thread-safety: a module-level Lock serialises writes; reads use their
own short-lived connections.  SQLite WAL mode is enabled so concurrent
readers never block the writer.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent / ".cache" / "usage.db"

_lock = threading.Lock()

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,
    session_id        TEXT    NOT NULL,
    query_preview     TEXT,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens      INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL    NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_session ON usage_log (session_id);
CREATE INDEX IF NOT EXISTS idx_ts      ON usage_log (ts);
"""


def init_db() -> None:
    """Create the DB file and schema if they do not already exist.

    Safe to call multiple times — all DDL uses IF NOT EXISTS.
    """
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_CREATE_SQL)
            conn.commit()
        finally:
            conn.close()


def log_turn(
    session_id: str,
    query_preview: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    cost_usd: float,
) -> None:
    """Insert one usage record.

    Silently swallows all errors so a DB hiccup never crashes the main app.
    """
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with _lock:
            conn = sqlite3.connect(str(_DB_PATH))
            try:
                conn.execute(
                    "INSERT INTO usage_log "
                    "(ts, session_id, query_preview, prompt_tokens,"
                    " completion_tokens, total_tokens, cost_usd) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts,
                        session_id,
                        query_preview[:120],
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        cost_usd,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass  # analytics must never crash the main app


def get_all_time_stats() -> dict:
    """Return aggregate stats across all sessions stored in the DB.

    Returns a dict with keys:
        total_turns    (int)
        total_sessions (int)
        total_tokens   (int)
        total_cost_usd (float)

    Returns all-zeros dict if the DB is empty or unavailable.
    """
    try:
        conn = sqlite3.connect(str(_DB_PATH))
        try:
            row = conn.execute(
                "SELECT COUNT(*), COUNT(DISTINCT session_id),"
                " SUM(total_tokens), SUM(cost_usd) FROM usage_log"
            ).fetchone()
        finally:
            conn.close()
        return {
            "total_turns": int(row[0] or 0),
            "total_sessions": int(row[1] or 0),
            "total_tokens": int(row[2] or 0),
            "total_cost_usd": float(row[3] or 0.0),
        }
    except Exception:
        return {
            "total_turns": 0,
            "total_sessions": 0,
            "total_tokens": 0,
            "total_cost_usd": 0.0,
        }
