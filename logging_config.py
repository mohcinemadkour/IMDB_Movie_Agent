"""
logging_config.py
-----------------
Configures application-wide structured JSON logging.

Call setup_logging() once at application startup (app.py).
All other modules obtain a logger via logging.getLogger(__name__).

Environment variables:
  LOG_LEVEL  — logging level: DEBUG | INFO | WARNING | ERROR (default: INFO)
  LOG_FILE   — path to log file (default: logs/app.log).
               Set to empty string to disable file logging.

To ship logs to a centralised sink (CloudWatch, Datadog, etc.) add the
appropriate handler below alongside the existing handlers.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_LOG_FILE = "logs/app.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_BACKUP_COUNT = 5               # keep last 5 rotated files


def setup_logging() -> None:
    """Configure the root logger with JSON-formatted stdout and file handlers.

    Handlers added:
      1. StreamHandler(stdout)   — always present, for Streamlit console / container logs
      2. RotatingFileHandler     — writes to LOG_FILE (default logs/app.log);
                                   omitted when LOG_FILE env var is set to empty string.

    Falls back to plain-text formatting if python-json-logger is not installed.
    Safe to call multiple times (idempotent — replaces handlers on the root logger).
    """
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    try:
        from pythonjsonlogger.jsonlogger import JsonFormatter

        formatter: logging.Formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    except ImportError:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handlers: list[logging.Handler] = []

    # 1. Stdout handler (always)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    handlers.append(stdout_handler)

    # 2. Rotating file handler (opt-out via LOG_FILE="")
    log_file = os.getenv("LOG_FILE", _DEFAULT_LOG_FILE)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = handlers  # replace all (idempotent on Streamlit reruns)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
