"""
logging_config.py
-----------------
Configures application-wide structured JSON logging.

Call setup_logging() once at application startup (app.py).
All other modules obtain a logger via logging.getLogger(__name__).

Log level is controlled by the LOG_LEVEL environment variable (default: INFO).
To ship logs to a centralised sink (CloudWatch, Datadog, etc.) replace the
StreamHandler below with the appropriate handler/integration.
"""

import logging
import os
import sys


def setup_logging() -> None:
    """Configure the root logger with a JSON formatter.

    Falls back to plain-text formatting if python-json-logger is not installed.
    Safe to call multiple times (idempotent — replaces the first handler only).
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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)

    # Prevent duplicate handlers on Streamlit reruns
    if root.handlers:
        root.handlers[0] = handler
    else:
        root.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("langgraph").setLevel(logging.WARNING)
