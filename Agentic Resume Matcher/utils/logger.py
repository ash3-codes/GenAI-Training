"""
utils/logger.py
---------------
Structured JSON logger for the pipeline.
Every LangGraph node calls log_node() to record its execution.
Writes to logs/pipeline.log (one JSON object per line).
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timezone

from pythonjsonlogger.json import JsonFormatter


# ── Setup ─────────────────────────────────────────────────────────────────────

def _get_log_file() -> Path:
    """Import here to avoid circular imports with settings."""
    try:
        from config.settings import LOG_FILE
        return Path(LOG_FILE)
    except Exception:
        return Path("logs/pipeline.log")


def _build_logger() -> logging.Logger:
    log_file = _get_log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("resume_matcher")
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    formatter = JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (shows during development)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


_logger = _build_logger()


# ── Public API ────────────────────────────────────────────────────────────────

def log_node(
    node: str,
    status: str,                    # "success" | "error" | "skipped"
    latency_ms: float = 0.0,
    error: str | None = None,
    extra: dict | None = None,
) -> dict:
    """
    Log a node execution event. Returns the log entry dict.

    Usage inside a node:
        start = time.time()
        ... do work ...
        log_node("parse_resume_node", "success", latency_ms=(time.time()-start)*1000)
    """
    entry = {
        "node":       node,
        "status":     status,
        "latency_ms": round(latency_ms, 2),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }
    if error:
        entry["error"] = error
    if extra:
        entry.update(extra)

    if status == "error":
        _logger.error(node, extra=entry)
    else:
        _logger.info(node, extra=entry)

    return entry


def log_info(msg: str, **kwargs):
    """General info log."""
    _logger.info(msg, extra=kwargs or {})


def log_error(msg: str, **kwargs):
    """General error log."""
    _logger.error(msg, extra=kwargs or {})


# ── Timer context manager ─────────────────────────────────────────────────────

class NodeTimer:
    """
    Context manager that automatically logs node timing.

    Usage:
        with NodeTimer("parse_resume_node", state) as t:
            ... do work ...
            t.extra = {"resumes_parsed": 5}
        # Automatically logs on exit
    """
    def __init__(self, node_name: str, state: dict):
        self.node_name = node_name
        self.state = state
        self.extra: dict = {}
        self._start = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = (time.time() - self._start) * 1000
        status = "error" if exc_type else "success"
        error_msg = str(exc_val) if exc_val else None

        entry = log_node(
            node=self.node_name,
            status=status,
            latency_ms=latency,
            error=error_msg,
            extra=self.extra if self.extra else None,
        )

        # Append to state.node_logs
        if "node_logs" in self.state:
            self.state["node_logs"].append(entry)

        # Don't suppress exceptions
        return False
