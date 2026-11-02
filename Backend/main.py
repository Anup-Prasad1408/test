"""Bunny AI API backend.

Gemini credentials are read exclusively from the backend host environment.
This file works with the existing bunny.py, ai.py and memory.py modules.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from bunny import Bunny
from memory import clear_history, clear_memory, delete_session, load_memory


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bunny-api")


# ---------------------------------------------------------------------------
# App and CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bunny AI API",
    version="2.2.0",
    description="Backend API for Bunny AI Assistant",
)

DEFAULT_ORIGINS = [
    "https://anup-prasad1408.github.io",
    "https://anup-prasad008.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def get_allowed_origins() -> list[str]:
    """Read comma-separated browser origins from CORS_ORIGINS."""
    configured = os.getenv("CORS_ORIGINS", "").strip()
    source = configured.split(",") if configured else DEFAULT_ORIGINS
    origins = [item.strip().rstrip("/") for item in source if item.strip()]
    # Never allow wildcard origins for this credential-free API.
    return [origin for origin in origins if origin != "*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
_ACTIVE_BUNNIES: dict[str, Bunny] = {}
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_MAP_LOCK = threading.Lock()


def get_session_id(value: str | None) -> str:
    """Validate an incoming session identifier or create a new random one."""
    candidate = (value or "").strip()
    if not candidate:
        return secrets.token_urlsafe(24)
    if len(candidate) > 160 or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return candidate


def get_session_resources(session_id: str) -> tuple[Bunny, threading.RLock]:
    """Get a Bunny instance and a per-session lock safely."""
    with _SESSION_MAP_LOCK:
        bunny = _ACTIVE_BUNNIES.get(session_id)
        if bunny is None:
            bunny = Bunny(session_id)
            _ACTIVE_BUNNIES[session_id] = bunny
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
            lock = threading.RLock()
