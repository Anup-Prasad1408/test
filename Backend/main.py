"""Bunny AI FastAPI backend.

Accepts both legacy JSON chat requests and multipart/form-data requests with
multiple attachments. Gemini credentials are read only from Render's env vars.
"""
from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ai import ask_ai
from bunny import Bunny
from memory import clear_history, clear_memory, delete_session, load_memory

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bunny-api")

app = FastAPI(
    title="Bunny AI API",
    version="3.1.0",
    description="Bunny AI assistant with multimodal and document uploads.",
)

DEFAULT_ORIGINS = [
    "https://anup-prasad008.github.io",
    "https://anup-prasad1408.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


def get_allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "").strip()
    source = configured.split(",") if configured else DEFAULT_ORIGINS
    origins = [origin.strip().rstrip("/") for origin in source if origin.strip()]
    # Explicit origins only. Do not combine wildcard origins with credentials.
    return [origin for origin in origins if origin != "*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
)

_ACTIVE_BUNNIES: dict[str, Bunny] = {}
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_MAP_LOCK = threading.Lock()

MAX_FILES = 8
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS_PER_FILE = 25_000
MAX_TOTAL_TEXT_CHARS = 60_000
MAX_MESSAGE_CHARS = 20_000


class SessionRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=160)


def get_session_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if not candidate:
        return secrets.token_urlsafe(24)
    if len(candidate) > 160 or not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    return candidate


def get_session_resources(session_id: str) -> tuple[Bunny, threading.RLock]:
    with _SESSION_MAP_LOCK:
        bunny = _ACTIVE_BUNNIES.get(session_id)
        if bunny is None:
            bunny = Bunny(session_id)
            _ACTIVE_BUNNIES[session_id] = bunny
        lock = _SESSION_LOCKS.get(session_id)
        if lock is None:
