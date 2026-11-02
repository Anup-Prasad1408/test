"""Bunny AI FastAPI backend with safe multimodal/document uploads."""
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
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
    version="3.0.0",
    description="Bunny AI assistant with image and document understanding.",
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
    configured = os.getenv("CORS_ORIGINS", "").strip()
    source = configured.split(",") if configured else DEFAULT_ORIGINS
    return [value.strip().rstrip("/") for value in source if value.strip() and value.strip() != "*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)

_ACTIVE_BUNNIES: dict[str, Bunny] = {}
_SESSION_LOCKS: dict[str, threading.RLock] = {}
_SESSION_MAP_LOCK = threading.Lock()
MAX_FILES = 8
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_TEXT_CHARS_PER_FILE = 35_000


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
        lock = _SESSION_LOCKS.setdefault(session_id, threading.RLock())
        return bunny, lock


def require_gemini_key() -> str:
    value = os.getenv("GEMINI_API_KEY", "").strip()
    if not value:
        raise HTTPException(status_code=503, detail="Gemini API is not configured. Add GEMINI_API_KEY to Render Environment.")
    return value

