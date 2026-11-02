
"""Bunny AI - JSON-only FastAPI backend."""

import logging
import os
import re
import secrets
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai import ask_ai

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bunny-api")

app = FastAPI(
    title="Bunny AI API",
    version="4.0.0",
    description="Bunny AI JSON chat API"
)

DEFAULT_ORIGINS = [
    "https://anup-prasad008.github.io",
    "https://anup-prasad1408.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

configured_origins = os.getenv("CORS_ORIGINS", "").strip()

origins = (
    [x.strip().rstrip("/") for x in configured_origins.split(",") if x.strip()]
    if configured_origins
    else DEFAULT_ORIGINS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)

MAX_MESSAGE_LENGTH = 20000
MAX_HISTORY_MESSAGES = 24

sessions = {}
session_locks = {}
sessions_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(default=None, max_length=160)
    history_context: list[dict] = Field(default_factory=list)


class SessionRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=160)


def get_session_id(value):
    value = (value or "").strip()

    if not value:
        return secrets.token_urlsafe(24)

    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise HTTPException(status_code=400, detail="Invalid session ID.")

    return value


def get_session_resources(session_id):
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = []

        if session_id not in session_locks:
            session_locks[session_id] = threading.RLock()

        return sessions[session_id], session_locks[session_id]


@app.get("/")
async def root():
    return {
        "name": "Bunny AI API",
        "status": "online",
        "version": "4.0.0"
    }


@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "service": "Bunny AI",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY")),
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    session_id = get_session_id(request.session_id)
    history, lock = get_session_resources(session_id)

    with lock:
        if not history and request.history_context:
            for item in request.history_context[-12:]:
                role = item.get("role")
                content = str(item.get("content", "")).strip()

                if role in ("user", "assistant") and content:
                    history.append({
                        "role": role,
                        "content": content[:MAX_MESSAGE_LENGTH]
                    })

        history.append({
            "role": "user",
            "content": message
        })

        history = history[-MAX_HISTORY_MESSAGES:]

        try:
            reply = ask_ai(history)

            if not isinstance(reply, str) or not reply.strip():
                raise RuntimeError("Gemini returned an empty response.")

            history.append({
                "role": "assistant",
                "content": reply.strip()
            })

            sessions[session_id] = history[-MAX_HISTORY_MESSAGES:]

            return {
                "success": True,
                "reply": reply.strip(),
                "session_id": session_id
            }

        except Exception as error:
            logger.exception("Chat request failed")

            if history and history[-1]["role"] == "user":
                history.pop()

            raise HTTPException(
                status_code=502,
                detail=str(error) or "AI service request failed."
            )


@app.post("/api/session/clear")
async def clear_session(request: SessionRequest):
    session_id = get_session_id(request.session_id)

    history, lock = get_session_resources(session_id)

    with lock:
        history.clear()

    return {
        "success": True,
        "message": "Conversation cleared."
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session_id = get_session_id(session_id)
    history, lock = get_session_resources(session_id)

    with lock:
        return {
            "session_id": session_id,
            "messages": history
        }
