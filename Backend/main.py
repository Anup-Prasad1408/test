"""
Bunny AI Backend
FastAPI + Google Gemini
API credentials are stored only on the backend.
"""

from __future__ import annotations

import os
import secrets
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bunny import Bunny
from memory import (
    clear_history,
    clear_memory,
    delete_session,
    load_memory,
)

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bunny-api")

app = FastAPI(
    title="Bunny AI API",
    version="3.0.0",
)

# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

ALLOWED_ORIGINS = [
    "https://anup-prasad1408.github.io",
    "https://anup-prasad008.github.io",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# --------------------------------------------------
# SESSION STORAGE
# --------------------------------------------------

ACTIVE_BUNNIES: dict[str, Bunny] = {}


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=20000,
    )

    session_id: str | None = Field(
        default=None,
        max_length=200,
    )


class SessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        max_length=200,
    )


class ChatResponse(BaseModel):
    reply: str
    action: dict[str, Any] | None = None
    session_id: str


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_api_key() -> str:
    """Read Gemini API key from Render environment."""

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured on the server.",
        )

    return api_key


def get_session_id(value: str | None) -> str:
    value = (value or "").strip()

    if value:
        return value

    return secrets.token_urlsafe(18)


def get_bunny(session_id: str) -> Bunny:
    if session_id not in ACTIVE_BUNNIES:
        ACTIVE_BUNNIES[session_id] = Bunny(session_id)

    return ACTIVE_BUNNIES[session_id]


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Bunny AI",
        "status": "online",
        "version": "3.0.0",
        "message": "Bunny backend is running.",
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "Bunny AI",
        "backend": "online",
    }


# --------------------------------------------------
# CHAT API
# --------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):

    api_key = get_api_key()

    session_id = get_session_id(payload.session_id)

    bunny = get_bunny(session_id)

    try:
        result = bunny.handle_message(
            payload.message,
            api_key,
        )

        if not isinstance(result, dict):
            raise RuntimeError("Invalid Bunny response.")

        reply = result.get("reply")

        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Bunny returned an empty response.")

        return ChatResponse(
            reply=reply,
            action=result.get("action"),
            session_id=session_id,
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Bunny chat processing failed.")

        raise HTTPException(
            status_code=500,
            detail="Bunny could not process your message. Check backend logs.",
        )


# --------------------------------------------------
# CLEAR CHAT SESSION
# --------------------------------------------------

@app.post("/api/session/clear")
def clear_session(payload: SessionRequest):

    session_id = get_session_id(payload.session_id)

    clear_history(session_id)
    clear_memory(session_id)

    ACTIVE_BUNNIES.pop(session_id, None)

    return {
        "ok": True,
        "session_id": session_id,
        "message": "Chat cleared successfully.",
    }


# --------------------------------------------------
# DELETE SESSION
# --------------------------------------------------

@app.delete("/api/session")
def delete_session_endpoint(payload: SessionRequest):

    session_id = get_session_id(payload.session_id)

    delete_session(session_id)

    ACTIVE_BUNNIES.pop(session_id, None)

    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session deleted successfully.",
    }


# --------------------------------------------------
# MEMORY
# --------------------------------------------------

@app.get("/api/session/{session_id}/memory")
def get_memory(session_id: str):

    return {
        "session_id": session_id,
        "memory": load_memory(session_id),
    }


# --------------------------------------------------
# SERVER START
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )
