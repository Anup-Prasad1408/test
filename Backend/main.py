"""
Bunny AI Backend
FastAPI + Gemini
"""

from __future__ import annotations

import os
import secrets
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bunny import Bunny
from memory import clear_history, clear_memory, delete_session, load_memory


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Bunny AI API",
    version="2.0.0",
    description="Backend API for Bunny AI Assistant",
)


# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [
    # GitHub Pages
    "https://anup-prasad1408.github.io",

    # Local development
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
    allow_headers=["Content-Type", "Accept"],
)


# ============================================================
# SESSION STORAGE
# ============================================================

_ACTIVE_BUNNIES: dict[str, Bunny] = {}


# ============================================================
# MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=20_000
    )

    session_id: str | None = Field(
        default=None,
        max_length=200
    )


class ChatResponse(BaseModel):
    reply: str
    action: dict[str, Any] | None = None
    session_id: str


class SessionRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        max_length=200
    )


# ============================================================
# HELPERS
# ============================================================

def get_session_id(value: str | None) -> str:
    """
    Return existing session ID or create a secure new one.
    """

    value = (value or "").strip()

    if value:
        return value

    return secrets.token_urlsafe(18)


def get_bunny(session_id: str) -> Bunny:
    """
    Get existing Bunny instance for session
    or create a new one.
    """

    bunny = _ACTIVE_BUNNIES.get(session_id)

    if bunny is None:
        bunny = Bunny(session_id)
        _ACTIVE_BUNNIES[session_id] = bunny

    return bunny


def get_gemini_api_key() -> str:
    """
    Read Gemini API key only from Render environment variables.
    """

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API is not configured on the server. "
                "Please add GEMINI_API_KEY in Render Environment Variables."
            ),
        )

    return api_key


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():
    return {
        "name": "Bunny AI",
        "status": "online",
        "message": "Bunny backend is running.",
        "version": "2.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():
    """
    Used by frontend to check backend availability.
    """

    return {
        "status": "ok",
        "service": "Bunny AI",
        "backend": "online",
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):

    # Check Gemini configuration
    api_key = get_gemini_api_key()

    # Get/create session
    session_id = get_session_id(payload.session_id)

    # Get Bunny instance
    bunny = get_bunny(session_id)

    try:
        result = bunny.handle_message(
            payload.message,
            api_key
        )

    except Exception as exc:

        print("BUNNY ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Bunny could not process the request: {str(exc)}",
        )

    return ChatResponse(
        reply=result.get(
            "reply",
            "Sorry, I could not generate a response."
        ),

        action=result.get("action"),

        session_id=session_id,
    )


# ============================================================
# CLEAR SESSION
# ============================================================

@app.post("/api/session/clear")
async def clear_session(payload: SessionRequest):

    session_id = get_session_id(payload.session_id)

    clear_history(session_id)
    clear_memory(session_id)

    _ACTIVE_BUNNIES.pop(session_id, None)

    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session cleared successfully.",
    }


# ============================================================
# DELETE SESSION
# ============================================================

@app.delete("/api/session")
async def delete_session_endpoint(
    payload: SessionRequest
):

    session_id = get_session_id(payload.session_id)

    delete_session(session_id)

    _ACTIVE_BUNNIES.pop(session_id, None)

    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session deleted successfully.",
    }


# ============================================================
# MEMORY
# ============================================================

@app.get("/api/session/{session_id}/memory")
async def get_memory(session_id: str):

    return {
        "session_id": session_id,
        "memory": load_memory(session_id),
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )
