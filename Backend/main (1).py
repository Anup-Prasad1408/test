"""Bunny AI Web Backend - FastAPI server."""
from __future__ import annotations

import secrets
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bunny import Bunny
from memory import clear_history, clear_memory, delete_session, load_memory

app = FastAPI(title="Bunny AI API", version="1.0.0")

# Convenient for local development and a GitHub Pages frontend.
# In production, replace "*" with your exact frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ACTIVE_BUNNIES: dict[str, Bunny] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20_000)
    api_key: str = Field(..., min_length=10, max_length=500)
    session_id: str | None = Field(default=None, max_length=200)


class ChatResponse(BaseModel):
    reply: str
    action: dict[str, Any] | None = None
    session_id: str


class SessionRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=200)


def _get_session_id(value: str | None) -> str:
    value = (value or "").strip()
    return value or secrets.token_urlsafe(18)


def _get_bunny(session_id: str) -> Bunny:
    bunny = _ACTIVE_BUNNIES.get(session_id)
    if bunny is None:
        bunny = Bunny(session_id)
        _ACTIVE_BUNNIES[session_id] = bunny
    return bunny


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Bunny AI", "status": "online", "message": "Bunny backend is running."}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    session_id = _get_session_id(payload.session_id)
    bunny = _get_bunny(session_id)
    result = bunny.handle_message(payload.message, api_key)

    return ChatResponse(
        reply=result["reply"],
        action=result.get("action"),
        session_id=session_id,
    )


@app.post("/api/session/clear")
async def clear_session(payload: SessionRequest) -> dict[str, Any]:
    session_id = _get_session_id(payload.session_id)
    clear_history(session_id)
    clear_memory(session_id)
    _ACTIVE_BUNNIES.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@app.delete("/api/session")
async def delete_session_endpoint(payload: SessionRequest) -> dict[str, Any]:
    session_id = _get_session_id(payload.session_id)
    delete_session(session_id)
    _ACTIVE_BUNNIES.pop(session_id, None)
    return {"ok": True, "session_id": session_id}


@app.get("/api/session/{session_id}/memory")
async def get_memory(session_id: str) -> dict[str, str]:
    return {"session_id": session_id, "memory": load_memory(session_id)}


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
