            _SESSION_LOCKS[session_id] = lock
        return bunny, lock


def require_gemini_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API is not configured. Add GEMINI_API_KEY to the "
                "backend service environment variables and redeploy."
            ),
        )
    return api_key


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=160)


class SessionRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=160)


class ChatResponse(BaseModel):
    reply: str
    action: dict[str, Any] | None = None
    session_id: str


# ---------------------------------------------------------------------------
# Basic routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Bunny AI",
        "status": "online",
        "message": "Bunny backend is running.",
        "version": "2.1.0",
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Used by the frontend to verify backend reachability and key setup."""
    return {
        "status": "ok",
        "service": "Bunny AI",
        "backend": "online",
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY", "").strip()),
        "model": os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"),
    }


# ---------------------------------------------------------------------------
# Chat route
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    api_key = require_gemini_key()
    session_id = get_session_id(payload.session_id)
    bunny, session_lock = get_session_resources(session_id)

    try:
        # Bunny's conversation history is stateful. Serialize requests per session.
        result = await run_in_threadpool(
            _run_bunny_message,
            bunny,
            session_lock,
            message,
            api_key,
        )
    except HTTPException:
        raise
    except Exception as exc:  # Keep internal details in server logs only.
        logger.exception("Bunny failed to process a message for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Bunny could not process this request. Check the backend logs.",
        ) from exc

    reply = str(result.get("reply") or "Bunny could not generate a response.")
    action = result.get("action")
    if not isinstance(action, dict):
        action = None

    return ChatResponse(reply=reply, action=action, session_id=session_id)


def _run_bunny_message(
    bunny: Bunny,
    session_lock: threading.RLock,
    message: str,
    api_key: str,
) -> dict[str, Any]:
    with session_lock:
        return bunny.handle_message(message, api_key)


# ---------------------------------------------------------------------------
# Session and memory routes
# ---------------------------------------------------------------------------
@app.post("/api/session/clear")
def clear_session(payload: SessionRequest) -> dict[str, Any]:
    session_id = get_session_id(payload.session_id)
    clear_history(session_id)
    clear_memory(session_id)
    with _SESSION_MAP_LOCK:
        _ACTIVE_BUNNIES.pop(session_id, None)
        _SESSION_LOCKS.pop(session_id, None)
    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session cleared successfully.",
    }


@app.delete("/api/session")
def delete_session_endpoint(payload: SessionRequest) -> dict[str, Any]:
    session_id = get_session_id(payload.session_id)
    delete_session(session_id)
    with _SESSION_MAP_LOCK:
        _ACTIVE_BUNNIES.pop(session_id, None)
        _SESSION_LOCKS.pop(session_id, None)
    return {
        "ok": True,
        "session_id": session_id,
        "message": "Session deleted successfully.",
    }


@app.get("/api/session/{session_id}/memory")
def get_session_memory(session_id: str) -> dict[str, str]:
    safe_session_id = get_session_id(session_id)
    return {
        "session_id": safe_session_id,
        "memory": load_memory(safe_session_id),
    }


# ---------------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
