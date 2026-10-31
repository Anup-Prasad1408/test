"""Bunny AI - persistent chat history and memory."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "chat_history.json"
MEMORY_FILE = DATA_DIR / "memory.json"
_LOCK = threading.Lock()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    _ensure_data_dir()
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, data: Any) -> None:
    _ensure_data_dir()
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_history(session_id: str = "default") -> list[dict[str, str]]:
    raw = _read_json(HISTORY_FILE, {})
    if isinstance(raw, list):
        return raw if session_id == "default" else []
    if not isinstance(raw, dict):
        return []
    value = raw.get(session_id, [])
    return value if isinstance(value, list) else []


def save_history(messages: list[dict[str, str]], session_id: str = "default") -> None:
    with _LOCK:
        raw = _read_json(HISTORY_FILE, {})
        if isinstance(raw, list):
            raw = {"default": raw}
        if not isinstance(raw, dict):
            raw = {}
        raw[session_id or "default"] = messages
        _write_json(HISTORY_FILE, raw)


def load_memory(session_id: str = "default") -> str:
    raw = _read_json(MEMORY_FILE, {})
    if not isinstance(raw, dict):
        return ""
    if "summary" in raw and session_id == "default":
        return str(raw.get("summary") or "")
    value = raw.get(session_id, "")
    if isinstance(value, dict):
        return str(value.get("summary") or "")
    return str(value or "")


def save_memory(summary: str, session_id: str = "default") -> None:
    with _LOCK:
        raw = _read_json(MEMORY_FILE, {})
        if not isinstance(raw, dict):
            raw = {}
        if session_id == "default" and set(raw.keys()) <= {"summary"}:
            raw["summary"] = summary
        else:
            raw[session_id or "default"] = {"summary": summary}
        _write_json(MEMORY_FILE, raw)


def clear_history(session_id: str = "default") -> None:
    with _LOCK:
        raw = _read_json(HISTORY_FILE, {})
        if isinstance(raw, list):
            if session_id == "default":
                _write_json(HISTORY_FILE, [])
            return
        if not isinstance(raw, dict):
            return
        raw.pop(session_id or "default", None)
        _write_json(HISTORY_FILE, raw)


def clear_memory(session_id: str = "default") -> None:
    with _LOCK:
        raw = _read_json(MEMORY_FILE, {})
        if not isinstance(raw, dict):
            return
        if session_id == "default" and "summary" in raw:
            raw["summary"] = ""
        else:
            raw.pop(session_id or "default", None)
        _write_json(MEMORY_FILE, raw)


def delete_session(session_id: str = "default") -> None:
    clear_history(session_id)
    clear_memory(session_id)
