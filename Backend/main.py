"""Bunny AI FastAPI backend.

Supports JSON chat requests and multipart/form-data requests with multiple
attachments. Configure GEMINI_API_KEY on Render; never put the key in frontend
code or commit it to GitHub.
"""
from __future__ import annotations

import csv
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile as StarletteUploadFile

from ai import ask_ai, get_model_name
from bunny import Bunny
from memory import delete_session

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("bunny-api")

app = FastAPI(
    title="Bunny AI API",
    version="4.0.0",
    description="Bunny AI chat API with Gemini and supported file attachments.",
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

    return [
        origin.strip().rstrip("/")
        for origin in source
        if origin.strip() and origin.strip() != "*"
    ]


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
MAX_TOTAL_TEXT_CHARS = 60_000
MAX_MESSAGE_CHARS = 20_000
MAX_HISTORY_MESSAGES = 12

IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

BINARY_GEMINI_TYPES = {".pdf"}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv",
    ".json", ".jsonl", ".xml", ".html", ".htm",
    ".css", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".jsx", ".py", ".java", ".c", ".cpp", ".h",
    ".hpp", ".cs", ".php", ".rb", ".go", ".rs",
    ".sql", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".log", ".sh", ".bat", ".ps1",
}

SUPPORTED_EXTENSIONS = (
    IMAGE_MIME_TYPES
    | {".pdf": "application/pdf"}
    | {extension: "text/plain" for extension in TEXT_EXTENSIONS}
    | {
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
    }
)


def get_api_key() -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()


def get_session_id(value: str | None) -> str:
    candidate = (value or "").strip()

    if not candidate:
        return secrets.token_urlsafe(24)

    if len(candidate) > 160 or not re.fullmatch(
        r"[A-Za-z0-9_-]+", candidate
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id."
        )

    return candidate


def get_session_resources(
    session_id: str
) -> tuple[Bunny, threading.RLock]:

    with _SESSION_MAP_LOCK:

        bunny = _ACTIVE_BUNNIES.get(session_id)

        if bunny is None:
            bunny = Bunny(session_id)
            _ACTIVE_BUNNIES[session_id] = bunny

        lock = _SESSION_LOCKS.get(session_id)

        if lock is None:
            lock = threading.RLock()
            _SESSION_LOCKS[session_id] = lock

        return bunny, lock


def normalize_history_context(raw: Any) -> list[dict[str, str]]:

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, list):
        return []

    cleaned = []

    for item in raw[-MAX_HISTORY_MESSAGES:]:

        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role in {"user", "assistant"} and content:

            cleaned.append({
                "role": role,
                "content": content[:MAX_MESSAGE_CHARS]
            })

    return cleaned


def decode_text(data: bytes) -> str:

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1"
    ):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def extract_docx(data: bytes) -> str:

    from docx import Document

    document = Document(io.BytesIO(data))
    blocks = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            blocks.append(text)

    for table_index, table in enumerate(
        document.tables, start=1
    ):

        blocks.append(f"[Table {table_index}]")

        for row in table.rows:

            blocks.append(
                " | ".join(
                    cell.text.strip().replace("\n", " ")
                    for cell in row.cells
                )
            )

    return "\n".join(blocks)


def extract_pptx(data: bytes) -> str:

    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))
    blocks = []

    for slide_number, slide in enumerate(
        presentation.slides, start=1
    ):

        blocks.append(f"[Slide {slide_number}]")

        for shape in slide.shapes:

            if getattr(shape, "has_text_frame", False):

                text = shape.text.strip()

                if text:
                    blocks.append(text)

            if getattr(shape, "has_table", False):

                for row in shape.table.rows:

                    blocks.append(
                        " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                        )
                    )

    return "\n".join(blocks)


def extract_xlsx(data: bytes) -> str:

    from openpyxl import load_workbook

    workbook = load_workbook(
        io.BytesIO(data),
        read_only=True,
        data_only=True
    )

    blocks = []

    try:

        for sheet in workbook.worksheets:

            blocks.append(f"[Sheet: {sheet.title}]")

            for row in sheet.iter_rows(values_only=True):

                values = [
                    "" if value is None else str(value)
                    for value in row
                ]

                if any(values):
                    blocks.append(" | ".join(values))

                if sum(len(item) for item in blocks) > MAX_TOTAL_TEXT_CHARS:
                    break

            if sum(len(item) for item in blocks) > MAX_TOTAL_TEXT_CHARS:
                break

    finally:
        workbook.close()

    return "\n".join(blocks)


def extract_xls(data: bytes) -> str:

    import xlrd

    workbook = xlrd.open_workbook(
        file_contents=data,
        on_demand=True
    )

    blocks = []

    try:

        for sheet in workbook.sheets():

            blocks.append(f"[Sheet: {sheet.name}]")

            max_rows = min(sheet.nrows, 500)
            max_cols = min(sheet.ncols, 60)

            for row_index in range(max_rows):

                values = [
                    str(sheet.cell_value(row_index, col))
                    for col in range(max_cols)
                ]

                if any(values):
                    blocks.append(" | ".join(values))

                if sum(len(item) for item in blocks) > MAX_TOTAL_TEXT_CHARS:
                    break

            if sum(len(item) for item in blocks) > MAX_TOTAL_TEXT_CHARS:
                break

    finally:
        workbook.release_resources()

    return "\n".join(blocks)


def extract_csv(data: bytes, extension: str) -> str:

    content = decode_text(data)

    delimiter = "\t" if extension == ".tsv" else ","

    try:

        rows = csv.reader(
            io.StringIO(content),
            delimiter=delimiter
        )

        return "\n".join(
            " | ".join(cell for cell in row)
            for row in rows
        )

    except csv.Error:
        return content


def process_attachment(
    filename: str,
    mime_type: str,
    data: bytes
) -> dict[str, Any]:

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: {suffix or 'unknown'}. "
                "Supported: images, PDF, DOCX, PPTX, XLSX, XLS "
                "and common text/code files."
            )
        )

    expected_mime = SUPPORTED_EXTENSIONS[suffix]

    if not mime_type or mime_type == "application/octet-stream":
        mime_type = expected_mime

    if suffix in IMAGE_MIME_TYPES or suffix in BINARY_GEMINI_TYPES:

        return {
            "name": filename,
            "mime_type": mime_type,
            "data": data
        }

    try:

        if suffix == ".docx":
            extracted = extract_docx(data)

        elif suffix == ".pptx":
            extracted = extract_pptx(data)

        elif suffix == ".xlsx":
            extracted = extract_xlsx(data)

        elif suffix == ".xls":
            extracted = extract_xls(data)

        elif suffix in {".csv", ".tsv"}:
            extracted = extract_csv(data, suffix)

        else:
            extracted = decode_text(data)

    except Exception as exc:

        logger.exception(
            "Could not parse uploaded file %s",
            filename
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not read {filename}. "
                "Check that the file is valid and not password-protected."
            )
        ) from exc

    extracted = (extracted or "").strip()

    if not extracted:
        extracted = "[No readable text was extracted from this file.]"

    return {
        "name": filename,
        "text": extracted[:MAX_TOTAL_TEXT_CHARS]
    }


async def read_chat_request(
    request: Request
) -> tuple[str, str | None, Any, list[StarletteUploadFile]]:

    content_type = request.headers.get(
        "content-type", ""
    ).lower()

    message = ""
    session_id = None
    history_context = None
    uploads = []

    if (
        "multipart/form-data" in content_type
        or "application/x-www-form-urlencoded" in content_type
    ):

        try:
            form = await request.form()

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail="Could not parse form data."
            ) from exc

        message = str(
            form.get("message")
            or form.get("prompt")
            or ""
        )

        session_id = str(
            form.get("session_id") or ""
        ) or None

        history_context = form.get("history_context")

        for field_name in ("files", "files[]", "file"):

            for item in form.getlist(field_name):

                if isinstance(item, StarletteUploadFile):
                    uploads.append(item)

    elif "application/json" in content_type or not content_type:

        try:
            payload = await request.json()

        except Exception as exc:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Request body must be valid JSON "
                    "or multipart form data."
                )
            ) from exc

        if not isinstance(payload, dict):

            raise HTTPException(
                status_code=400,
                detail="JSON request body must be an object."
            )

        message = str(
            payload.get("message")
            or payload.get("prompt")
            or payload.get("text")
            or payload.get("query")
            or ""
        )

        session_id = str(
            payload.get("session_id") or ""
        ) or None

        history_context = payload.get(
            "history_context",
            payload.get("history")
        )

    else:

        raise HTTPException(
            status_code=415,
            detail="Use application/json or multipart/form-data."
        )

    if len(uploads) > MAX_FILES:

        raise HTTPException(
            status_code=413,
            detail=f"You can upload up to {MAX_FILES} files per message."
        )

    return message.strip(), session_id, history_context, uploads


@app.get("/")
async def root():

    return {
        "service": "Bunny AI API",
        "status": "online",
        "health": "/api/health",
        "docs": "/docs"
    }


@app.get("/api/health")
@app.get("/health")
async def health():

    key_configured = bool(get_api_key())

    return {
        "status": "ok",
        "service": "Bunny AI API",
        "gemini_configured": key_configured,
        "model": get_model_name()
    }


@app.post("/api/chat")
async def chat(request: Request):

    (
        message,
        requested_session_id,
        history_context,
        uploads
    ) = await read_chat_request(request)

    if len(message) > MAX_MESSAGE_CHARS:

        raise HTTPException(
            status_code=413,
            detail=f"Message is too long. Limit: {MAX_MESSAGE_CHARS} characters."
        )

    if not message and not uploads:

        raise HTTPException(
            status_code=422,
            detail="Enter a message or attach at least one file."
        )

    api_key = get_api_key()

    if not api_key:

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API key is not configured. "
                "Add GEMINI_API_KEY in Render → Environment."
            )
        )

    session_id = get_session_id(requested_session_id)

    bunny, session_lock = get_session_resources(session_id)

    context = normalize_history_context(history_context)

    if context:

        with session_lock:

            system_message = (
                bunny.messages[0]
                if bunny.messages
                and bunny.messages[0].get("role") == "system"
                else None
            )

            bunny.messages = (
                ([system_message] if system_message else [])
                + context
            )

    media_parts = []
    total_upload_bytes = 0
    total_extracted_chars = 0
    file_names = []

    for upload in uploads:

        filename = Path(
            upload.filename or "attachment"
        ).name[:180]

        raw = await upload.read(MAX_FILE_BYTES + 1)

        if len(raw) > MAX_FILE_BYTES:

            raise HTTPException(
                status_code=413,
                detail=f"{filename} is larger than 15 MB."
            )

        total_upload_bytes += len(raw)

        if total_upload_bytes > MAX_TOTAL_BYTES:

            raise HTTPException(
                status_code=413,
                detail="Combined attachment size must be 25 MB or less."
            )

        if not raw:

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty."
            )

        mime_type = (
            upload.content_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

        part = process_attachment(
            filename,
            mime_type,
            raw
        )

        if isinstance(part.get("text"), str):

            remaining = (
                MAX_TOTAL_TEXT_CHARS
                - total_extracted_chars
            )

            if remaining <= 0:

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Extracted attachment text exceeds "
                        "the 60,000-character limit."
                    )
                )

            part["text"] = part["text"][:remaining]

            total_extracted_chars += len(part["text"])

        media_parts.append(part)
        file_names.append(filename)

    user_message = message or (
        "Please analyze the attached file(s), explain their "
        "important content, and answer based on them."
    )

    api_user_text = user_message

    if file_names:

        api_user_text += (
            "\n\n[Attached files: "
            + ", ".join(file_names)
            + "]"
        )

    def process_turn():

        with session_lock:

            if media_parts:

                reply = ask_ai(
                    api_key,
                    user_message,
                    bunny.messages,
                    media_parts=media_parts
                )

                bunny._save_turn(
                    api_user_text,
                    reply
                )

                return {
                    "reply": reply,
                    "action": None
                }

            return bunny.handle_message(
                user_message,
                api_key
            )

    try:

        result = await run_in_threadpool(process_turn)

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Chat request failed for session %s",
            session_id
        )

        detail = str(exc).replace(
            api_key,
            "[REDACTED]"
        )[:500]

        raise HTTPException(
            status_code=502,
            detail=f"Bunny could not generate a response: {detail}"
        ) from exc

    reply = (
        result.get("reply")
        if isinstance(result, dict)
        else None
    )

    if not isinstance(reply, str) or not reply.strip():

        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response. Please try again."
        )

    return {
        "reply": reply,
        "session_id": session_id,
        "action": (
            result.get("action")
            if isinstance(result, dict)
            else None
        )
    }


@app.post("/api/session/clear")
async def clear_chat_session(request: Request):

    try:
        payload = await request.json()

    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    requested = str(
        payload.get("session_id") or ""
    ).strip()

    if not requested:

        return {
            "status": "ok",
            "cleared": False,
            "message": "No session_id supplied."
        }

    session_id = get_session_id(requested)

    with _SESSION_MAP_LOCK:

        _ACTIVE_BUNNIES.pop(session_id, None)
        _SESSION_LOCKS.pop(session_id, None)

    delete_session(session_id)

    return {
        "status": "ok",
        "cleared": True,
        "session_id": session_id
    }


@app.delete("/api/session/{session_id}")
async def delete_chat_session(session_id: str):

    clean_id = get_session_id(session_id)

    with _SESSION_MAP_LOCK:

        _ACTIVE_BUNNIES.pop(clean_id, None)
        _SESSION_LOCKS.pop(clean_id, None)

    delete_session(clean_id)

    return {
        "status": "ok",
        "cleared": True,
        "session_id": clean_id
    }
