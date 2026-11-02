
"""Bunny AI FastAPI Backend."""

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
from starlette.datastructures import UploadFile

from ai import ask_ai, get_model_name
from bunny import Bunny
from memory import delete_session


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bunny-api")

app = FastAPI(
    title="Bunny AI API",
    version="4.0.0"
)


# ---------------- CORS ----------------

DEFAULT_ORIGINS = [
    "https://anup-prasad1408.github.io",
    "https://anup-prasad008.github.io",
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
    allow_headers=["*"],
)


# ---------------- SETTINGS ----------------

MAX_FILES = 8
MAX_FILE_SIZE = 15 * 1024 * 1024
MAX_TOTAL_SIZE = 25 * 1024 * 1024
MAX_MESSAGE_LENGTH = 20000
MAX_TEXT_LENGTH = 60000

ACTIVE_SESSIONS = {}
SESSION_LOCKS = {}
SESSION_MAP_LOCK = threading.Lock()


IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json",
    ".jsonl", ".xml", ".html", ".htm", ".css",
    ".js", ".mjs", ".ts", ".tsx", ".jsx",
    ".py", ".java", ".c", ".cpp", ".h",
    ".cs", ".php", ".rb", ".go", ".rs",
    ".sql", ".yaml", ".yml", ".toml",
    ".ini", ".log", ".sh", ".bat", ".ps1"
}

SUPPORTED_EXTENSIONS = (
    set(IMAGE_TYPES)
    | {".pdf", ".docx", ".pptx", ".xlsx", ".xls"}
    | TEXT_EXTENSIONS
)


# ---------------- API KEY ----------------

def get_api_key():

    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()


# ---------------- SESSION MANAGEMENT ----------------

def validate_session_id(value=None):

    value = (value or "").strip()

    if not value:
        return secrets.token_urlsafe(24)

    if len(value) > 160 or not re.fullmatch(
        r"[A-Za-z0-9_-]+", value
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID."
        )

    return value


def get_session(session_id):

    with SESSION_MAP_LOCK:

        bunny = ACTIVE_SESSIONS.get(session_id)

        if bunny is None:
            bunny = Bunny(session_id)
            ACTIVE_SESSIONS[session_id] = bunny

        lock = SESSION_LOCKS.get(session_id)

        if lock is None:
            lock = threading.RLock()
            SESSION_LOCKS[session_id] = lock

        return bunny, lock


# ---------------- HISTORY ----------------

def clean_history(history):

    if isinstance(history, str):

        try:
            history = json.loads(history)
        except Exception:
            return []

    if not isinstance(history, list):
        return []

    result = []

    for item in history[-12:]:

        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).lower()
        content = str(item.get("content", "")).strip()

        if role in ("user", "assistant") and content:

            result.append({
                "role": role,
                "content": content[:MAX_MESSAGE_LENGTH]
            })

    return result


# ---------------- TEXT EXTRACTION ----------------

def decode_text(data):

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):

        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def extract_docx(data):

    from docx import Document

    doc = Document(io.BytesIO(data))

    output = []

    for paragraph in doc.paragraphs:

        if paragraph.text.strip():
            output.append(paragraph.text.strip())

    for table in doc.tables:

        for row in table.rows:

            output.append(
                " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )
            )

    return "\n".join(output)


def extract_pptx(data):

    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))

    output = []

    for number, slide in enumerate(
        presentation.slides, 1
    ):

        output.append(f"\n[Slide {number}]")

        for shape in slide.shapes:

            if getattr(shape, "has_text_frame", False):

                if shape.text.strip():
                    output.append(shape.text.strip())

            if getattr(shape, "has_table", False):

                for row in shape.table.rows:

                    output.append(
                        " | ".join(
                            cell.text.strip()
                            for cell in row.cells
                        )
                    )

    return "\n".join(output)


def extract_xlsx(data):

    from openpyxl import load_workbook

    workbook = load_workbook(
        io.BytesIO(data),
        read_only=True,
        data_only=True
    )

    output = []

    try:

        for sheet in workbook.worksheets:

            output.append(f"\n[Sheet: {sheet.title}]")

            for row in sheet.iter_rows(values_only=True):

                values = [
                    str(value) if value is not None else ""
                    for value in row
                ]

                if any(values):
                    output.append(" | ".join(values))

                if sum(map(len, output)) >= MAX_TEXT_LENGTH:
                    break

            if sum(map(len, output)) >= MAX_TEXT_LENGTH:
                break

    finally:
        workbook.close()

    return "\n".join(output)


def extract_xls(data):

    import xlrd

    workbook = xlrd.open_workbook(
        file_contents=data,
        on_demand=True
    )

    output = []

    try:

        for sheet in workbook.sheets():

            output.append(f"\n[Sheet: {sheet.name}]")

            for row_index in range(min(sheet.nrows, 500)):

                values = [
                    str(sheet.cell_value(row_index, col))
                    for col in range(min(sheet.ncols, 60))
                ]

                if any(values):
                    output.append(" | ".join(values))

                if sum(map(len, output)) >= MAX_TEXT_LENGTH:
                    break

    finally:
        workbook.release_resources()

    return "\n".join(output)


def extract_file(filename, data, mime_type):

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:

        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file format: {extension}"
        )

    if extension in IMAGE_TYPES or extension == ".pdf":

        return {
            "name": filename,
            "mime_type": mime_type,
            "data": data
        }

    if extension == ".docx":
        content = extract_docx(data)

    elif extension == ".pptx":
        content = extract_pptx(data)

    elif extension == ".xlsx":
        content = extract_xlsx(data)

    elif extension == ".xls":
        content = extract_xls(data)

    elif extension in (".csv", ".tsv"):

        delimiter = "\t" if extension == ".tsv" else ","

        try:
            rows = csv.reader(
                io.StringIO(decode_text(data)),
                delimiter=delimiter
            )

            content = "\n".join(
                " | ".join(row)
                for row in rows
            )

        except csv.Error:
            content = decode_text(data)

    else:
        content = decode_text(data)

    return {
        "name": filename,
        "text": content[:MAX_TEXT_LENGTH]
    }


# ---------------- REQUEST PARSER ----------------

async def parse_request(request: Request):

    content_type = request.headers.get(
        "content-type", ""
    ).lower()

    message = ""
    session_id = None
    history = None
    files = []

    if "multipart/form-data" in content_type:

        form = await request.form()

        message = str(
            form.get("message")
            or form.get("prompt")
            or ""
        )

        session_id = form.get("session_id")
        history = form.get("history_context")

        for field in ("files", "files[]", "file"):

            for item in form.getlist(field):

                if isinstance(item, UploadFile):
                    files.append(item)

    elif "application/json" in content_type or not content_type:

        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON request."
            )

        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=400,
                detail="JSON body must be an object."
            )

        message = str(
            payload.get("message")
            or payload.get("prompt")
            or payload.get("text")
            or ""
        )

        session_id = payload.get("session_id")

        history = payload.get(
            "history_context",
            payload.get("history")
        )

    else:

        raise HTTPException(
            status_code=415,
            detail="Unsupported request content type."
        )

    if len(files) > MAX_FILES:

        raise HTTPException(
            status_code=413,
            detail=f"Maximum {MAX_FILES} files allowed."
        )

    return message.strip(), session_id, history, files


# ---------------- HEALTH ----------------

@app.get("/")
async def root():

    return {
        "service": "Bunny AI API",
        "status": "online",
        "health": "/api/health",
        "docs": "/docs"
    }


@app.get("/health")
@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "service": "Bunny AI",
        "gemini_configured": bool(get_api_key()),
        "model": get_model_name()
    }


# ---------------- MAIN CHAT API ----------------

@app.post("/api/chat")
async def chat(request: Request):

    message, requested_session, history, uploads = (
        await parse_request(request)
    )

    if len(message) > MAX_MESSAGE_LENGTH:

        raise HTTPException(
            status_code=413,
            detail="Message is too long."
        )

    if not message and not uploads:

        raise HTTPException(
            status_code=422,
            detail="Enter a message or upload a file."
        )

    api_key = get_api_key()

    if not api_key:

        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured on Render."
        )

    session_id = validate_session_id(requested_session)

    bunny, lock = get_session(session_id)

    context = clean_history(history)

    if context:

        with lock:

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

    attachments = []
    filenames = []
    total_size = 0
    total_text = 0

    for upload in uploads:

        filename = Path(
            upload.filename or "attachment"
        ).name[:180]

        data = await upload.read(MAX_FILE_SIZE + 1)

        if len(data) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 15 MB limit."
            )

        if not data:

            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty."
            )

        total_size += len(data)

        if total_size > MAX_TOTAL_SIZE:

            raise HTTPException(
                status_code=413,
                detail="Total attachment size exceeds 25 MB."
            )

        mime_type = (
            upload.content_type
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

        attachment = extract_file(
            filename,
            data,
            mime_type
        )

        if "text" in attachment:

            remaining = MAX_TEXT_LENGTH - total_text

            if remaining <= 0:

                raise HTTPException(
                    status_code=413,
                    detail="Extracted text exceeds the limit."
                )

            attachment["text"] = attachment["text"][:remaining]
            total_text += len(attachment["text"])

        attachments.append(attachment)
        filenames.append(filename)

    user_message = message or (
        "Please analyze the attached files and explain their contents."
    )

    stored_message = user_message

    if filenames:

        stored_message += (
            "\n\n[Attached files: "
            + ", ".join(filenames)
            + "]"
        )

    def generate_response():

        with lock:

            if attachments:

                reply = ask_ai(
                    api_key,
                    user_message,
                    bunny.messages,
                    media_parts=attachments
                )

                bunny._save_turn(
                    stored_message,
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

        result = await run_in_threadpool(
            generate_response
        )

    except HTTPException:
        raise

    except Exception as error:

        logger.exception("Bunny chat generation failed")

        safe_error = str(error).replace(
            api_key,
            "[REDACTED]"
        )[:500]

        raise HTTPException(
            status_code=502,
            detail=f"Bunny AI error: {safe_error}"
        ) from error

    if not isinstance(result, dict):

        raise HTTPException(
            status_code=502,
            detail="Invalid response from Bunny."
        )

    reply = result.get("reply")

    if not isinstance(reply, str) or not reply.strip():

        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response."
        )

    return {
        "reply": reply,
        "session_id": session_id,
        "action": result.get("action")
    }


# ---------------- CLEAR SESSION ----------------

@app.post("/api/session/clear")
async def clear_session(request: Request):

    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    requested = payload.get("session_id")

    if not requested:

        return {
            "status": "ok",
            "cleared": False
        }

    session_id = validate_session_id(
        str(requested)
    )

    with SESSION_MAP_LOCK:

        ACTIVE_SESSIONS.pop(session_id, None)
        SESSION_LOCKS.pop(session_id, None)

    delete_session(session_id)

    return {
        "status": "ok",
        "cleared": True,
        "session_id": session_id
    }


@app.delete("/api/session/{session_id}")
async def delete_session_api(session_id: str):

    session_id = validate_session_id(session_id)

    with SESSION_MAP_LOCK:

        ACTIVE_SESSIONS.pop(session_id, None)
        SESSION_LOCKS.pop(session_id, None)

    delete_session(session_id)

    return {
        "status": "ok",
        "cleared": True,
        "session_id": session_id
    }
