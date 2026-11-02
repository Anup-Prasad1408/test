"""Bunny AI Gemini layer: balanced chat, complete code tasks and file understanding."""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types

logger = logging.getLogger("bunny-ai")
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

SYSTEM_PROMPT = """
You are Bunny, a capable and friendly AI assistant created by Anup Prasad.
Reply in the user's language (Hindi, Hinglish or English) and match their tone.
Be helpful, accurate, natural and respectful. Do not add generic greetings to
an ongoing conversation. Be reasonably concise for simple questions, but give
complete explanations for complex questions and practical tasks.

For software development requests, provide complete, runnable implementations.
Include every file that must be created or replaced, responsive design, working
interactions, validation, useful loading/error states, and clear setup steps.
Never omit essential code with ellipses, TODO placeholders, or phrases such as
"same as above". Keep frontend secrets out of client-side code. Preserve the
user's existing architecture and working features when modifying a project.

When a user attaches files, inspect the supplied file contents and visuals.
Answer based on those attachments, identify the filenames you used when useful,
and say clearly when a file format or scanned content could not be interpreted.
Do not pretend to have inspected missing or unsupported file contents.
""".strip()

CODE_TASK_PROMPT = """
FULL IMPLEMENTATION MODE:
Treat this as a real development task. Prioritize correctness and completeness
rather than shortening the response. Follow every requirement. For websites,
include full HTML, CSS and JavaScript (or all requested framework files),
responsive desktop/mobile layouts, smooth but restrained animations, accessible
controls, and working functionality. For multi-file projects, show the file tree
and complete contents of every file that needs creating or replacing. Do not
use pseudocode, ellipses, partial snippets, or TODO placeholders for required
features. State any assumptions and deployment steps.
""".strip()

FILE_TASK_PROMPT = """
ATTACHMENT ANALYSIS MODE:
Use the attached document text and/or media as primary evidence. Read relevant
sections, tables, charts and visual details. Answer the user's actual question.
For summaries, organize key points. For spreadsheets, explain relevant sheets,
columns and calculations. For code files, explain or debug the supplied code.
Do not invent content that is not present in the attachments.
""".strip()

_CODE_WORDS = re.compile(
    r"\b(website|web\s*site|webpage|web\s*app|landing\s*page|portfolio|"
    r"dashboard|frontend|backend|html|css|javascript|typescript|react|next\.?js|"
    r"vue|angular|python|api|script|software|application|component|code|project)\b",
    re.IGNORECASE,
)
_ACTION_WORDS = re.compile(
    r"\b(create|build|make|develop|generate|write|design|implement|fix|update|"
    r"complete|full|provide|bana|banao|banado|banana|likh|likho|likhkar|taiyar)\b",
    re.IGNORECASE,
)
_COMPLETE_WORDS = re.compile(
    r"\b(complete|full|entire|working|production\s*ready)\b.{0,60}"
    r"\b(code|website|project|app|implementation)\b|"
    r"\b(code|website|project|app)\b.{0,60}"
    r"\b(complete|full|entire|working)\b",
    re.IGNORECASE,
)


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def _model_output_ceiling(model: str) -> int:
    """Keep configured output limits below the selected model's known ceiling."""
    name = model.lower()
    if name.startswith("gemini-4-argon"):
        return 1_048_576
    # Current Gemini Flash-family models support up to 65,536 output tokens.
    return 65_536


def _token_limit(env_name: str, default: int, model: str) -> int:
    try:
        requested = int(os.getenv(env_name, str(default)))
    except (TypeError, ValueError):
        requested = default
    return max(1, min(requested, _model_output_ceiling(model)))

