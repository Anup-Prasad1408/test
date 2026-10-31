"""Bunny AI - Gemini/AI layer.

The API key is supplied by the web frontend for each request and is not
read from key.txt or persisted by this module.
"""
from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"

SYSTEM_PROMPT = """
You are Bunny, a friendly and helpful AI chatbot created by Anup Prasad.
Respond in a warm, engaging, and fun way, like a best friend.
Understand and reply in the user's language (e.g., if they speak Hindi, respond in Hindi; if English, in English).
Keep replies concise but natural. If the user asks something specific, answer accurately, but always add a friendly touch.
Be empathetic, fun, and supportive in all interactions.
Avoid repeating generic questions like "how are you" unless directly asked. Respond based on the user's input.
""".strip()


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _build_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if not content or role == "system":
            continue
        if role == "assistant":
            role = "model"
        elif role != "user":
            continue
        contents.append({"role": role, "parts": [{"text": content}]})
    return contents


def generate_gemini_response(
    api_key: str,
    messages: list[dict[str, Any]],
    temperature: float = 1.0,
    max_output_tokens: int = 5000,
) -> str:
    api_key = (api_key or "").strip()
    if not api_key:
        raise ValueError("Gemini API key is missing.")

    system_instruction = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "system"),
        SYSTEM_PROMPT,
    )
    contents = _build_contents(messages)
    if not contents:
        raise ValueError("No user message was supplied.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=get_model_name(),
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text.strip()


def ask_ai(api_key: str, question: str, messages: list[dict[str, Any]]) -> str:
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    working_messages = list(messages)
    working_messages.append({"role": "user", "content": question})
    return generate_gemini_response(api_key, working_messages, 1.0, 5000)


def generate_summary(api_key: str, messages: list[dict[str, Any]]) -> str:
    user_messages = [m for m in messages if m.get("role") == "user"]
    if len(user_messages) <= 2:
        return ""

    convo_messages = [m for m in messages if m.get("role") != "system"]
    convo_messages.append({
        "role": "user",
        "content": (
            "Summarize the key points, topics discussed, user preferences, "
            "and important information from this conversation in 2-3 sentences. "
            "Focus on what Bunny should remember for future interactions."
        ),
    })
    return generate_gemini_response(api_key, convo_messages, 0.5, 200)
