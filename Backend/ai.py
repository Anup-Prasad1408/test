
"""Bunny AI - Google Gemini integration."""

import logging
import os

from google import genai
from google.genai import types

logger = logging.getLogger("bunny-ai")

DEFAULT_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are Bunny, a friendly and intelligent AI assistant created by Anup Prasad.

LANGUAGE:
- Reply in the user's language.
- Support Hindi, Hinglish and English.
- Match the user's communication style naturally.

BEHAVIOUR:
- Be friendly, helpful and conversational.
- Understand the full conversation context.
- Avoid unnecessary greetings and repetitive introductions.
- Give clear and useful explanations.
- Keep simple answers concise.
- Explain complex topics properly.
- Never invent facts or claim to have performed actions you did not perform.

CODING:
- Provide complete, working code when requested.
- Include all required files when a project needs multiple files.
- Avoid pseudocode and incomplete implementations.
- Explain setup and deployment when relevant.
- Preserve existing project architecture when modifying code.

CONVERSATION:
- Remember relevant context within the current conversation.
- Answer follow-up questions using previous messages.
- Ask for clarification only when necessary.
""".strip()

CODE_TASK_PROMPT = """
FULL IMPLEMENTATION MODE:

When the user requests software development:
1. Understand all requirements.
2. Provide complete implementation.
3. Include necessary imports and dependencies.
4. Include responsive design when building interfaces.
5. Include functional event handlers.
6. Include error handling and validation.
7. Explain how to run and deploy the result.

Never replace required code with placeholders or ellipses.
""".strip()


def get_model_name():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from the backend environment."
        )

    return genai.Client(api_key=api_key)


def ask_ai(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation history is empty.")

    client = get_client()

    contents = []

    for item in messages[-24:]:
        role = item.get("role")
        text = str(item.get("content", "")).strip()

        if not text:
            continue

        if role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            continue

        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=text)]
            )
        )

    if not contents:
        raise ValueError("No valid conversation messages found.")

    latest_message = str(messages[-1].get("content", ""))

    system_instruction = SYSTEM_PROMPT

    coding_words = (
        "code", "website", "html", "css", "javascript",
        "python", "backend", "frontend", "api", "project",
        "app", "software", "program", "debug"
    )

    if any(word in latest_message.lower() for word in coding_words):
        system_instruction += "\n\n" + CODE_TASK_PROMPT

    try:
        response = client.models.generate_content(
            model=get_model_name(),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.7,
                max_output_tokens=8192
            )
        )

        answer = response.text

        if not answer or not answer.strip():
            raise RuntimeError(
                "Gemini returned an empty response. Try again."
            )

        return answer.strip()

    except Exception as error:
        logger.exception("Gemini generation failed")
        raise RuntimeError(
            "AI generation failed: " + str(error)
        ) from error
