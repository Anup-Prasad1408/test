"""Bunny AI assistant logic, adapted from the user's original terminal program."""
from __future__ import annotations

import ast
import datetime
import operator
import re
import urllib.parse
from typing import Any

from ai import ask_ai, generate_summary
from memory import load_history, load_memory, save_history, save_memory

SYSTEM_PROMPT = """
You are Bunny, a friendly and helpful AI chatbot created by Anup Prasad.
Respond in a warm, engaging, and fun way, like a best friend.
Understand and reply in the user's language (e.g., if they speak Hindi, respond in Hindi; if English, in English).
Keep replies concise but natural. If the user asks something specific, answer accurately, but always add a friendly touch.
Be empathetic, fun, and supportive in all interactions.
Avoid repeating generic questions like "how are you" unless directly asked. Respond based on the user's input.
""".strip()

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
}


def build_initial_messages(session_id: str) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT
    memory = load_memory(session_id)
    if memory:
        system += f"\n\nImportant memory from previous conversations:\n{memory}"

    messages = [{"role": "system", "content": system}]
    for message in load_history(session_id):
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    return messages


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported operator")
        left, right = _eval_ast(node.left), _eval_ast(node.right)
        try:
            return float(_ALLOWED_OPERATORS[op](left, right))
        except ZeroDivisionError:
            raise ZeroDivisionError from None
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _ALLOWED_OPERATORS:
            raise ValueError("Unsupported unary operator")
        return float(_ALLOWED_OPERATORS[op](_eval_ast(node.operand)))
    raise ValueError("Unsupported expression")


def evaluate_bodmas(expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        return "Send an expression, for example: 2 + 5 * 3"
    if not re.fullmatch(r"[0-9+\-*/%().\s]+", expr):
        return "Only numbers and arithmetic operators are allowed."
    try:
        result = _eval_ast(ast.parse(expr, mode="eval"))
        return f"Result = {result:.2f}"
    except ZeroDivisionError:
        return "Division by zero!"
    except Exception:
        return "Could not evaluate that expression. Only arithmetic is supported."


def arithmetic_many(user_input: str) -> str:
    parts = user_input.split()
    if len(parts) < 3:
        return "Use an operation followed by at least two numbers, e.g. sum 1 2 3"
    cmd = parts[0].lower()
    try:
        nums = [float(x) for x in parts[1:]]
    except ValueError:
        return "I couldn't parse the numbers."

    result = nums[0]
    for n in nums[1:]:
        if cmd in {"sum", "add", "jod"}:
            result += n
        elif cmd in {"sub", "subtract", "ghata"}:
            result -= n
        elif cmd in {"mul", "multiply", "guna"}:
            result *= n
        elif cmd in {"div", "divide", "bhag"}:
            if n == 0:
                return "Division by zero!"
            result /= n
        else:
            return "Unknown operation. Use sum/add/sub/mul/div."
    return f"Result = {result:.2f}"


def number_system_conversion(text: str) -> str:
    parts = text.lower().split()
    if len(parts) < 3:
        return (
            "Usage: convert binary 1010 | convert decimal 10 | "
            "convert octal 12 | convert hex ff"
        )
    bases = {
        "binary": 2, "bin": 2,
        "decimal": 10, "dec": 10,
        "octal": 8, "oct": 8,
        "hex": 16, "hexa": 16, "hexadecimal": 16,
    }
    base_name, value = parts[1], parts[2]
    if base_name not in bases:
        return "Supported bases: binary, decimal, octal and hex."
    try:
        number = int(value, bases[base_name])
    except ValueError:
        return f"'{value}' is not a valid {base_name} number."
    return (
        f"Decimal: {number}\n"
        f"Binary: {bin(number)[2:]}\n"
        f"Octal: {oct(number)[2:]}\n"
        f"Hexa: {hex(number)[2:].upper()}"
    )


def _url(base: str, query: str) -> str:
    return base + urllib.parse.quote_plus(query)


def _extract_after(text: str, phrases: tuple[str, ...]) -> str:
    for phrase in phrases:
        m = re.search(re.escape(phrase), text, flags=re.IGNORECASE)
        if m:
            value = text[m.end():].strip(" :,-")
            if value:
                return value
    return ""


def detect_action(text: str) -> dict[str, Any] | None:
    lower = text.lower().strip()
    sites = {
        "open google": ("https://www.google.com", "Google"),
        "google kholo": ("https://www.google.com", "Google"),
        "open youtube": ("https://www.youtube.com", "YouTube"),
        "youtube kholo": ("https://www.youtube.com", "YouTube"),
        "open instagram": ("https://www.instagram.com", "Instagram"),
        "instagram kholo": ("https://www.instagram.com", "Instagram"),
        "open snapchat": ("https://www.snapchat.com", "Snapchat"),
        "snapchat kholo": ("https://www.snapchat.com", "Snapchat"),
        "open facebook": ("https://www.facebook.com", "Facebook"),
        "facebook kholo": ("https://www.facebook.com", "Facebook"),
        "open whatsapp": ("https://web.whatsapp.com", "WhatsApp Web"),
        "whatsapp kholo": ("https://web.whatsapp.com", "WhatsApp Web"),
        "open spotify": ("https://open.spotify.com", "Spotify"),
        "spotify kholo": ("https://open.spotify.com", "Spotify"),
        "open jiosaavn": ("https://www.jiosaavn.com", "JioSaavn"),
        "jiosaavn kholo": ("https://www.jiosaavn.com", "JioSaavn"),
    }
    if lower in sites:
        url, label = sites[lower]
        return {"type": "open_url", "url": url, "label": label}

    query = _extract_after(text, (
        "google search", "search google", "google par search",
        "google par", "google kholo aur",
    ))
    if query:
        return {"type": "open_url", "url": _url("https://www.google.com/search?q=", query), "label": f"Google search: {query}"}

    query = _extract_after(text, (
        "youtube search", "search youtube", "youtube par search",
        "youtube par", "youtube kholo aur", "youtube khol kar", "youtube kholke",
    ))
    if query:
        return {"type": "open_url", "url": _url("https://www.youtube.com/results?search_query=", query), "label": f"YouTube search: {query}"}

    query = _extract_after(text, (
        "spotify search", "search spotify", "spotify par search", "spotify par", "spotify kholo aur",
    ))
    if query:
        return {"type": "open_url", "url": "https://open.spotify.com/search/" + urllib.parse.quote(query), "label": f"Spotify search: {query}"}

    location = _extract_after(text, ("weather in", "weather for", "mausam in", "mausam", "weather"))
    if location:
        return {"type": "open_url", "url": _url("https://www.google.com/search?q=", f"weather in {location}"), "label": f"Weather: {location}"}
    return None


class Bunny:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id or "default"
        self.messages = build_initial_messages(self.session_id)

    def _save_turn(self, user_text: str, answer: str) -> None:
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": answer})
        save_history([
            m for m in self.messages if m.get("role") in {"user", "assistant"}
        ], self.session_id)

    def _save_summary(self, api_key: str) -> None:
        try:
            summary = generate_summary(api_key, self.messages)
            if summary:
                save_memory(summary, self.session_id)
        except Exception:
            pass

    def handle_message(self, text: str, api_key: str) -> dict[str, Any]:
        user_input = (text or "").strip()
        if not user_input:
            return {"reply": "I'm listening 😊", "action": None}

        lower = user_input.lower()

        if lower == "clear":
            self.messages = build_initial_messages(self.session_id)[:1]
            save_history([], self.session_id)
            return {"reply": "Chat cleared. Fresh start! 🐰✨", "action": {"type": "clear_chat"}}

        if "bye" in lower.split() or "exit" in lower.split():
            self._save_summary(api_key)
            return {"reply": "Aww, you're leaving already? Take care, my friend! 🐰💙", "action": None}

        if lower in {"time", "date", "samay", "tarikh"}:
            answer = f"Current local date and time is {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": None}

        if lower == "convert" or lower == "number system" or lower == "convert karo" or lower.startswith(("convert binary ", "convert decimal ", "convert octal ", "convert hex ", "convert hexa ")):
            answer = number_system_conversion(user_input)
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": None}

        if any(lower.startswith(p) for p in ("sum ", "add ", "sub ", "subtract ", "mul ", "multiply ", "div ", "divide ", "jod ", "ghata ", "guna ", "bhag ")):
            answer = arithmetic_many(lower)
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": None}

        if any(c in lower for c in "+-*/()%"):
            if all(c in set("0123456789.+-*/()% ") for c in lower):
                answer = evaluate_bodmas(lower)
                self._save_turn(user_input, answer)
                return {"reply": answer, "action": None}

        if lower.startswith(("lyrics", "gaana lyrics", "song lyrics")):
            song = _extract_after(user_input, ("lyrics", "gaana lyrics", "song lyrics"))
            if song:
                question = (
                    f"Help with the song '{song}'. Do not provide non-user-provided copyrighted lyrics in full; "
                    "provide a summary, meaning, or brief excerpt instead."
                )
                try:
                    answer = ask_ai(api_key, question, self.messages)
                except Exception as exc:
                    answer = f"Gemini API error: {exc}"
            else:
                answer = "Which song do you mean? Example: lyrics Shape of You"
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": None}

        action = detect_action(user_input)
        if action:
            answer = f"Opening {action['label']}..."
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": action}

        if lower.startswith(("play music", "song ", "music ", "gaana ")):
            query = _extract_after(user_input, ("play music", "song", "music", "gaana"))
            if query:
                action = {"type": "open_url", "url": _url("https://www.youtube.com/results?search_query=", query), "label": f"YouTube music search: {query}"}
                answer = f"Searching YouTube for {query} 🎵"
            else:
                answer = "Try: play music Arijit Singh, or open JioSaavn"
                action = None
            self._save_turn(user_input, answer)
            return {"reply": answer, "action": action}

        try:
            answer = ask_ai(api_key, user_input, self.messages)
        except Exception as exc:
            answer = f"Gemini API error: {exc}"
        self._save_turn(user_input, answer)
        return {"reply": answer, "action": None}
