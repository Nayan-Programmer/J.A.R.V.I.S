"""
JARVIS CODEX SERVICE
====================
Professional-grade code generation backed by Groq LLMs.

Given a free-form prompt like:
    "write a python script that scrapes hacker news front page"
    "give me a react component for a dark-mode toggle in typescript"
    "build a fastapi crud endpoint for users"

`generate_code(prompt)` returns a dict:
    {
        "language":     "python",
        "filename":     "hn_scraper.py",
        "code":         "<full source>",
        "explanation":  "<how it works / how to run / dependencies>",
        "dependencies": ["httpx", "beautifulsoup4"],
        "model":        "llama-3.3-70b-versatile",
    }

The service rotates through configured Groq keys, tries multiple coder
models in order, and parses strict JSON.  If JSON parsing fails it falls
back to extracting a fenced code block from the raw response.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from config import GROQ_API_KEYS

logger = logging.getLogger("J.A.R.V.I.S")


# Models in priority order — first one that succeeds wins.
# These are all live Groq models known to be strong at code.
CODEX_MODELS: List[str] = [
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
]


CODEX_SYSTEM_PROMPT = """You are JARVIS-CODEX, a world-class senior software engineer.

You write PRODUCTION-QUALITY, professional code:
  - correct, complete, and immediately runnable
  - idiomatic for the chosen language and ecosystem
  - well structured with clear names and short focused functions
  - safe defaults (input validation, error handling, no hard-coded secrets)
  - includes brief inline comments ONLY where they add real value
  - uses modern language features and current best practices
  - chooses the right language for the task if the user does not specify
    one (Python for scripting/AI/data, TypeScript+React for web UI,
    Node.js for servers if JS is requested, Go/Rust for systems, etc.)

OUTPUT FORMAT — STRICT
Return a SINGLE valid JSON object and NOTHING else. No markdown fences,
no prose before or after. Schema:

{
  "language":     "<canonical lowercase name, e.g. python, typescript, javascript, go, rust, java, cpp, csharp, bash, sql, html>",
  "filename":     "<suggested filename WITH extension>",
  "code":         "<the complete source code as one string, with real newlines>",
  "explanation":  "<short paragraph: what the code does, how to run it, any setup needed>",
  "dependencies": ["<external package names only, may be empty>"]
}

Hard rules:
  - The "code" field MUST contain the full file contents — never truncated,
    never replaced with placeholders like '...' or 'TODO'.
  - Do NOT wrap the code in ``` fences inside the JSON string.
  - Escape newlines and quotes properly so the JSON parses.
  - Pick exactly ONE primary file. If the task truly needs multiple files,
    concatenate them into "code" with clear "# === filename ===" headers
    and put the main entry-point name in "filename".
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_EXT_BY_LANG = {
    "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js", "node": ".js",
    "typescript": ".ts", "ts": ".ts",
    "tsx": ".tsx", "jsx": ".jsx",
    "react": ".tsx",
    "go": ".go", "golang": ".go",
    "rust": ".rs", "rs": ".rs",
    "java": ".java",
    "cpp": ".cpp", "c++": ".cpp", "cxx": ".cpp",
    "c": ".c",
    "csharp": ".cs", "c#": ".cs", "cs": ".cs",
    "bash": ".sh", "shell": ".sh", "sh": ".sh",
    "sql": ".sql",
    "html": ".html",
    "css": ".css",
    "ruby": ".rb", "rb": ".rb",
    "php": ".php",
    "swift": ".swift",
    "kotlin": ".kt",
    "json": ".json",
    "yaml": ".yml", "yml": ".yml",
}


def _safe_slug(text: str, maxlen: int = 40) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug[:maxlen] or "snippet")


def _extension_for(language: str, fallback: str = ".txt") -> str:
    return _EXT_BY_LANG.get((language or "").strip().lower(), fallback)


def _strip_fences(text: str) -> str:
    """Remove ```lang ... ``` fences if the model ignored instructions."""
    if not text:
        return ""
    t = text.strip()
    fence = re.match(r"^```[\w+-]*\s*\n?(.*?)\n?```\s*$", t, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return t


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first balanced { ... } block out of arbitrary text and JSON-parse it."""
    if not text:
        return None
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # Find first '{' and matching '}'
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = t[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def _fallback_from_text(raw: str, prompt: str) -> Dict[str, Any]:
    """Last-ditch: pull the first fenced code block from raw text."""
    raw = raw or ""
    m = re.search(r"```([\w+-]*)\s*\n(.*?)```", raw, re.DOTALL)
    if m:
        lang = (m.group(1) or "").strip().lower() or "text"
        code = m.group(2).strip("\n")
    else:
        lang = "text"
        code = raw.strip()
    ext = _extension_for(lang, ".txt")
    return {
        "language": lang,
        "filename": _safe_slug(prompt) + ext,
        "code": code,
        "explanation": "Auto-extracted from non-JSON model response.",
        "dependencies": [],
    }


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------

class CodexService:
    """Generate professional code using a chain of Groq coder models."""

    def __init__(self) -> None:
        if not GROQ_API_KEYS:
            raise ValueError("CodexService requires at least one GROQ API key.")
        self._keys: List[str] = list(GROQ_API_KEYS)
        self._key_index = 0

    # -- internal --------------------------------------------------------

    def _next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _invoke(self, model: str, prompt: str) -> str:
        last_err: Optional[Exception] = None
        # Try every key once for this model.
        for _ in range(len(self._keys)):
            key = self._next_key()
            try:
                llm = ChatGroq(
                    groq_api_key=key,
                    model=model,
                    temperature=0.2,
                    max_tokens=4096,
                    request_timeout=90,
                    model_kwargs={"response_format": {"type": "json_object"}},
                )
                resp = llm.invoke([
                    SystemMessage(content=CODEX_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ])
                content = getattr(resp, "content", "") or ""
                if content.strip():
                    return content
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                # If JSON-mode isn't supported by this model, retry without it.
                if "response_format" in msg or "json_object" in msg or "json mode" in msg:
                    try:
                        llm = ChatGroq(
                            groq_api_key=key,
                            model=model,
                            temperature=0.2,
                            max_tokens=4096,
                            request_timeout=90,
                        )
                        resp = llm.invoke([
                            SystemMessage(content=CODEX_SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ])
                        content = getattr(resp, "content", "") or ""
                        if content.strip():
                            return content
                    except Exception as exc2:
                        last_err = exc2
                logger.warning("[CODEX] model=%s key=** failed: %s", model, exc)
        if last_err:
            raise last_err
        return ""

    # -- public ----------------------------------------------------------

    def generate_code(self, prompt: str) -> Dict[str, Any]:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")

        user_prompt = (
            "Task:\n" + prompt.strip() + "\n\n"
            "Produce the JSON object as specified."
        )

        last_err: Optional[Exception] = None
        for model in CODEX_MODELS:
            try:
                raw = self._invoke(model, user_prompt)
                if not raw:
                    continue
                parsed = _extract_json_object(raw) or _fallback_from_text(raw, prompt)
                code = (parsed.get("code") or "").strip()
                if not code:
                    continue

                language = (parsed.get("language") or "text").strip().lower()
                filename = (parsed.get("filename") or "").strip()
                if not filename:
                    filename = _safe_slug(prompt) + _extension_for(language, ".txt")

                explanation = (parsed.get("explanation") or "").strip()
                deps = parsed.get("dependencies") or []
                if not isinstance(deps, list):
                    deps = [str(deps)]

                return {
                    "language": language,
                    "filename": filename,
                    "code": code,
                    "explanation": explanation,
                    "dependencies": [str(d) for d in deps if d],
                    "model": model,
                }
            except Exception as exc:
                last_err = exc
                logger.warning("[CODEX] model=%s failed: %s", model, exc)
                continue

        raise RuntimeError(
            "Codex failed on all models: " + (str(last_err) if last_err else "unknown")
        )

    # -- file helpers ----------------------------------------------------

    @staticmethod
    def save_to_file(result: Dict[str, Any], out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r"[^a-zA-Z0-9_.\-]", "_", result.get("filename") or "snippet.txt")
        path = out_dir / filename
        path.write_text(result.get("code", ""), encoding="utf-8")
        return path


# Module-level singleton (lazy)
_codex_singleton: Optional[CodexService] = None


def get_codex() -> CodexService:
    global _codex_singleton
    if _codex_singleton is None:
        _codex_singleton = CodexService()
    return _codex_singleton


def format_code_result_for_chat(result: Dict[str, Any]) -> str:
    """Pretty markdown rendering for the chat bubble."""
    lang = result.get("language") or ""
    filename = result.get("filename") or "snippet"
    code = result.get("code") or ""
    explanation = (result.get("explanation") or "").strip()
    deps = result.get("dependencies") or []

    parts = []
    parts.append("**" + filename + "**" + (" — " + explanation if explanation else ""))
    if deps:
        parts.append("Dependencies: " + ", ".join(str(d) for d in deps))
    parts.append("```" + (lang or "") + "\n" + code + "\n```")
    return "\n\n".join(parts)
