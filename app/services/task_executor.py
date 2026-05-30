"""
TASK EXECUTOR
=============
Executes tasks classified by BrainService.

IMMEDIATE  - synchronous, returns browser actions (open URL, camera, etc.)
BACKGROUND - async, returns task_id; frontend polls GET /tasks/{task_id}

Supported intents
  open / play / google_search / youtube_search
  open_webcam / close_webcam
  generate_image / content
  generate_ppt / generate_ppt_email
  track_phone
"""

import asyncio
import logging
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.utils.app_launcher import launch_app
from app.services.decision_types import (
    INTENT_OPEN, INTENT_PLAY, INTENT_GOOGLE_SEARCH, INTENT_YOUTUBE_SEARCH,
    INTENT_GENERATE_IMAGE, INTENT_GENERATE_IMAGE_EMAIL, INTENT_CONTENT,
    INTENT_OPEN_WEBCAM, INTENT_CLOSE_WEBCAM,
    INTENT_GENERATE_PPT, INTENT_GENERATE_PPT_EMAIL,
    INTENT_TRACK_PHONE,
    INTENT_GENERATE_CODE, INTENT_GENERATE_CODE_EMAIL,
    BACKGROUND_TASK_TYPES,
)
from app.services.task_manager import task_manager

logger = logging.getLogger("J.A.R.V.I.S")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

async def execute_tasks(
    intents: List[Tuple[str, Dict[str, Any]]],
    groq_service=None,
    session_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:

    actions: Dict[str, Any] = {}
    background_tasks: List[Dict[str, Any]] = []
    reply_parts: List[str] = []

    wopens: List[str] = []
    plays: List[str] = []
    googlesearches: List[str] = []
    youtubesearches: List[str] = []
    cam_action: Optional[Dict] = None

    for intent_key, payload in intents:
        try:

            # ── open ─────────────────────────────────────────────────────
            if intent_key == INTENT_OPEN:
                url = payload.get("url", "https://www.google.com")
                wopens.append(url)
                reply_parts.append(_open_reply(url))
                app_name = _guess_app_name(url)
                if app_name:
                    try:
                        launched, _ = launch_app(app_name)
                        if launched:
                            logger.info("[EXECUTOR] Desktop app launched: %s", app_name)
                    except Exception:
                        pass

            # ── play ─────────────────────────────────────────────────────
            elif intent_key == INTENT_PLAY:
                query = payload.get("query") or payload.get("message", "")
                video_id = await _fetch_first_youtube_video_id(query)
                if video_id:
                    yt_url = "https://www.youtube.com/watch?v=" + video_id + "&autoplay=1"
                else:
                    yt_url = _yt_search_url(query)
                plays.append(yt_url)
                reply_parts.append("Playing " + query + " on YouTube.")

            # ── google_search ─────────────────────────────────────────────
            elif intent_key == INTENT_GOOGLE_SEARCH:
                query = payload.get("query") or payload.get("message", "")
                googlesearches.append(_google_url(query))
                reply_parts.append("Searching Google for " + query + ".")

            # ── youtube_search ────────────────────────────────────────────
            elif intent_key == INTENT_YOUTUBE_SEARCH:
                query = payload.get("query") or payload.get("message", "")
                youtubesearches.append(_yt_search_url(query))
                reply_parts.append("Searching YouTube for " + query + ".")

            # ── webcam ────────────────────────────────────────────────────
            elif intent_key == INTENT_OPEN_WEBCAM:
                cam_action = {"action": "open"}
                reply_parts.append("Opening your camera.")

            elif intent_key == INTENT_CLOSE_WEBCAM:
                cam_action = {"action": "close"}
                reply_parts.append("Closing your camera.")

            # ── generate_image ────────────────────────────────────────────
            elif intent_key == INTENT_GENERATE_IMAGE:
                prompt = payload.get("prompt") or payload.get("message", "an image")
                task_id = task_manager.create_task("generate_image", prompt[:120])
                background_tasks.append({"task_id": task_id, "type": "generate image", "label": prompt[:80]})
                reply_parts.append("Generating an image of " + prompt + " in the background.")
                asyncio.create_task(_run_image_generation(task_id, prompt))

            # ── generate_image_email ──────────────────────────────────────
            elif intent_key == INTENT_GENERATE_IMAGE_EMAIL:
                prompt = payload.get("prompt") or payload.get("message", "an image")
                task_id = task_manager.create_task("generate_image_email", prompt[:120])
                background_tasks.append({"task_id": task_id, "type": "generate image + email", "label": prompt[:80]})
                reply_parts.append("Generating an image of " + prompt + " and emailing it to you.")
                asyncio.create_task(_run_image_email_generation(task_id, prompt))

            # ── content ───────────────────────────────────────────────────
            elif intent_key == INTENT_CONTENT:
                prompt = payload.get("prompt") or payload.get("message", "write something")
                task_id = task_manager.create_task("content", prompt[:120])
                background_tasks.append({"task_id": task_id, "type": "content", "label": prompt[:80]})
                reply_parts.append("Writing " + prompt + " in the background.")
                if groq_service:
                    asyncio.create_task(_run_content_generation(task_id, prompt, groq_service))
                else:
                    task_manager.fail_task(task_id, "No LLM service available")

            # ── generate_ppt ──────────────────────────────────────────────
            elif intent_key == INTENT_GENERATE_PPT:
                topic = payload.get("prompt") or payload.get("message", "general topic")
                task_id = task_manager.create_task("generate_ppt", topic[:120])
                background_tasks.append({"task_id": task_id, "type": "generate_ppt", "label": topic[:80]})
                reply_parts.append("Creating a PowerPoint presentation on " + topic + ". Ready shortly.")
                if groq_service:
                    asyncio.create_task(_run_ppt_generation(task_id, topic, groq_service, send_email=False))
                else:
                    task_manager.fail_task(task_id, "No LLM service available")

            # ── generate_ppt_email ────────────────────────────────────────
            elif intent_key == INTENT_GENERATE_PPT_EMAIL:
                topic = payload.get("prompt") or payload.get("message", "general topic")
                task_id = task_manager.create_task("generate_ppt_email", topic[:120])
                background_tasks.append({"task_id": task_id, "type": "generate_ppt_email", "label": topic[:80]})
                reply_parts.append("Creating a presentation on " + topic + " and sending it to your email.")
                if groq_service:
                    asyncio.create_task(_run_ppt_generation(task_id, topic, groq_service, send_email=True))
                else:
                    task_manager.fail_task(task_id, "No LLM service available")

            # ── track_phone ───────────────────────────────────────────────
            elif intent_key == INTENT_TRACK_PHONE:
                phone = payload.get("phone") or payload.get("message", "")
                task_id = task_manager.create_task("track_phone", phone[:60])
                background_tasks.append({"task_id": task_id, "type": "track_phone", "label": phone[:60]})
                reply_parts.append("Looking up phone number " + phone + ".")
                asyncio.create_task(_run_phone_lookup(task_id, phone))

            # ── generate_code ─────────────────────────────────────────────
            elif intent_key == INTENT_GENERATE_CODE:
                prompt = payload.get("prompt") or payload.get("message", "a small program")
                task_id = task_manager.create_task("generate_code", prompt[:120])
                background_tasks.append({"task_id": task_id, "type": "generate_code", "label": prompt[:80]})
                reply_parts.append("Writing code for: " + prompt + ".")
                asyncio.create_task(_run_code_generation(task_id, prompt, send_email=False))

            # ── generate_code_email ───────────────────────────────────────
            elif intent_key == INTENT_GENERATE_CODE_EMAIL:
                prompt = payload.get("prompt") or payload.get("message", "a small program")
                task_id = task_manager.create_task("generate_code_email", prompt[:120])
                background_tasks.append({"task_id": task_id, "type": "generate_code_email", "label": prompt[:80]})
                reply_parts.append("Writing code for: " + prompt + " and emailing it to you.")
                asyncio.create_task(_run_code_generation(task_id, prompt, send_email=True))

        except Exception as exc:
            logger.error("[EXECUTOR] intent=%s error=%s", intent_key, exc)
            reply_parts.append("I encountered an error with that task. Please try again.")

    if wopens:         actions["wopens"]         = wopens
    if plays:          actions["plays"]           = plays
    if googlesearches: actions["googlesearches"]  = googlesearches
    if youtubesearches:actions["youtubesearches"] = youtubesearches
    if cam_action:     actions["cam"]             = cam_action

    return actions, background_tasks, ("  \n".join(reply_parts) if reply_parts else "Done!")


# ─────────────────────────────────────────────────────────────
# IMMEDIATE HELPERS
# ─────────────────────────────────────────────────────────────

def _guess_app_name(url: str) -> Optional[str]:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lstrip("www.")
        return netloc.split(".")[0].lower()
    except Exception:
        return None


def _open_reply(url: str) -> str:
    try:
        from urllib.parse import urlparse
        site = urlparse(url).netloc.lstrip("www.").split(".")[0].capitalize()
    except Exception:
        site = url
    return "Opening " + site + " for you."


def _yt_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


def _google_url(query: str) -> str:
    return "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)


async def _fetch_first_youtube_video_id(query: str) -> Optional[str]:
    try:
        import re as _re
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        url = _yt_search_url(query)
        async with httpx.AsyncClient(timeout=10.0, headers=headers, follow_redirects=True) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return None
        matches = _re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
        for vid_id in matches:
            if len(vid_id) == 11:
                return vid_id
    except Exception as exc:
        logger.warning("[PLAY] Could not fetch first video ID: %s", exc)
    return None


# ─────────────────────────────────────────────────────────────
# BACKGROUND RUNNERS
# ─────────────────────────────────────────────────────────────

async def _run_image_generation(task_id: str, prompt: str) -> None:
    try:
        task_manager.update_status(task_id, "running")
        logger.info("[IMAGE GEN] task=%s prompt=%s", task_id[:8], prompt[:60])

        encoded = urllib.parse.quote(prompt)
        seed = abs(hash(prompt + task_id)) % 999999
        image_url = (
            "https://image.pollinations.ai/prompt/" + encoded
            + "?width=1024&height=1024&seed=" + str(seed)
            + "&nologo=true&enhance=true"
        )

        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            resp = await client.get(image_url)
        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "image" not in ctype:
            raise RuntimeError("Pollinations HTTP " + str(resp.status_code))

        task_manager.complete_task(task_id, {
            "type": "image", "url": image_url,
            "prompt": prompt, "width": 1024, "height": 1024,
        })
        logger.info("[IMAGE GEN] completed task=%s", task_id[:8])
    except Exception as exc:
        logger.error("[IMAGE GEN] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))


async def _run_ppt_generation(
    task_id: str, topic: str, groq_service, send_email: bool = False
) -> None:
    try:
        task_manager.update_status(task_id, "running")
        logger.info("[PPT GEN] task=%s topic=%s", task_id[:8], topic[:60])

        from app.services.ppt_service import generate_ppt
        loop = asyncio.get_event_loop()
        ppt_path = await loop.run_in_executor(None, lambda: generate_ppt(topic, groq_service))

        result = {
            "type": "ppt",
            "filename": ppt_path.name,
            "download_url": "/download/ppt/" + ppt_path.name,
            "topic": topic,
            "emailed": False,
            "email_message": "",
        }

        if send_email:
            from app.services.email_service import send_email as _send_gmail
            body = (
                "Hi,\n\n"
                "Your Jarvis presentation on '" + topic + "' is attached.\n\n"
                "Generated by J.A.R.V.I.S.\n"
            )
            email_result = await loop.run_in_executor(
                None,
                lambda: _send_gmail(
                    subject="Your Jarvis Presentation: " + topic,
                    body=body,
                    attachment_path=ppt_path,
                ),
            )
            result["emailed"] = email_result.get("success", False)
            result["email_message"] = email_result.get("message", "")

        task_manager.complete_task(task_id, result)
        logger.info("[PPT GEN] completed task=%s file=%s", task_id[:8], ppt_path.name)
    except Exception as exc:
        logger.error("[PPT GEN] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))


async def _run_phone_lookup(task_id: str, phone: str) -> None:
    try:
        task_manager.update_status(task_id, "running")
        logger.info("[PHONE] task=%s number=%s", task_id[:8], phone)

        from app.services.phone_service import track_phone, format_phone_result_for_chat
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, lambda: track_phone(phone))
        text = format_phone_result_for_chat(info)

        task_manager.complete_task(task_id, {
            "type": "phone", "text": text, "data": info, "phone": phone,
        })
        logger.info("[PHONE] completed task=%s", task_id[:8])
    except Exception as exc:
        logger.error("[PHONE] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))


async def _run_content_generation(task_id: str, prompt: str, groq_service) -> None:
    try:
        task_manager.update_status(task_id, "running")
        logger.info("[CONTENT] task=%s prompt=%s", task_id[:8], prompt[:60])

        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None,
            lambda: groq_service.get_response(
                "Please write the following in full, with proper formatting and detail: " + prompt,
                chat_history=None,
            ),
        )
        task_manager.complete_task(task_id, {
            "type": "content", "text": content, "prompt": prompt,
        })
        logger.info("[CONTENT] completed task=%s", task_id[:8])
    except Exception as exc:
        logger.error("[CONTENT] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))


async def _run_image_email_generation(task_id: str, prompt: str) -> None:
    """Generate an image via Pollinations, download it, then email it as
    an attachment to the configured GMAIL_ADDRESS."""
    import tempfile
    from pathlib import Path

    try:
        task_manager.update_status(task_id, "running")
        logger.info("[IMAGE+EMAIL] task=%s prompt=%s", task_id[:8], prompt[:60])

        encoded = urllib.parse.quote(prompt)
        seed = abs(hash(prompt + task_id)) % 999999
        image_url = (
            "https://image.pollinations.ai/prompt/" + encoded
            + "?width=1024&height=1024&seed=" + str(seed)
            + "&nologo=true&enhance=true"
        )

        async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
            resp = await client.get(image_url)

        ctype = resp.headers.get("content-type", "")
        if resp.status_code != 200 or "image" not in ctype:
            raise RuntimeError("Pollinations HTTP " + str(resp.status_code))

        # Persist to a temp file so the email service can attach it.
        ext = ".png" if "png" in ctype else ".jpg"
        tmp_dir = Path(tempfile.gettempdir()) / "jarvis_images"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        safe_name = "jarvis_" + task_id[:8] + ext
        image_path = tmp_dir / safe_name
        image_path.write_bytes(resp.content)

        from app.services.email_service import send_email as _send_gmail
        body = (
            "Hi,\n\n"
            "Here is the image you asked J.A.R.V.I.S. to generate.\n\n"
            "Prompt: " + prompt + "\n\n"
            "— J.A.R.V.I.S.\n"
        )
        loop = asyncio.get_event_loop()
        email_result = await loop.run_in_executor(
            None,
            lambda: _send_gmail(
                subject="Your J.A.R.V.I.S. image: " + prompt[:60],
                body=body,
                attachment_path=str(image_path),
            ),
        )

        task_manager.complete_task(task_id, {
            "type": "image",
            "url": image_url,
            "prompt": prompt,
            "width": 1024,
            "height": 1024,
            "emailed": email_result.get("success", False),
            "email_message": email_result.get("message", ""),
        })
        logger.info(
            "[IMAGE+EMAIL] completed task=%s emailed=%s",
            task_id[:8], email_result.get("success", False),
        )
    except Exception as exc:
        logger.error("[IMAGE+EMAIL] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))


async def _run_code_generation(task_id: str, prompt: str, send_email: bool = False) -> None:
    """Generate professional code using CodexService and (optionally) email it."""
    from pathlib import Path

    try:
        task_manager.update_status(task_id, "running")
        logger.info("[CODEX] task=%s prompt=%s email=%s",
                    task_id[:8], prompt[:60], send_email)

        from app.services.code_service import (
            get_codex, format_code_result_for_chat, CodexService,
        )
        codex = get_codex()
        loop = asyncio.get_event_loop()

        # 1. Generate the code (blocking SDK call -> threadpool).
        result = await loop.run_in_executor(None, lambda: codex.generate_code(prompt))

        # 2. Save it under database/generated so it's downloadable.
        out_dir = Path(__file__).resolve().parent.parent.parent / "database" / "generated" / "code"
        code_path = await loop.run_in_executor(
            None, lambda: CodexService.save_to_file(result, out_dir)
        )

        payload = {
            "type": "code",
            "language": result["language"],
            "filename": code_path.name,
            "code": result["code"],
            "explanation": result["explanation"],
            "dependencies": result["dependencies"],
            "model": result.get("model"),
            "download_url": "/download/code/" + code_path.name,
            "text": format_code_result_for_chat(result),
            "emailed": False,
            "email_message": "",
        }

        # 3. Optionally email it as an attachment.
        if send_email:
            from app.services.email_service import send_email as _send_gmail
            body = (
                "Hi,\n\n"
                "Here is the code J.A.R.V.I.S. generated for you.\n\n"
                "Task: " + prompt + "\n"
                "Language: " + result["language"] + "\n"
                "Model: " + str(result.get("model")) + "\n"
                + ("Dependencies: " + ", ".join(result["dependencies"]) + "\n"
                   if result["dependencies"] else "")
                + "\nExplanation:\n" + (result["explanation"] or "(none)") + "\n\n"
                + "— J.A.R.V.I.S.\n"
            )
            email_result = await loop.run_in_executor(
                None,
                lambda: _send_gmail(
                    subject="Your J.A.R.V.I.S. code: " + prompt[:60],
                    body=body,
                    attachment_path=str(code_path),
                ),
            )
            payload["emailed"] = email_result.get("success", False)
            payload["email_message"] = email_result.get("message", "")

        task_manager.complete_task(task_id, payload)
        logger.info("[CODEX] completed task=%s file=%s", task_id[:8], code_path.name)
    except Exception as exc:
        logger.error("[CODEX] failed task=%s: %s", task_id[:8], exc)
        task_manager.fail_task(task_id, str(exc))
