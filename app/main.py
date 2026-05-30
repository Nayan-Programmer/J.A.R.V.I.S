from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

from app.models import ChatRequest, ChatResponse, StreamRequest

RATE_LIMIT_MESSAGE = (
    "You've reached your daily API limit for this assistant. "
    "Your credits will reset in a few hours, or you can upgrade your plan for more. "
    "Please try again later."
)


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in str(exc) or "rate limit" in msg or "tokens per day" in msg


from app.services.vector_store import VectorStoreService
from app.services.groq_service import GroqService
from app.services.realtime_service import RealtimeGroqService
from app.services.chat_service import ChatService
from app.services.vision_service import VisionService
from app.services.brain_service import BrainService
from app.services.task_executor import execute_tasks
from app.services.task_manager import task_manager
from app.utils.key_rotation import init_rotator
from app.utils.app_launcher import launch_app
from config import VECTOR_STORE_DIR, GROQ_API_KEYS


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("J.A.R.V.I.S")

# ── Global service references ─────────────────────────────────────────────────
vector_store_service: VectorStoreService = None
groq_service: GroqService = None
realtime_service: RealtimeGroqService = None
chat_service: ChatService = None
vision_service: VisionService = None
brain_service: BrainService = None

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# TTS settings from .env
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-JennyNeural")
TTS_RATE  = os.getenv("TTS_RATE",  "+22%")


# ==============================================================================
# TTS HELPER
# ==============================================================================

async def _tts_to_base64(text: str) -> Optional[str]:
    """
    Convert text to MP3 audio via edge-tts (Microsoft Neural TTS, free).
    Returns base64-encoded MP3 string, or None on failure.
    """
    if not text or not text.strip():
        return None
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text.strip(), TTS_VOICE, rate=TTS_RATE)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        audio_bytes = buf.getvalue()
        if not audio_bytes:
            return None
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as exc:
        logger.warning("[TTS] Generation failed: %s", exc)
        return None


# ==============================================================================
# APP LIFESPAN
# ==============================================================================

def print_title():
    title = """
    ╔══════════════════════════════════════════════════════════╗
   ║                                                          ║
   ║         ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗          ║
   ║         ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝          ║
   ║         ██║███████║██████╔╝██║   ██║██║███████╗          ║
   ║    ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║          ║
   ║    ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║          ║
   ║     ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝          ║
   ║                                                          ║
   ║          Just A Rather Very Intelligent System           ║
   ║                                                          ║
   ╚══════════════════════════════════════════════════════════╝
    """
    print(title)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store_service, groq_service, realtime_service
    global chat_service, vision_service, brain_service

    print_title()
    logger.info("=" * 60)
    logger.info("J.A.R.V.I.S - Starting Up...")
    logger.info("=" * 60)

    try:
        if GROQ_API_KEYS:
            init_rotator(len(GROQ_API_KEYS))

        logger.info("Initializing vector store service...")
        vector_store_service = VectorStoreService()
        vector_store_service.create_vector_store()
        logger.info("Vector store initialized successfully")

        logger.info("Initializing Groq service (general queries)...")
        groq_service = GroqService(vector_store_service)
        logger.info("Groq service initialized successfully")

        logger.info("Initializing Realtime Groq service (with Tavily search)...")
        realtime_service = RealtimeGroqService(vector_store_service)
        logger.info("Realtime Groq service initialized successfully")

        logger.info("Initializing chat service...")
        chat_service = ChatService(groq_service, realtime_service)
        logger.info("Chat service initialized successfully")

        logger.info("Initializing vision service...")
        vision_service = VisionService()
        logger.info("Vision service initialized successfully")

        logger.info("Initializing brain service...")
        brain_service = BrainService(groq_service=groq_service)
        logger.info("Brain service initialized successfully")

        # Test edge-tts import at startup so we warn early if missing
        try:
            import edge_tts  # noqa: F401
            logger.info(" - TTS (edge-tts): Ready  [voice: %s, rate: %s]", TTS_VOICE, TTS_RATE)
        except ImportError:
            logger.warning(" - TTS: edge-tts not installed. Run: pip install edge-tts")

        logger.info("=" * 60)
        logger.info("J.A.R.V.I.S is online!")
        logger.info("Open: http://localhost:8000")
        logger.info("Docs: http://localhost:8000/docs")
        logger.info("=" * 60)

        yield

        logger.info("Shutting down J.A.R.V.I.S...")
        if chat_service:
            for sid in list(chat_service.sessions.keys()):
                chat_service.save_chat_session(sid)
        logger.info("All sessions saved. Goodbye!")

    except Exception as e:
        logger.error(f"Fatal error during startup: {e}", exc_info=True)
        raise


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="J.A.R.V.I.S API",
    description="Just A Rather Very Intelligent System",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# SSE HELPER
# ==============================================================================

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ==============================================================================
# MAIN STREAMING ENDPOINT  ─  POST /chat/jarvis/stream
# ==============================================================================

CAM_BYPASS_TOKEN = "__CAM_FRAME_ATTACHED__"


@app.post("/chat/jarvis/stream")
async def chat_jarvis_stream(request: StreamRequest):
    """
    Single SSE streaming endpoint for all Jarvis interactions.

    SSE event types emitted:
      {"session_id": "..."}           — first event always
      {"activity": {...}}             — routing/processing status
      {"chunk": "text"}               — streamed reply tokens
      {"audio": "<base64_mp3>"}       — TTS audio (only when request.tts=true)
      {"actions": {...}}              — browser-side actions
      {"background_tasks": [...]}     — background tasks to poll
      {"done": true}                  — stream complete
      {"error": "message"}            — error
    """
    if not chat_service or not brain_service:
        raise HTTPException(status_code=503, detail="Services not initialized")

    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    has_image = bool(request.imgbase64) or CAM_BYPASS_TOKEN in message
    clean_message = message.replace(CAM_BYPASS_TOKEN, "").strip()
    image_b64 = request.imgbase64 or None
    want_tts = bool(request.tts)

    async def generate():
        try:
            # ── Session ────────────────────────────────────────────────────
            session_id = chat_service.get_or_create_session(request.session_id)
            yield _sse({"session_id": session_id})

            chat_history = chat_service.format_history_for_llm(session_id)

            # ── Route decision ──────────────────────────────────────────────
            if has_image and image_b64:
                route = "camera"
                task_types = []
            else:
                route, task_types, method, ms = brain_service.classify(
                    clean_message, chat_history
                )

            yield _sse({"activity": {
                "event": "routing",
                "route": route,
                "detail": clean_message[:80],
            }})

            # ── CAMERA / VISION ─────────────────────────────────────────────
            if route == "camera" and image_b64:
                yield _sse({"activity": {
                    "event": "vision_analyzing",
                    "message": "Analyzing image...",
                }})
                full_reply = ""
                for chunk in vision_service.analyze_image_stream(
                    clean_message or "What do you see?", image_b64, chat_history
                ):
                    full_reply += chunk
                    yield _sse({"chunk": chunk})
                    await asyncio.sleep(0)

                chat_service.add_message(session_id, "user", clean_message or "What do you see?")
                chat_service.add_message(session_id, "assistant", full_reply)
                chat_service.save_chat_session(session_id)

                if want_tts and full_reply:
                    audio_b64 = await _tts_to_base64(full_reply)
                    if audio_b64:
                        yield _sse({"audio": audio_b64})

                yield _sse({"done": True})
                return

            # ── TASK ────────────────────────────────────────────────────────
            if route == "task":
                yield _sse({"activity": {
                    "event": "tasks_executing",
                    "message": f"Executing: {task_types}",
                }})
                intents = brain_service.extract_task_payloads(
                    clean_message, task_types, chat_history
                )
                actions, bg_tasks, reply_text = await execute_tasks(
                    intents, groq_service=groq_service, session_id=session_id
                )
                if actions:
                    yield _sse({"actions": actions})
                if bg_tasks:
                    yield _sse({"background_tasks": bg_tasks})
                yield _sse({"chunk": reply_text})

                chat_service.add_message(session_id, "user", clean_message)
                chat_service.add_message(session_id, "assistant", reply_text)
                chat_service.save_chat_session(session_id)

                if want_tts and reply_text:
                    audio_b64 = await _tts_to_base64(reply_text)
                    if audio_b64:
                        yield _sse({"audio": audio_b64})

                yield _sse({"done": True})
                return

            # ── MIXED (task + question) ─────────────────────────────────────
            if route == "mixed":
                yield _sse({"activity": {
                    "event": "tasks_executing",
                    "message": "Mixed: executing tasks + chat",
                }})
                intents = brain_service.extract_task_payloads(
                    clean_message, task_types, chat_history
                )
                actions, bg_tasks, _ = await execute_tasks(
                    intents, groq_service=groq_service, session_id=session_id
                )
                if actions:
                    yield _sse({"actions": actions})
                if bg_tasks:
                    yield _sse({"background_tasks": bg_tasks})
                # Fall through to general chat for the text reply

            # ── REALTIME ────────────────────────────────────────────────────
            if route == "realtime":
                yield _sse({"activity": {
                    "event": "searching_web",
                    "message": "Searching the web...",
                }})
                loop = asyncio.get_event_loop()
                response_text = await loop.run_in_executor(
                    None,
                    lambda: realtime_service.get_response(clean_message, chat_history),
                )
                for word in response_text.split(" "):
                    yield _sse({"chunk": word + " "})
                    await asyncio.sleep(0.01)

                chat_service.add_message(session_id, "user", clean_message)
                chat_service.add_message(session_id, "assistant", response_text)
                chat_service.save_chat_session(session_id)

                if want_tts and response_text:
                    audio_b64 = await _tts_to_base64(response_text)
                    if audio_b64:
                        yield _sse({"audio": audio_b64})

                yield _sse({"done": True})
                return

            # ── GENERAL (default / mixed fallthrough) ───────────────────────
            yield _sse({"activity": {
                "event": "streaming_started",
                "route": route,
            }})
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(
                None,
                lambda: groq_service.get_response(clean_message, chat_history),
            )
            for word in response_text.split(" "):
                yield _sse({"chunk": word + " "})
                await asyncio.sleep(0.01)

            chat_service.add_message(session_id, "user", clean_message)
            chat_service.add_message(session_id, "assistant", response_text)
            chat_service.save_chat_session(session_id)

            if want_tts and response_text:
                audio_b64 = await _tts_to_base64(response_text)
                if audio_b64:
                    yield _sse({"audio": audio_b64})

            yield _sse({"done": True})

        except Exception as exc:
            logger.error("[STREAM] Error: %s", exc, exc_info=True)
            if _is_rate_limit_error(exc):
                yield _sse({"error": RATE_LIMIT_MESSAGE})
            else:
                yield _sse({"error": "Something went wrong. Please try again."})
            yield _sse({"done": True})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ==============================================================================
# TASK POLLING  ─  GET /tasks/{task_id}
# ==============================================================================

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    return JSONResponse(task_manager.to_response(task_id))


@app.get("/download/ppt/{filename}")
async def download_ppt(filename: str):
    import re as _re
    from pathlib import Path as _P
    safe = _re.sub(r"[^a-zA-Z0-9_\-.]", "", filename)
    out  = _P(__file__).parent.parent / "database" / "generated" / safe
    if not out.exists() or out.suffix.lower() != ".pptx":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(out),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=safe,
    )


@app.get("/download/code/{filename}")
async def download_code(filename: str):
    """Download a Codex-generated source file."""
    import re as _re
    from pathlib import Path as _P
    safe = _re.sub(r"[^a-zA-Z0-9_\-.]", "", filename)
    out = _P(__file__).parent.parent / "database" / "generated" / "code" / safe
    if not out.exists() or not out.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        str(out),
        media_type="text/plain; charset=utf-8",
        filename=safe,
    )


# ==============================================================================
# DESKTOP APP LAUNCHER  ─  POST /open-app
# ==============================================================================

@app.post("/open-app")
async def open_desktop_app(body: dict):
    """
    Launch a desktop application on the user's local PC.
    Called by the frontend when the user asks to open an app by name.

    Body: { "name": "whatsapp" }
    Returns: { "launched": true/false, "message": "..." }
    """
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"launched": False, "message": "No app name provided"})
    loop = asyncio.get_event_loop()
    launched, message = await loop.run_in_executor(None, lambda: launch_app(name))
    logger.info("[OPEN-APP] name=%r launched=%s message=%s", name, launched, message)
    return JSONResponse({"launched": launched, "message": message})


# ==============================================================================
# CLASSIC CHAT ENDPOINTS (non-streaming, backward compat)
# ==============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        return ChatResponse(response=response_text, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/chat/realtime", response_model=ChatResponse)
async def chat_realtime(request: ChatRequest):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        session_id = chat_service.get_or_create_session(request.session_id)
        response_text = chat_service.process_realtime_message(session_id, request.message)
        chat_service.save_chat_session(session_id)
        return ChatResponse(response=response_text, session_id=session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        if _is_rate_limit_error(e):
            raise HTTPException(status_code=429, detail=RATE_LIMIT_MESSAGE)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    try:
        messages = chat_service.get_chat_history(session_id)
        return {
            "session_id": session_id,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# HEALTH
# ==============================================================================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vector_store":    vector_store_service is not None,
        "groq_service":    groq_service is not None,
        "realtime_service": realtime_service is not None,
        "chat_service":    chat_service is not None,
        "vision_service":  vision_service is not None,
        "brain_service":   brain_service is not None,
    }


# ==============================================================================
# FRONTEND  ─  /app/viewer.html + static mount
# ==============================================================================

@app.get("/app/viewer.html")
async def viewer_html():
    viewer = FRONTEND_DIR / "viewer.html"
    if viewer.exists():
        return FileResponse(str(viewer), media_type="text/html")
    raise HTTPException(status_code=404, detail="viewer.html not found")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ==============================================================================
# STANDALONE RUN
# ==============================================================================

def run():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    run()
