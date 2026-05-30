"""
VISION SERVICE (ADVANCED)
=========================

Real multimodal vision using Groq's Llama 4 / Llama 3.2 vision models.
The image is sent as a base64 data URL in the OpenAI-compatible
multimodal `content` array — the model actually SEES the image.

Falls back across:
  1. meta-llama/llama-4-scout-17b-16e-instruct       (best, multimodal)
  2. meta-llama/llama-4-maverick-17b-128e-instruct  (multimodal)
  3. llama-3.2-90b-vision-preview                    (legacy vision)
  4. llama-3.2-11b-vision-preview                    (smaller fallback)

Across every available GROQ_API_KEY.
"""

import base64
import logging
import re
from typing import Generator, Optional, List, Tuple

from groq import Groq
from config import GROQ_API_KEYS, JARVIS_SYSTEM_PROMPT
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")

# Ordered best -> fallback. Groq retires/renames vision models periodically;
# we try several so the service degrades gracefully.
VISION_MODELS: List[str] = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
]

_VISION_SYSTEM = (
    JARVIS_SYSTEM_PROMPT
    + "\n\nYou are now in VISION mode. The user has shared an image with you. "
    "Analyse it carefully and describe exactly what you see. "
    "If the user asks a specific question, answer it using visual evidence "
    "from the image. Be concise, accurate, and never hallucinate details "
    "that are not actually visible."
)

# Soft cap on image size we forward (Groq enforces its own limits).
_MAX_IMAGE_BYTES = 4 * 1024 * 1024  # 4 MB after decoding


class VisionService:
    def __init__(self) -> None:
        if not GROQ_API_KEYS:
            raise RuntimeError("No GROQ_API_KEY found in environment.")

        self.clients: List[Groq] = [Groq(api_key=k) for k in GROQ_API_KEYS]
        logger.info(
            "[VISION] Initialised | keys=%d | models=%s",
            len(self.clients),
            VISION_MODELS,
        )

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------
    def analyze_image_stream(
        self,
        message: str,
        image_base64: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Generator[str, None, None]:
        data_url = self._normalize_image(image_base64)
        if not data_url:
            yield "I couldn't read that image. Please try sending it again."
            return

        history_msgs = self._build_history(chat_history)
        user_question = (message or "").strip() or "What do you see in this image?"

        time_info = get_time_information()
        system_msg = f"{_VISION_SYSTEM}\n\nCurrent time: {time_info}"

        multimodal_user = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }

        last_err: Optional[Exception] = None

        for key_idx, client in enumerate(self.clients):
            for model in VISION_MODELS:
                try:
                    logger.info(
                        "[VISION] try key#%d model=%s", key_idx + 1, model
                    )
                    stream = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_msg},
                            *history_msgs,
                            multimodal_user,
                        ],
                        max_tokens=1024,
                        temperature=0.5,
                        stream=True,
                    )

                    produced_any = False
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            produced_any = True
                            yield delta.content

                    if produced_any:
                        logger.info(
                            "[VISION] OK key#%d model=%s", key_idx + 1, model
                        )
                        return
                    # else: empty completion -> try next model
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    msg = str(exc).lower()
                    logger.warning(
                        "[VISION] key#%d model=%s failed: %s",
                        key_idx + 1,
                        model,
                        exc,
                    )
                    # If the model doesn't exist on this key, just try the next.
                    if any(
                        s in msg
                        for s in (
                            "model_decommissioned",
                            "model not found",
                            "does not exist",
                            "decommissioned",
                            "invalid model",
                        )
                    ):
                        continue
                    # Auth or quota: skip the rest of this key's models.
                    if any(
                        s in msg
                        for s in (
                            "invalid api key",
                            "unauthorized",
                            "401",
                            "rate limit",
                            "quota",
                        )
                    ):
                        break
                    continue

        logger.error("[VISION] All vision models failed: %s", last_err)
        yield (
            "I'm having trouble analysing the image right now "
            "(all vision models failed). Please try again in a moment."
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_history(
        chat_history: Optional[List[Tuple[str, str]]],
    ) -> List[dict]:
        msgs: List[dict] = []
        if not chat_history:
            return msgs
        for user_msg, ai_msg in chat_history[-6:]:
            if user_msg:
                msgs.append({"role": "user", "content": str(user_msg)})
            if ai_msg:
                msgs.append({"role": "assistant", "content": str(ai_msg)})
        return msgs

    @staticmethod
    def _normalize_image(image_base64: str) -> Optional[str]:
        """Return a clean `data:image/...;base64,XXXX` URL, or None."""
        if not image_base64 or not isinstance(image_base64, str):
            return None

        raw = image_base64.strip()
        mime = "image/jpeg"

        if raw.startswith("data:"):
            m = re.match(r"data:(image/[a-zA-Z0-9.+-]+);base64,(.*)", raw, re.S)
            if not m:
                return None
            mime, payload = m.group(1), m.group(2)
        else:
            payload = raw

        # Strip whitespace/newlines that often sneak in from the frontend.
        payload = re.sub(r"\s+", "", payload)

        try:
            decoded = base64.b64decode(payload, validate=False)
        except Exception:
            return None

        if not decoded:
            return None

        if len(decoded) > _MAX_IMAGE_BYTES:
            # Re-encode the original payload but warn — Groq may still reject.
            logger.warning(
                "[VISION] Large image (%.1f MB) — may exceed model limit",
                len(decoded) / (1024 * 1024),
            )

        return f"data:{mime};base64,{payload}"
