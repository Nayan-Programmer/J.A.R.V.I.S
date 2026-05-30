from pydantic import BaseModel, Field
from typing import List, Optional


# ==============================================================================
# MESSAGE AND REQUEST/RESPONSE MODELS
# ==============================================================================

class ChatMessage(BaseModel):
    """
    A single message in a conversation (user or assistant).

    Stored in order inside a session. No timestamp; order defines chronology.
    """
    role: str      # Either "user" (human) or "assistant" (Jarvis)
    content: str   # The message text.

class ChatRequest(BaseModel):
    """
    Request body for POST /chat and POST /chat/realtime.

    - message: Required. The user's question or message. Must be 1-32,000 characters
      (validated by Pydantic; empty or too long returns 422).
    - session_id: Optional. If omitted, the server creates a new session and returns
      its ID. If provided, the server uses it (and loads from disk if that session exists).
    """
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """
    Response body for POST /chat and POST /chat/realtime.

    - response: The assistant's reply text.
    - session_id: The session this message belongs to; send it on the next request to continue.
    """
    response: str
    session_id: str


class ChatHistory(BaseModel):
    """
    Internal model representing the full conversation: session id plus ordered list of messages.
    Used when saving a session to disk (chat_service serializes this to JSON).
    """
    session_id: str
    messages: List[ChatMessage]

class StreamRequest(BaseModel):
    """Request body for POST /chat/jarvis/stream (SSE streaming endpoint)."""
    message: str = Field(..., min_length=1, max_length=32_000)
    session_id: str | None = None
    tts: bool = False
    imgbase64: str | None = None
