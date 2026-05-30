from typing import List, Optional

import logging

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from config import GROQ_API_KEYS, GROQ_MODEL, JARVIS_SYSTEM_PROMPT
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information

logger = logging.getLogger("J.A.R.V.I.S")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def escape_curly_braces(text: str) -> str:
    """
    Escape curly braces for LangChain templates.
    """
    if not text:
        return text

    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Check whether the exception is a rate limit error.
    """
    msg = str(exc).lower()

    return (
        "429" in str(exc)
        or "rate limit" in msg
        or "tokens per day" in msg
    )


def _mask_api_key(key: str) -> str:
    """
    Mask API key for safe logging.
    """
    if not key or len(key) <= 12:
        return "***masked***"

    return f"{key[:8]}...{key[-4:]}"


# ============================================================================
# GROQ SERVICE
# ============================================================================

class GroqService:

    # Shared round-robin counter
    _shared_key_index = 0
    _lock = None

    def __init__(
        self,
        vector_store_service: VectorStoreService
    ):

        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured."
            )

        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model=GROQ_MODEL,
                temperature=0.8,
            )
            for key in GROQ_API_KEYS
        ]

        self.vector_store_service = vector_store_service

        logger.info(
            f"Initialized GroqService with "
            f"{len(GROQ_API_KEYS)} API key(s)"
        )

    # =========================================================================
    # INTERNAL LLM INVOCATION
    # =========================================================================

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str,
    ) -> str:

        """
        Invoke Groq model using round-robin API keys.
        """

        n = len(self.llms)

        start_i = (
            GroqService._shared_key_index % n
        )

        current_key_index = (
            GroqService._shared_key_index
        )

        GroqService._shared_key_index += 1

        masked_key = _mask_api_key(
            GROQ_API_KEYS[start_i]
        )

        logger.info(
            f"Using API key "
            f"#{start_i + 1}/{n} "
            f"({masked_key}) "
            f"(round-robin index: "
            f"{current_key_index})"
        )

        last_exc = None
        keys_tried = []

        # Try all keys
        for j in range(n):

            i = (start_i + j) % n
            keys_tried.append(i)

            try:

                chain = prompt | self.llms[i]

                response = chain.invoke({
                    "history": messages,
                    "question": question,
                })

                # Fallback success logging
                if j > 0:

                    masked_success_key = (
                        _mask_api_key(
                            GROQ_API_KEYS[i]
                        )
                    )

                    logger.warning(
                        f"Fallback successful "
                        f"with key #{i + 1}: "
                        f"{masked_success_key}"
                    )

                return response.content

            except Exception as e:

                last_exc = e

                masked_failed_key = (
                    _mask_api_key(
                        GROQ_API_KEYS[i]
                    )
                )

                if _is_rate_limit_error(e):

                    logger.warning(
                        f"Rate limit hit on "
                        f"key #{i + 1}: "
                        f"{masked_failed_key}"
                    )

                else:

                    logger.error(
                        f"API key #{i + 1} failed "
                        f"({masked_failed_key}): "
                        f"{str(e)}"
                    )

                # Try next key if available
                if j < n - 1:
                    continue

        # All keys failed
        masked_all_keys = ", ".join([
            _mask_api_key(
                GROQ_API_KEYS[i]
            )
            for i in keys_tried
        ])

        logger.critical(
            f"All API keys failed. "
            f"Tried keys: {masked_all_keys}"
        )

        raise Exception(
            f"Error getting response from Groq: "
            f"{str(last_exc)}"
        ) from last_exc

    # =========================================================================
    # MAIN RESPONSE FUNCTION
    # =========================================================================

    def get_response(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
    ) -> str:

        """
        Generate assistant response.
        """

        try:

            context = ""

            # ================================================================
            # VECTOR SEARCH
            # ================================================================

            try:

                retriever = (
                    self.vector_store_service
                    .get_retriever(k=10)
                )

                context_docs = retriever.get_relevant_documents(
                    question
                )

                if context_docs:

                    context = "\n".join([
                        doc.page_content
                        for doc in context_docs
                    ])

            except Exception as retrieval_err:

                logger.warning(
                    "Vector retrieval failed: %s",
                    retrieval_err
                )

            # ================================================================
            # SYSTEM MESSAGE
            # ================================================================

            time_info = get_time_information()

            system_message = (
                JARVIS_SYSTEM_PROMPT
                + f"\n\nCurrent time and date:\n"
                + time_info
            )

            if context:

                system_message += (
                    "\n\nRelevant context:\n"
                    + escape_curly_braces(context)
                )

            # ================================================================
            # PROMPT TEMPLATE
            # ================================================================

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),

                MessagesPlaceholder(
                    variable_name="history"
                ),

                ("human", "{question}"),
            ])

            # ================================================================
            # CHAT HISTORY
            # ================================================================

            messages = []

            if chat_history:

                for human_msg, ai_msg in chat_history:

                    messages.append(
                        HumanMessage(
                            content=human_msg
                        )
                    )

                    messages.append(
                        AIMessage(
                            content=ai_msg
                        )
                    )

            # ================================================================
            # LLM CALL
            # ================================================================

            return self._invoke_llm(
                prompt,
                messages,
                question
            )

        except Exception as e:

            logger.exception(
                "GroqService failed"
            )

            raise Exception(
                f"Error getting response "
                f"from Groq: {str(e)}"
            ) from e