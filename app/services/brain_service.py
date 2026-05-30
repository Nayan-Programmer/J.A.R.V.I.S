import logging
import re
import time
from typing import List, Optional, Tuple, Literal
from config import GROQ_API_KEYS, INTENT_CLASSIFY_MODEL

logger = logging.getLogger("J.A.R.V.I.S")

CategoryType = Literal["general", "realtime", "camera", "task"]
ALL_CATEGORIES: List[str] = ["general", "realtime", "camera", "task", "mixed"]

TaskType = Literal[
    "open", "play", "generate_image", "content",
    "google_search", "youtube_search",
    "open_webcam", "close_webcam",
]

ALL_TASK_TYPES: List[str] = [
    "open", "play", "generate_image", "content",
    "google_search", "youtube_search",
    "open_webcam", "close_webcam",
]

MAX_CONTEXT_TURNS = 6
MAX_MESSAGE_PREVIEW = 600

_PRIMARY_BRAIN_PROMPT = """You are the decision-maker for JARVIS. Classify the user's message into EXACTLY ONE category.

=== CATEGORIES ===

**camera** — User wants to ANALYZE, IDENTIFY, or SEE something visual. They are holding, showing, or displaying something and want you to look at it.
Examples: "What is this?" / "What am I holding?" / "What do you see?" / "Describe what I'm showing" / "Identify this" / "What's in my hand?" / "Look at this" / "Read this" / "Can you see this?" / "Check this out"
- Any request where the user expects you to LOOK at something through the camera → camera

**task** — User wants ONLY an ACTION performed (no question to answer). Opening apps/websites, playing music/video, generating images, writing content, searching Google/YouTube, or controlling the webcam.
Examples: "Open YouTube" / "Play despacito" / "Generate image of a cat" / "Write an essay about AI" / "Search for Python tutorials" / "Open webcam" / "Close webcam" / "Launch Netflix" / "Go to Facebook" / "Make me a picture of a sunset" / "Draw a cat" / "Create an image of mountains"
- ANY request to open, launch, play, generate, draw, create, write, draft, compose, search, or control webcam → task
- "Open webcam" / "Turn on camera" / "Close webcam" / "Turn off camera" → task
- Image/picture/drawing requests → task (NOT camera)

**mixed** — User's message contains BOTH a conversational question AND a SPECIFIC EXPLICIT task (open/play/generate/write) in the SAME message.
Examples: "What is machine learning? Also generate an image of a neural network" / "Tell me about Python and open YouTube" / "How does AI work? And write me an essay about it"
- ONLY use mixed when the message has BOTH a question AND a clear action verb (open, play, generate, draw, write, search Google/YouTube)
- "search on the internet" / "look it up" / "find out" are NOT tasks — they just mean the user wants information → realtime
- If the user is just asking for information (even if they say "search"), use realtime, NOT mixed
- When in doubt between mixed and realtime → realtime

**realtime** — User needs CURRENT, LIVE, or RECENT information that requires web search.
Examples: "Who is Elon Musk?" / "Latest news" / "What's the weather?" / "Current stock price" / "Today's headlines" / "Tell me about [famous person]" / "What happened in [event]?" / "How much does X cost?" / "Reviews of X" / "Best restaurants near me"
- Questions about PEOPLE (who is, tell me about), EVENTS (what happened), PRICES, REVIEWS, NEWS → realtime
- Questions about anything that changes over time or needs up-to-date info → realtime
- When unsure if your knowledge is current enough → realtime (PREFER realtime over general for factual questions)

**general** — Chat from knowledge only. No web search needed. ONLY for: greetings, casual chat, opinions, advice, coding help, math, static facts, personal questions about the user's stored data.
Examples: "Hello" / "Tell me a joke" / "What is 2+2?" / "What is the capital of France?" / "How do I improve my coding?" / "Do you know my website?" / "What's the link of my website?" / "Thanks" / "You're funny" / "How are you?"
- Greetings, casual chat, opinions, personal advice → general
- Questions answerable from knowledge or stored user data → general
- "Do you know my X?" / "What's my X?" → general (answer from stored data, NOT web search)
- Static, unchanging facts (math, geography, definitions) → general

=== CONTEXTUAL INTELLIGENCE ===
CRITICAL: You MUST read the conversation history to understand context.

**Corrections & Clarifications** — When the user corrects a previous response or clarifies something:
- "No I said X not Y" / "It's X not Y" / "I meant X" / "Not that one, the other one" / "That's wrong" → classify the SAME category as the original request
- Example: User said "open jarvis4everyone.com", assistant opened wrong site, user says "it's integer 4 not f-o-r" → STILL a task
- Example: User asked a question, got wrong answer, says "no, I meant..." → STILL general/realtime

**Follow-ups** — When the user continues a topic from previous messages:
- "What about X?" / "And also..." / "Can you also..." / "More details" / "Elaborate" → classify based on what the follow-up is ASKING FOR, but consider prior context
- "Do that again" / "Try again" / "One more time" / "Another one" → SAME category as the previous request

**References to previous messages**:
- "The one I just mentioned" / "Like I said" / "That website" / "That song" → resolve from conversation history, classify based on the actual intent

=== DISAMBIGUATION RULES ===
- "Draw/Generate/Create [X]" → task (image generation), NOT camera
- "Make/Create a presentation/PPT/slides" → task
- "Track/lookup/find info on phone number [X]" → task
- "What is this?" / "What am I holding?" (no explicit generation request) → camera
- "What is [concept]?" (asking about a topic, not pointing at something) → general or realtime
- "Tell me about [person/company/event]" → realtime (needs current info)
- "Tell me about [concept/theory]" → general (static knowledge)
- "How to [do something]" → general (advice/tutorial)
- "How is [something] doing?" / "How is [person]?" → realtime (needs current status)

=== RULES ===
- Output EXACTLY ONE word: general, realtime, camera, task, or mixed
- Nothing else. No explanation. Just the category name.
- Tasks (open, play, generate, write, search, webcam) ALONE → task
- Question + task in SAME message → mixed
- Corrections/clarifications → SAME category as the original request they are correcting
- When in doubt between general and realtime → realtime
- When in doubt between general and task → check if an ACTION is requested"""


_TASK_BRAIN_PROMPT = """You are a very accurate Decision-Making Model for JARVIS. You decide what kind of task the user wants and extract the CLEAN query/topic for each task.

*** Do not answer any query, just decide what kind of task and extract the accurate query/topic. ***

=== OUTPUT FORMAT ===
Respond with: task_type clean_query
For multiple tasks, separate with commas: task_type1 query1, task_type2 query2

=== TASK TYPES & EXAMPLES ===

-> 'open (website/app name)' — Open a website or app.
   "Open YouTube" → open youtube
   "Go to Facebook and Instagram" → open facebook, open instagram
   "Open jarvis4everyone.com" → open jarvis4everyone.com
   "Launch Netflix" → open netflix

-> 'open_webcam' — Turn on camera/webcam.
   "Open webcam" / "Turn on camera" / "Start the camera" → open_webcam

-> 'close_webcam' — Turn off camera/webcam.
   "Close webcam" / "Turn off camera" / "Stop camera" → close_webcam

-> 'play (song/video name)' — Play music or video on YouTube.
   "Play Dhurandhar title track" → play Dhurandhar title track
   "Hello Jarvis can you play Shape of You on YouTube" → play Shape of You
   "Hey Jarvis Teja song Dhurandhar title track can you play that on YouTube" → play Teja Dhurandhar title track
   "Play some relaxing music" → play relaxing music
   "Put on some jazz" → play jazz music

-> 'generate_image (image prompt)' — Generate/draw/create an image or picture.
   "Generate image of a cat" → generate_image a cat
   "Draw a sunset over mountains" → generate_image sunset over mountains
   "Can you make me a picture of Iron Man" → generate_image Iron Man
   "Create an image of a futuristic city" → generate_image futuristic city
   "Make a logo for my brand" → generate_image logo design
   IMPORTANT: The prompt should be DESCRIPTIVE. Keep the full visual description, add detail if user was vague.

-> 'generate_image_email (image prompt)' — Generate an image AND email it. ONLY if user says send/email/mail.
   "Generate image of a tiger and email it to me" → generate_image_email a tiger
   "Draw a sunset and send it on my email" → generate_image_email a sunset

-> 'content (topic)' — Write content (essay, poem, letter, code, email, etc.)
   "Write an essay about AI" → content essay about AI
   "Draft a leave application" → content leave application
   "Can you write me a poem about love" → content poem about love
   "Write Python code for sorting" → content Python code for sorting algorithm
   "Write an email to my boss" → content professional email to boss

-> 'google_search (search topic)' — Search something on Google.
   "Search for Python tutorials" → google_search Python tutorials
   "Google what is quantum computing" → google_search what is quantum computing
   "Hey Jarvis look up best restaurants nearby" → google_search best restaurants nearby

-> 'youtube_search (search topic)' — Search on YouTube (NOT play, just search).
   "Search YouTube for cooking recipes" → youtube_search cooking recipes
   "Find videos about machine learning on YouTube" → youtube_search machine learning

-> 'generate_ppt (presentation topic)' — Create a PowerPoint/slides. No email.
   "Make a presentation about climate change" → generate_ppt climate change
   "Create slides on machine learning" → generate_ppt machine learning
   "Build a PPT about AI" → generate_ppt AI

-> 'generate_ppt_email (presentation topic)' — Create PPT AND email/send it. ONLY if user says send/email.
   "Make a PPT on AI and send to my email" → generate_ppt_email AI
   "Create slides on climate change and email me" → generate_ppt_email climate change

-> 'track_phone (phone number)' — Look up / track / find info about a phone number.
   "Track phone number +1 800 555 1234" → track_phone +18005551234
   "What country is +44 7911 123456 from?" → track_phone +447911123456
   "Lookup 09876543210" → track_phone 09876543210

-> 'generate_code (coding task)' — Write professional code / a program / a script / a function / a component.
   ONLY use this when the user clearly wants RUNNABLE CODE (program, script, function, class, component, endpoint, API, CRUD, scraper, bot, algorithm, snippet).
   Keep the full task description as the query.
   "Write a python script to scrape hacker news" → generate_code python script to scrape hacker news
   "Build a React dark mode toggle in typescript" → generate_code React dark mode toggle in typescript
   "Give me a fastapi crud endpoint for users" → generate_code fastapi crud endpoint for users
   "Code a binary search in C++" → generate_code binary search in C++
   "Write me a discord bot in node" → generate_code discord bot in node
   "Make a function that reverses a linked list in Java" → generate_code function that reverses a linked list in Java

-> 'generate_code_email (coding task)' — Generate code AND email it. ONLY if user says send/email/mail.
   "Write a python web scraper and email it to me" → generate_code_email python web scraper
   "Code a todo app in react and send it on my email" → generate_code_email todo app in react


=== CONTEXTUAL CORRECTIONS ===
When the user is correcting a previous task, use conversation history to understand the original intent:
- "it's integer 4 not f-o-r" (after "open jarvisforeveryone.com") → open jarvis4everyone.com
- "no, the other one" (after "play X") → play (corrected song name from context)
- "not that song, I meant the remix" → play (original song name remix)

=== CRITICAL RULES ===
*** Extract ONLY the relevant topic/query — REMOVE greetings (hello, hey, hi), assistant name (Jarvis), filler words (can you, please, for me), platform names (on YouTube, on Google), and command words from the query. ***
*** "Open webcam" / "Turn on camera" / "Start camera" → open_webcam (NEVER "open webcam" as a website) ***
*** "Close webcam" / "Turn off camera" → close_webcam ***
*** For multiple tasks in one message: "Open Facebook and play Despacito" → open facebook, play Despacito ***
*** If user says to play AND you detect YouTube context, output ONLY 'play (query)' — do NOT add a separate youtube_search. Playing IS searching YouTube. ***
*** "Draw/Generate/Create [image description]" → generate_image (keep full visual description) ***
*** Output ONLY the structured response. No explanation. No extra text. ***"""

class BrainService:

    def __init__(self, groq_service=None):
        self.groq_service = groq_service
        self._llms = []
        self._last_task_decisions = []

        if GROQ_API_KEYS:
            try:
                from langchain_groq import ChatGroq
                self._llms = [
                    ChatGroq(
                        groq_api_key=key,
                        model_name=INTENT_CLASSIFY_MODEL,
                        temperature=0.0,
                        max_tokens=200,
                        request_timeout=15,
                    )
                    for key in GROQ_API_KEYS
                ]

                logger.info("[BRAIN] Two-stage decision model initialized (%s) with %d key(s)",
                            INTENT_CLASSIFY_MODEL, len(self._llms))
                
            except Exception as e:
                logger.warning("[BRAIN] Failed to create Groq: %s", e)
                
        if not self._llms and not groq_service:
            logger.warning("[BRAIN] No Groq. Will use rule-based fallback.")

    def classify_primary(
        self,
        user_message: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        key_index: int = 0,
    ) -> Tuple[str, str, int]:
        msg = (user_message or "").strip()
        if not msg:
            return ("general", "empty", 0)

        user_content = self._build_context(msg, chat_history)
        t0 = time.perf_counter()

        category, method = self._run_llm(
            _PRIMARY_BRAIN_PROMPT, user_content, key_index, ALL_CATEGORIES, "general"
        )

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("[BRAIN-PRIMARY] %s -> %s (%d ms, %s)", msg[:50], category, elapsed_ms, method)
        return (category, method, elapsed_ms)

    _TASK_FEW_SHOTS = [
        ("how are you?", "general how are you?"),
        ("open chrome and tell me about mahatma gandhi.", "open chrome, general tell me about mahatma gandhi"),
        ("open chrome and firefox", "open chrome, open firefox"),
        ("play Dhurandhar title track on YouTube", "play Dhurandhar title track"),
        ("hello Jarvis can you play Shape of You", "play Shape of You"),
        ("generate image of a lion and open facebook", "generate_image a lion, open facebook"),
        ("search for Python tutorials on google", "google_search Python tutorials"),
        ("search YouTube for cooking recipes", "youtube_search cooking recipes"),
        ("write an application for leave and play some music", "content application for leave, play some music"),
        ("hey Jarvis Teja song Dhurandhar title track can you play that on YouTube", "play Teja Dhurandhar title track"),
        ("can you open the website Jarvis for everyone", "open jarvisforeveryone.com"),
        ("draw me a beautiful sunset over the ocean", "generate_image beautiful sunset over the ocean"),
        ("can you make a picture of a dragon breathing fire", "generate_image dragon breathing fire"),
        ("create an image of a futuristic city at night", "generate_image futuristic city at night"),
        ("write me a poem about the stars", "content poem about the stars"),
        ("open webcam", "open_webcam"),
        ("turn off the camera", "close_webcam"),
        ("play some lo-fi beats", "play lo-fi beats"),
        ("open YouTube and play Arijit Singh songs", "open youtube, play Arijit Singh songs"),
        ("make a presentation about climate change", "generate_ppt climate change"),
        ("create a PPT on AI and send it to my email", "generate_ppt_email AI"),
        ("track the phone number +1 800 555 1234", "track_phone +18005551234"),
        ("lookup phone 09876543210", "track_phone 09876543210"),
        ("whose number is +91 98765 43210", "track_phone +919876543210"),
        ("+14155552671", "track_phone +14155552671"),
        ("9876543210", "track_phone 9876543210"),
        ("generate an image of a tiger and email it to me", "generate_image_email a tiger"),
        ("create a picture of the eiffel tower and send it to my email", "generate_image_email the eiffel tower"),
        ("draw a sunset and mail it to me", "generate_image_email a sunset"),
        ("make an image of a dog and send it on my email", "generate_image_email a dog"),
        ("write a python script to scrape hacker news front page", "generate_code python script to scrape hacker news front page"),
        ("build me a react dark mode toggle in typescript", "generate_code react dark mode toggle in typescript"),
        ("code a binary search algorithm in c++", "generate_code binary search algorithm in c++"),
        ("write a fastapi crud endpoint for users", "generate_code fastapi crud endpoint for users"),
        ("create a discord bot in node js", "generate_code discord bot in node js"),
        ("write a python web scraper and email it to me", "generate_code_email python web scraper"),
        ("code a todo app in react and send it on my email", "generate_code_email todo app in react"),
    ]

    def classify_task(
        self,
        user_message: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        key_index: int = 0,
    ) -> Tuple[List[str], str, int]:
        msg = (user_message or "").strip()
        if not msg:
            self._last_task_decisions = []
            return (["open"], "empty", 0)

        m_lower = msg.lower()
        if any(x in m_lower for x in ["open webcam", "turn on camera", "start camera",
                                        "open the webcam", "start the camera", "turn on the camera"]):
            self._last_task_decisions = [("open_webcam", "")]
            return (["open_webcam"], "rule-fast", 0)
        
        if any(x in m_lower for x in ["close webcam", "turn off camera", "stop camera",
                                        "close the webcam", "stop the camera", "turn off the camera"]):
            self._last_task_decisions = [("close_webcam", "")]
            return (["close_webcam"], "rule-fast", 0)

        # ── Fast path: "generate code" / "write a script" requests ────
        # MUST run before the image fast-path because "create" / "generate"
        # appear in both intents — we disambiguate via code keywords.
        _email_signals = ("email", "e-mail", "mail it", "send it to my", "send to my", "send on my")
        _code_signals = (
            "code", "script", "program", "function ", "class ", "algorithm",
            "snippet", "endpoint", "api ", "crud", "scraper", "bot ",
            "component", "module", "package",
            "python", "javascript", "typescript", "react", "node",
            "java ", "c++", "cpp", "c#", "csharp", "golang", " go ",
            "rust", "ruby", "php", "swift", "kotlin", "bash", "shell",
            "sql ", "html", "css",
        )
        _code_verbs = ("write ", "code ", "build ", "make ", "create ",
                       "generate ", "draft ", "give me ", "implement ")
        if any(s in m_lower for s in _code_signals) and any(v in m_lower for v in _code_verbs):
            prompt = self._extract_code_prompt(msg)
            if any(s in m_lower for s in _email_signals):
                self._last_task_decisions = [("generate_code_email", prompt)]
                return (["generate_code_email"], "rule-fast", 0)
            self._last_task_decisions = [("generate_code", prompt)]
            return (["generate_code"], "rule-fast", 0)

        # ── Fast path: "generate image AND email it" combo ─────────────
        _image_signals = ("image", "picture", "photo", "drawing", "draw ", "generate ", "create ")
        if any(s in m_lower for s in _email_signals) and any(s in m_lower for s in _image_signals):
            prompt = self._extract_image_prompt(msg)
            self._last_task_decisions = [("generate_image_email", prompt)]
            return (["generate_image_email"], "rule-fast", 0)


        # ── Fast path: bare / explicit phone number ────────────────────
        _phone = self._extract_phone_number(msg)
        if _phone:
            digits_only = re.sub(r"\D", "", msg)
            if (
                # User said something like "track / lookup / whose number"
                any(k in m_lower for k in (
                    "track", "lookup", "look up", "whose number",
                    "phone number", "this number", "trace ",
                ))
                # ...or the message is essentially JUST a phone number
                or (len(digits_only) >= 7 and len(digits_only) >= int(len(re.sub(r"\s", "", msg)) * 0.7))
            ):
                self._last_task_decisions = [("track_phone", _phone)]
                return (["track_phone"], "rule-fast", 0)

        context_lines = []

        if chat_history:
            for u, a in chat_history[-MAX_CONTEXT_TURNS:]:
                u_preview = (u or "")[:MAX_MESSAGE_PREVIEW] + ("…" if len(u or "") > MAX_MESSAGE_PREVIEW else "")
                a_preview = (a or "")[:MAX_MESSAGE_PREVIEW] + ("…" if len(a or "") > MAX_MESSAGE_PREVIEW else "")
                context_lines.append(f"User: {u_preview}")
                context_lines.append(f"Assistant: {a_preview}")

        context_block = "\n".join(context_lines) if context_lines else ""
        context_section = f"Recent conversation:\n{context_block}\n\n" if context_block else ""
        user_content = f"{context_section}User: {msg[:MAX_MESSAGE_PREVIEW]}"
        t0 = time.perf_counter()

        raw_response, method = self._run_llm_structured(
            _TASK_BRAIN_PROMPT, user_content, key_index)

        decisions = self._parse_task_decisions(raw_response)
        self._last_task_decisions = decisions
        task_types = [d[0] for d in decisions]

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("[BRAIN-TASK] %s -> %s (%d ms, %s)", msg[:50], decisions, elapsed_ms, method)
        return (task_types, method, elapsed_ms)

    def classify(
        self,
        user_message: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        key_index: int = 0,
    ) -> Tuple[str, List[str], str, int]:
        category, method1, ms1 = self.classify_primary(user_message, chat_history, key_index)

        if category == "task":
            task_types, method2, ms2 = self.classify_task(user_message, chat_history, key_index)
            combined_method = f"{method1}+{method2}"
            return (category, task_types, combined_method, ms1 + ms2)

        return (category, [], method1, ms1)

    def extract_task_payloads(
        self, user_message: str, task_types: List[str],
        chat_history: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Tuple[str, dict]]:
        from app.services.decision_types import ROUTE_TO_INTENT, INTENT_OPEN

        decisions = getattr(self, '_last_task_decisions', [])

        intents = []

        if decisions:
            for task_type, clean_query in decisions:
                intent_key = ROUTE_TO_INTENT.get(task_type, task_type)
                payload = {"message": user_message, "raw": user_message}

                if task_type == "open":
                    if clean_query:
                        url = self._resolve_open_query(clean_query)
                    else:
                        # LLM returned empty query (rate-limit / rule-based fallback).
                        # Extract the site name directly from the original user message.
                        urls = self._extract_urls(user_message)
                        url = urls[0] if urls else "https://www.google.com"
                    payload["url"] = url
                elif task_type == "play":
                    payload["query"] = clean_query or self._extract_play_query(user_message)
                elif task_type in ("google_search", "youtube_search"):
                    payload["query"] = clean_query or self._extract_search_query(user_message)
                elif task_type == "generate_image":
                    payload["prompt"] = clean_query or self._extract_image_prompt(user_message)
                elif task_type in ("generate_code", "generate_code_email"):
                    payload["prompt"] = clean_query or self._extract_code_prompt(user_message)
                elif task_type == "content":
                    payload["prompt"] = clean_query or self._extract_content_prompt(user_message)
                elif task_type in ("generate_ppt", "generate_ppt_email"):
                    payload["prompt"] = clean_query or self._extract_content_prompt(user_message)
                elif task_type == "track_phone":
                    payload["phone"] = clean_query or self._extract_phone_number(user_message)
                intents.append((intent_key, payload))

        else:
            resolved_message = self._resolve_correction(user_message, chat_history)

            for task_type in task_types:
                intent_key = ROUTE_TO_INTENT.get(task_type, task_type)
                payloads = self._extract_payload(task_type, resolved_message)

                if isinstance(payloads, list):
                    for p in payloads:
                        intents.append((intent_key, p))

                else:
                    intents.append((intent_key, payloads))

        return intents

    def _resolve_open_query(self, query: str) -> str:

        q = query.strip().lower()

        if q in self.SITE_MAP:
            return self.SITE_MAP[q]

        if "." in q:
            return f"https://{q}" if not q.startswith("http") else q

        if q in self.SITE_MAP:
            return self.SITE_MAP[q]

        return f"https://www.{q}.com"

    def _resolve_correction(self, msg: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:

        if not chat_history:
            return msg

        m_lower = msg.lower().strip()
        correction_signals = ["not that", "no i said", "no, i said", "i meant", "it's not", "its not",
                              "that's wrong", "thats wrong", "not f-o-r", "not for ",
                              "the other", "try again", "do that again", "one more time",
                              "no no", "wrong one", "instead", "i didn't say"]
        
        is_correction = any(sig in m_lower for sig in correction_signals)

        if not is_correction:
            return msg

        for u, a in reversed(chat_history[-MAX_CONTEXT_TURNS:]):
            u_lower = (u or "").lower()

            if any(sig in u_lower for sig in correction_signals):
                continue

            if any(p in u_lower for p in ["open ", "play ", "search ", "generate ", "write ", "launch ", "go to "]):
                logger.info("[BRAIN] Correction detected. Original: '%s' | Correction: '%s'", u[:80], msg[:80])

                import re
                domain_match = re.search(r'([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z]{2,})+)', msg)

                if domain_match:
                    new_domain = domain_match.group(1)
                    return f"open {new_domain}"

                return f"{u} (correction: {msg})"
            
            break

        return msg

    def _build_context(self, msg: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:

        context_lines = []

        if chat_history:
            for u, a in chat_history[-MAX_CONTEXT_TURNS:]:
                u_preview = (u or "")[:MAX_MESSAGE_PREVIEW] + ("…" if len(u or "") > MAX_MESSAGE_PREVIEW else "")
                a_preview = (a or "")[:MAX_MESSAGE_PREVIEW] + ("…" if len(a or "") > MAX_MESSAGE_PREVIEW else "")
                context_lines.append(f"User: {u_preview}")
                context_lines.append(f"Assistant: {a_preview}")
        context_block = "\n".join(context_lines) if context_lines else "(No prior conversation)"
        msg_preview = msg[:MAX_MESSAGE_PREVIEW]

        correction_hint = ""
        m_lower = msg.lower().strip()
        correction_signals = ["not that", "no i said", "no, i said", "i meant", "it's not", "its not",
                              "that's wrong", "thats wrong", "i said", "not f-o-r", "not for",
                              "the other", "try again", "do that again", "one more time",
                              "no no", "wrong one", "instead", "i didn't say", "not what i"]
        
        if any(sig in m_lower for sig in correction_signals):
            correction_hint = "\n\nNOTE: This message appears to be a CORRECTION or CLARIFICATION of a previous request. Check the conversation history to determine what the user originally asked for, and classify this message as the SAME category as that original request."

        return f"""Conversation so far:
{context_block}

Current user message: {msg_preview}{correction_hint}

Classify. Output EXACTLY ONE category name."""

    def _run_llm(
        self, system_prompt: str, user_content: str, key_index: int,
        valid_options: List[str], default: str
    ) -> Tuple[str, str]:
        
        if self._llms:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                idx = key_index % len(self._llms)
                llm = self._llms[idx]
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
                text = (response.content or "").strip().lower()
                result = self._parse_single(text, valid_options, default)
                return (result, "llm")
            
            except Exception as e:
                logger.warning("[BRAIN] LLM failed: %s. Using rule-based.", e)

        msg = user_content.split("Current user message:")[-1].strip()[:500] if "Current user message:" in user_content else user_content[:500]
        result = self._rule_based_primary(msg)
        return (result, "rule-based")

    def _run_llm_multi(
        self, system_prompt: str, user_content: str, key_index: int,
        valid_options: List[str]
    ) -> Tuple[List[str], str]:
        
        if self._llms:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                idx = key_index % len(self._llms)
                llm = self._llms[idx]
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ])
                text = (response.content or "").strip().lower()
                results = self._parse_multi(text, valid_options)
                return (results, "llm")
            
            except Exception as e:
                logger.warning("[BRAIN-TASK] LLM failed: %s. Using rule-based.", e)

        msg = user_content.split("User task request:")[-1].strip()[:500] if "User task request:" in user_content else user_content[:500]
        results = self._rule_based_task(msg)
        return (results, "rule-based")

    def _parse_single(self, text: str, valid_options: List[str], default: str) -> str:

        if not text:
            return default
        
        text = text.strip().lower()

        for opt in valid_options:
            if text == opt:
                return opt

        for opt in valid_options:
            if opt in text:
                return opt
        return default

    def _parse_multi(self, text: str, valid_options: List[str]) -> List[str]:

        if not text:
            return ["open"]
        
        results = []
        seen = set()

        for part in re.split(r"[,;\s]+", text):
            r = part.strip().lower()

            if not r:
                continue

            for valid in valid_options:

                if valid == r or valid in r:
                    if valid not in seen:
                        results.append(valid)
                        seen.add(valid)
                    break

        return results if results else ["open"]

    def _run_llm_structured(
        self, system_prompt: str, user_content: str, key_index: int,
    ) -> Tuple[str, str]:
        
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

        few_shot_msgs = []
        for user_ex, ai_ex in self._TASK_FEW_SHOTS:
            few_shot_msgs.append(HumanMessage(content=f"User: {user_ex}"))
            few_shot_msgs.append(AIMessage(content=ai_ex))

        messages = [SystemMessage(content=system_prompt)] + few_shot_msgs + [HumanMessage(content=user_content)]

        if self._llms:

            try:
                idx = key_index % len(self._llms)
                llm = self._llms[idx]
                response = llm.invoke(messages)
                text = (response.content or "").strip()
                return (text, "llm")
            
            except Exception as e:
                logger.warning("[BRAIN-TASK] Structured LLM failed: %s. Using rule-based.", e)

        msg = user_content.split("User:")[-1].strip()[:500] if "User:" in user_content else user_content[:500]
        results = self._rule_based_task(msg)
        fallback = ", ".join(results)
        return (fallback, "rule-based")

    def _parse_task_decisions(self, raw_response: str) -> List[Tuple[str, str]]:

        if not raw_response:
            return [("open", "")]

        text = raw_response.replace("\n", ",").strip()

        # ORDER MATTERS: longest / most specific prefixes first, otherwise
        # "generate_image_email" would be eaten by "generate_image".
        TASK_PREFIXES = [
            "generate_code_email", "generate code email",
            "generate_image_email", "generate image email",
            "generate_ppt_email", "generate ppt email",
            "generate_code", "generate code",
            "generate_image", "generate image",
            "generate_ppt", "generate ppt",
            "google_search", "google search",
            "youtube_search", "youtube search",
            "open_webcam", "close_webcam",
            "track_phone", "track phone",
            "content", "open", "close", "play",
            "general", "realtime",
        ]

        NORMALIZE = {
            "generate image": "generate_image",
            "generate image email": "generate_image_email",
            "generate ppt": "generate_ppt",
            "generate ppt email": "generate_ppt_email",
            "generate code": "generate_code",
            "generate code email": "generate_code_email",
            "google search": "google_search",
            "youtube search": "youtube_search",
            "track phone": "track_phone",
        }


        decisions = []

        parts = [p.strip() for p in text.split(",") if p.strip()]

        for part in parts:
            part_lower = part.lower().strip()
            matched = False

            for prefix in TASK_PREFIXES:
                if part_lower.startswith(prefix):
                    query = part[len(prefix):].strip().rstrip(".!?")
                    task_type = NORMALIZE.get(prefix, prefix)

                    if task_type in ("general", "realtime"):
                        continue
                    decisions.append((task_type, query))
                    matched = True
                    break

            if not matched:
                for prefix in TASK_PREFIXES:
                    if prefix in part_lower:
                        idx = part_lower.index(prefix)
                        query = part[idx + len(prefix):].strip().rstrip(".!?")
                        task_type = NORMALIZE.get(prefix, prefix)
                        if task_type in ("general", "realtime"):
                            continue
                        decisions.append((task_type, query))
                        matched = True
                        break

                if not matched:
                    logger.warning("[BRAIN-TASK] Could not parse decision part: '%s'", part[:80])

        return decisions if decisions else [("open", "")]

    def _rule_based_primary(self, msg: str) -> str:
        m = (msg or "").strip().lower()

        if any(x in m for x in ["do you know my", "link of my website", "my website link", "what's my website", "know the link of my"]):
            return "general"

        if m in ("hello", "hi", "hey", "good morning", "good evening", "good afternoon",
                 "how are you", "what's up", "thanks", "thank you", "bye", "goodbye"):
            return "general"

        if any(x in m for x in ["what do you see", "what can you see", "what am i holding",
                                  "what is this", "describe this", "identify this",
                                  "what's in my hand", "look at this", "read this",
                                  "can you see", "check this out", "show you"]):
            return "camera"

        if any(x in m for x in ["open webcam", "turn on camera", "start camera",
                                  "close webcam", "turn off camera", "stop camera"]):
            return "task"

        task_patterns = [
            "open ", "launch ", "go to ", "visit ",
            "play ", "play the ", "play some ", "put on ",
            "generate image", "generate an image", "draw ", "create image", "create an image",
            "make me a picture", "make a picture", "picture of ", "image of ",
            "write ", "draft ", "compose ", "essay", "poem", "letter",
            "search for ", "look up ", "find me ", "google ",
            "search youtube", "find videos",
        ]

        if any(m.startswith(p) or p in m for p in task_patterns):
            return "task"

        # Bare phone numbers / "track this number" → task (track_phone)
        if any(k in m for k in ("track ", "lookup ", "look up ", "whose number", "phone number", "trace ")):
            return "task"
        digits_only = re.sub(r"\D", "", m)
        compact = re.sub(r"\s", "", m)
        if len(digits_only) >= 7 and len(digits_only) >= int(len(compact) * 0.7):
            return "task"

        if any(x in m for x in ["who is ", "who are ", "latest", "current", "news", "weather", "today",
                                  "recent", "stock price", "trending", "score", "tell me about ",
                                  "what happened", "how much does", "price of", "cost of",
                                  "reviews of", "best restaurants"]):
            return "realtime"

        return "general"

    def _rule_based_task(self, msg: str) -> List[str]:

        m = (msg or "").strip().lower()
        tasks = []

        if any(x in m for x in ["open webcam", "turn on camera", "start camera", "show me the camera",
                                  "open the webcam", "start the camera", "turn on the camera"]):
            return ["open_webcam"]
        if any(x in m for x in ["close webcam", "turn off camera", "stop camera",
                                  "close the webcam", "stop the camera", "turn off the camera"]):
            return ["close_webcam"]

        if m.startswith(("open ", "launch ", "go to ", "visit ", "can you open ")) or \
           ("open" in m and any(s in m for s in ["facebook", "youtube", "google", "netflix", "gmail", "instagram", "twitter", "linkedin"])):
            tasks.append("open")

        if m.startswith(("play ", "play the ", "play some ")) or " play " in m:
            tasks.append("play")

        if any(x in m for x in ["generate image", "draw ", "create image", "make me ", "picture of ", "image of "]):
            tasks.append("generate_image")

        if any(x in m for x in ["write ", "draft ", "compose ", "essay", "poem", "letter", "application "]):
            tasks.append("content")

        if "youtube" in m and any(x in m for x in ["search", "find"]):
            tasks.append("youtube_search")

        if any(x in m for x in ["search for ", "look up ", "find me ", "google "]) and "youtube" not in m:
            tasks.append("google_search")

        return tasks if tasks else ["open"]

    SITE_MAP = {
        "facebook": "https://www.facebook.com", "instagram": "https://www.instagram.com",
        "youtube": "https://www.youtube.com", "google": "https://www.google.com",
        "netflix": "https://www.netflix.com", "twitter": "https://twitter.com",
        "x.com": "https://x.com", "gmail": "https://mail.google.com",
        "whatsapp": "https://web.whatsapp.com", "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com", "discord": "https://discord.com/app",
        "spotify": "https://open.spotify.com", "tiktok": "https://www.tiktok.com",
        "amazon": "https://www.amazon.com", "github": "https://github.com",
        "wikipedia": "https://www.wikipedia.org", "stackoverflow": "https://stackoverflow.com",
        "medium": "https://medium.com", "notion": "https://www.notion.so",
        "figma": "https://www.figma.com", "canva": "https://www.canva.com",
        "zoom": "https://zoom.us", "drive": "https://drive.google.com",
        "maps": "https://www.google.com/maps",
        "jarvis for everyone": "https://jarvisforeveryone.com",
        "jarvisforeveryone": "https://jarvisforeveryone.com",
        "jarvis4everyone": "https://jarvis4everyone.com",
        "jarvis4everyone.com": "https://jarvis4everyone.com",
        "my website": "https://jarvisforeveryone.com",
        "jarvisforeveryone.com": "https://jarvisforeveryone.com",
        "chatgpt": "https://chatgpt.com",
        "chat gpt": "https://chatgpt.com",
        "openai": "https://chatgpt.com",
        "claude": "https://claude.ai",
        "gemini": "https://gemini.google.com",
        "perplexity": "https://www.perplexity.ai",
        "copilot": "https://copilot.microsoft.com",
        "twitch": "https://www.twitch.tv",
        "pinterest": "https://www.pinterest.com",
    }

    def _strip_filler(self, msg: str) -> str:
        cleaned = msg.strip()

        cleaned = re.sub(
            r'^(?:hello|hi|hey|yo|hiya|howdy|ok|okay|alright)\s+(?:jarvis|j\.?a\.?r\.?v\.?i\.?s\.?)\s*[,.]?\s*',
            '', cleaned, flags=re.I
        ).strip()

        cleaned = re.sub(
            r'^(?:hello|hi|hey|yo|hiya|howdy)\s*[,.]?\s*',
            '', cleaned, flags=re.I
        ).strip()

        cleaned = re.sub(
            r'^(?:jarvis|j\.?a\.?r\.?v\.?i\.?s\.?)\s*[,.]?\s*',
            '', cleaned, flags=re.I
        ).strip()

        cleaned = re.sub(r'\s+(?:please|pls|plz)\s*[.!?]*$', '', cleaned, flags=re.I).strip()
        cleaned = re.sub(r'\s+(?:for me|right now|now|asap)\s*[.!?]*$', '', cleaned, flags=re.I).strip()
        return cleaned if cleaned else msg.strip()

    def _extract_payload(self, task_type: str, message: str):

        if task_type == "open":
            urls = self._extract_urls(message)

            if len(urls) <= 1:
                return {"message": message, "raw": message, "url": urls[0] if urls else "https://www.google.com"}
            
            return [{"message": message, "raw": message, "url": u} for u in urls]
        
        if task_type == "open_webcam":
            return {"message": message, "raw": message}
        
        if task_type == "close_webcam":
            return {"message": message, "raw": message}

        payload = {"message": message, "raw": message}

        if task_type == "play":
            payload["query"] = self._extract_play_query(message)

        elif task_type == "generate_image":
            payload["prompt"] = self._extract_image_prompt(message)

        elif task_type == "content":
            payload["prompt"] = self._extract_content_prompt(message)

        elif task_type in ("google_search", "youtube_search"):
            payload["query"] = self._extract_search_query(message)

        return payload

    def _extract_urls(self, msg: str) -> list:

        from urllib.parse import urlparse
        msg_lower = msg.lower()
        urls, seen = [], set()

        def _add(u):
            if not u or u in seen:
                return
            
            u2 = u.strip().rstrip(".!?,")
            if not u2.startswith("http"):
                u2 = "https://" + u2

            try:
                p = urlparse(u2)
                if p.scheme not in ("http", "https"):
                    return
                
            except Exception:
                return
            
            urls.append(u2)
            seen.add(u2)

        for m in re.finditer(r"https?://[^\s]+", msg, re.I):
            _add(m.group(0))

        for name, url in self.SITE_MAP.items():
            if name in msg_lower:
                _add(url)

        if urls:
            return urls
        
        for prefix in ["open ", "launch ", "go to ", "visit ", "can you open "]:

            if prefix in msg_lower:
                rest = msg_lower.split(prefix, 1)[-1].replace(" for me", "").strip().rstrip(".!?")

                if rest:
                    for p in re.split(r"\s+and\s+|\s*,\s*", rest):
                        p = p.strip()

                        if p in self.SITE_MAP:
                            _add(self.SITE_MAP[p])

                        elif p and "." in p:
                            _add(p)

                        elif p:
                            _add("https://www." + p + ".com")

                break

        return urls if urls else ["https://www.google.com"]

    def _extract_play_query(self, msg: str) -> str:

        cleaned = self._strip_filler(msg)
        lower = cleaned.lower()

        m = re.search(r'^(.+?)\s+(?:can you|could you|please)\s+play\s+(?:that|it|this)\b', lower)

        if m:
            result = cleaned[:m.end(1)].strip().rstrip(".!?,")
            if result:
                return result

        m = re.search(
            r'(?:can you|could you|please)\s+play\s+(?:the\s+|a\s+|some\s+|me\s+)?(.+?)(?:\s+on\s+youtube|\s+for\s+me|\s+please\s*)?\s*[.!?]*$',
            lower,
        )

        if m:
            result = cleaned[m.start(1):m.end(1)].strip().rstrip(".!?,")
            if result and result.lower() not in ("that", "it", "this", "something"):
                return result

        m = re.search(
            r'play\s+(?:the\s+|a\s+|some\s+|me\s+)?(.+?)(?:\s+on\s+youtube|\s+for\s+me|\s+please\s*)?\s*[.!?]*$',
            lower,
        )
        
        if m:
            result = cleaned[m.start(1):m.end(1)].strip().rstrip(".!?,")
            if result and result.lower() not in ("that", "it", "this", "something"):
                return result

        for p in ["play ", "play the ", "play some ", "play a "]:
            if lower.startswith(p):
                return cleaned[len(p):].strip().rstrip(".!?")

        return cleaned.strip()

    def _extract_image_prompt(self, msg: str) -> str:

        m, lower = msg.strip(), msg.lower()
        extracted = None

        for p in ["generate ", "draw ", "create ", "make me ", "make a ", "picture of ", "image of ", "generator image of "]:
            if lower.startswith(p):
                extracted = m[len(p):].strip().rstrip(".!?")
                break

            if p in lower:
                extracted = m[lower.find(p) + len(p):].strip().rstrip(".!?")
                break

        if not extracted:
            return m
        
        boundaries = [
            r"\s+and\s+write\s", r"\s+and\s+open\s", r"\s+and\s+generate\s",
            r"\s+and\s+draw\s", r"\s+and\s+play\s", r"\s+and\s+search\s",
            r"\s+and\s+launch\s", r"\s+and\s+go\s+to\s", r"\s+and\s+visit\s",
        ]

        for b in boundaries:
            match = re.search(b, extracted.lower(), re.I)

            if match:
                extracted = extracted[:match.start()].strip().rstrip(".!?,")
                break

        return extracted.strip() if extracted.strip() else m

    def _extract_search_query(self, msg: str) -> str:

        cleaned = self._strip_filler(msg)
        lower = cleaned.lower()

        for p in ["search youtube for ", "search youtube ", "youtube search for ",
                   "search on youtube for ", "search on youtube ",
                   "search google for ", "search on google for ", "search on google ",
                   "search for ", "look up ", "find me ", "find ",
                   "google search for ", "google search ", "google "]:
            
            if p in lower:
                rest = cleaned[lower.find(p) + len(p):].strip().rstrip(".!?")
                return rest if rest else cleaned.strip()

        m = re.search(
            r'(?:can you|could you|please)\s+search\s+(?:for\s+)?(.+?)(?:\s+on\s+(?:youtube|google)|\s+for\s+me|\s+please\s*)?\s*[.!?]*$',
            lower,
        )

        if m:
            result = cleaned[m.start(1):m.end(1)].strip().rstrip(".!?,")
            if result:
                return result

        m = re.search(r'^(.+?)\s+on\s+(?:youtube|google)\s*[.!?]*$', lower)

        if m:
            result = cleaned[:m.end(1)].strip().rstrip(".!?,")

            if result:
                result2 = re.sub(
                    r'^(?:can you |could you |please )?(?:play|search|find|look up)\s+(?:the\s+|a\s+|some\s+|me\s+)?',
                    '', result, flags=re.I
                ).strip()
                return result2 if result2 else result

        stripped = re.sub(
            r'^(?:can you |could you |please )?(?:play|search|find|look up)\s+(?:the\s+|a\s+|some\s+|me\s+)?',
            '', lower, flags=re.I
        ).strip()

        stripped = re.sub(r'\s+on\s+(?:youtube|google)\s*[.!?]*$', '', stripped, flags=re.I).strip()
        stripped = re.sub(r'\s+(?:for me|please)\s*[.!?]*$', '', stripped, flags=re.I).strip()

        if stripped and stripped != lower:
            return stripped.rstrip(".!?,")
        return cleaned.strip()

    def _extract_content_prompt(self, msg: str) -> str:

        m, lower = msg.strip(), msg.lower()
        boundaries = [
            r"\s+and\s+open\s", r"\s+and\s+generate\s", r"\s+and\s+draw\s",
            r"\s+and\s+play\s", r"\s+and\s+search\s", r"\s+and\s+launch\s",
            r"\s+and\s+go\s+to\s", r"\s+and\s+visit\s",
        ]

        triggers = [
            "write application", "write an application", "write a application",
            "write a letter", "write letter", "draft a letter", "draft letter",
            "write an essay", "write essay", "write a poem", "write poem",
            "write a song", "write song", "compose a", "write the following",
            "write ", "draft ", "compose ",
        ]

        best_start = -1
        best_trigger = ""

        for t in triggers:
            pos = lower.find(t)
            if pos >= 0 and (best_start < 0 or pos < best_start):
                best_start = pos
                best_trigger = t

        if best_start < 0:
            return m
        
        segment = m[best_start:].strip()
        segment_lower = segment.lower()

        for b in boundaries:
            match = re.search(b, segment_lower, re.I)

            if match:
                segment = segment[:match.start()].strip().rstrip(".!?,")
                break
            
        segment = segment.rstrip(".!?,").strip()
        return segment if segment else m

    def _extract_phone_number(self, msg: str) -> str:
        """Pull the most-likely phone number out of a free-form message.

        Handles inputs like:
            "track +1 (415) 555-2671 for me"      -> "+14155552671"
            "whose number is 09876543210"          -> "09876543210"
            "phone 91 98765 43210"                 -> "+919876543210"
            "9876543210"                           -> "9876543210"
        """
        if not msg:
            return ""

        text = msg.strip()

        # 1. Prefer an explicit E.164-ish match with a leading "+".
        m = re.search(r"\+\s*\d[\d\s().-]{5,}\d", text)
        if m:
            return "+" + re.sub(r"\D", "", m.group(0))

        # 2. Otherwise grab the longest run of digits/format-chars.
        candidates = re.findall(r"\d[\d\s().-]{5,}\d", text)
        if candidates:
            best = max(candidates, key=lambda s: len(re.sub(r"\D", "", s)))
            digits = re.sub(r"\D", "", best)
            if 7 <= len(digits) <= 15:
                return digits

        # 3. Last resort: any digit run >= 7.
        digits = re.sub(r"\D", "", text)
        if 7 <= len(digits) <= 15:
            return digits
        return ""

    def _extract_code_prompt(self, msg: str) -> str:
        """Pull the coding task description out of a free-form message.

        "write a python script to scrape hacker news and email it to me"
            -> "python script to scrape hacker news"
        """
        if not msg:
            return ""
        m = msg.strip()
        lower = m.lower()

        triggers = [
            "write me ", "write a ", "write an ", "write the ", "write ",
            "code me ", "code a ", "code an ", "code the ", "code ",
            "build me ", "build a ", "build an ", "build the ", "build ",
            "make me ", "make a ", "make an ", "make the ", "make ",
            "create me ", "create a ", "create an ", "create the ", "create ",
            "generate me ", "generate a ", "generate an ", "generate the ",
            "generate code for ", "generate code ", "generate ",
            "give me ", "implement ", "draft ",
            "i need ", "can you write ", "can you build ", "can you code ",
            "please write ", "please code ",
        ]

        best_start = -1
        best_trigger = ""
        for t in triggers:
            pos = lower.find(t)
            if pos >= 0 and (best_start < 0 or pos < best_start):
                best_start = pos
                best_trigger = t

        if best_start >= 0:
            extracted = m[best_start + len(best_trigger):].strip()
        else:
            extracted = m

        # Strip email/send tail so we don't pollute the coding prompt.
        ex_lower = extracted.lower()
        tail_patterns = [
            r"\s+and\s+(?:email|e-mail|mail|send)\s+(?:it|that|this|me)?.*$",
            r"\s+(?:email|e-mail|mail|send)\s+(?:it|that|this)\s+(?:to|on)\s+(?:my|the)?\s*(?:email|mail|inbox)?.*$",
            r"\s+and\s+send\s+(?:it|that|this).*$",
        ]
        for pat in tail_patterns:
            mt = re.search(pat, ex_lower, re.I)
            if mt:
                extracted = extracted[:mt.start()].strip().rstrip(".!?,")
                break

        # Trim other "and open/play/search" tails.
        boundaries = [
            r"\s+and\s+open\s", r"\s+and\s+play\s", r"\s+and\s+search\s",
            r"\s+and\s+launch\s", r"\s+and\s+go\s+to\s", r"\s+and\s+visit\s",
        ]
        for b in boundaries:
            mt = re.search(b, extracted.lower(), re.I)
            if mt:
                extracted = extracted[:mt.start()].strip().rstrip(".!?,")
                break

        extracted = extracted.strip().rstrip(".!?,")
        return extracted if extracted else m
