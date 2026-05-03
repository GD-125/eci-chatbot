"""
app.py — Bilingual FastAPI chatbot server (English + Tamil).

Features:
  - Separate FAISS indexes per language (en, ta)
  - Input validation: blocks special characters, length limits, injection attempts
  - Language-aware fallback messages
  - Confidence blending for near-equal top results
  - Per-session conversation context memory
  - /api/chat accepts a `lang` field ("en" | "ta")

Run locally:
  uvicorn app:app --reload --port 8080

Deploy:
  See Dockerfile
"""

import json
import os
import pickle
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import fetchers
from fetchers.eci_results_scraper import get_all_cached_results
from fetchers.eci_rss_fetcher import get_rss_qa_pairs

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR            = Path("models")
SUPPORTED_LANGS      = ["en", "ta"]
PORT                 = int(os.getenv("PORT", "8080"))

TOP_K                     = 4      # retrieve top 4 candidates
CONFIDENCE_THRESHOLD        = 0.3   # cosine similarity floor for strong answers (lower for TF-IDF)
LOW_CONFIDENCE_THRESHOLD    = 0.15  # allow lower-confidence answers with a note
MAX_HISTORY               = 10     # conversation turns remembered per session
BLEND_DELTA               = 0.04   # blend 2nd result if score within this margin

# ── Input security ────────────────────────────────────────────────────────────
# Allow: Unicode letters (covers Tamil), digits, spaces, and selected symbols ? . , @ $
# Block: everything else (HTML, SQL, shell injection chars, etc.)
ALLOWED_CHARS_RE = re.compile(
    r"^[\w\s\u0B80-\u0BFF?.@,$'\-]+$",
    re.UNICODE,
)
MAX_MSG_LEN = 500
MIN_MSG_LEN = 2

# ── Language-aware fallback messages ──────────────────────────────────────────
FALLBACK = {
    "en": (
        "I'm sorry, I don't have specific information on that topic.\n"
        "As the Election Commission of India assistant, I can help with:\n"
        "- Voter registration and EPIC card\n"
        "- Finding your polling booth\n"
        "- Election process and timeline\n"
        "- ECI information and helpline\n\n"
        "Please ask about election-related topics, or call the Election Helpline: 1950 (toll-free)."
    ),
    "ta": (
        "மன்னிக்கவும், இந்த தலைப்பில் என்னிடம் குறிப்பிட்ட தகவல் இல்லை.\n"
        "இந்திய தேர்தல் ஆணைய உதவியாளராக, நான் உதவ முடியும்:\n"
        "- வாக்காளர் பதிவு மற்றும் EPIC அட்டை\n"
        "- வாக்குப்பதிவு மையத்தை கண்டறிவது\n"
        "- தேர்தல் செயல்முறை மற்றும் காலவரிசை\n"
        "- ECI தகவல் மற்றும் உதவி எண்\n\n"
        "தேர்தல் தொடர்பான தலைப்புகளை கேளுங்கள், அல்லது தேர்தல் உதவி எண்ணை அழைக்கவும்: 1950 (இலவசம்)."
    ),
}

# ── Out-of-scope fallback (live results / live data queries) ──────────────────
# These are queries asking for LIVE or SPECIFIC results that the static KB
# cannot provide. Serving a generic process-level answer would be misleading.
OOS_FALLBACK = {
    "en": (
        "I don't have access to live or specific election result data.\n\n"
        "For official Tamil Nadu or other state election results, please visit:\n"
        "- ECI Results: https://results.eci.gov.in\n"
        "- ECI Official Site: https://eci.gov.in\n"
        "- Election Commission of Tamil Nadu: https://www.elections.tn.gov.in\n\n"
        "I can help with voter registration, EPIC card, polling booths, and the general election process."
    ),
    "ta": (
        "என்னிடம் நேரடி தேர்தல் முடிவு தகவல் இல்லை.\n\n"
        "அதிகாரப்பூர்வ தமிழ்நாடு அல்லது பிற மாநில தேர்தல் முடிவுகளுக்கு:\n"
        "- ECI முடிவுகள்: https://results.eci.gov.in\n"
        "- ECI அதிகாரப்பூர்வ தளம்: https://eci.gov.in\n"
        "- தமிழ்நாடு தேர்தல் ஆணையம்: https://www.elections.tn.gov.in\n\n"
        "வாக்காளர் பதிவு, EPIC அட்டை, வாக்குச்சாவடி, மற்றும் தேர்தல் செயல்முறை பற்றி கேளுங்கள்."
    ),
}

OOS_LINKS = [
    {"label": "Official ECI Results",       "url": "https://results.eci.gov.in",             "type": "web"},
    {"label": "ECI Official Site",           "url": "https://eci.gov.in",                    "type": "web"},
    {"label": "TN Election Commission",      "url": "https://www.elections.tn.gov.in",       "type": "web"},
]

FALLBACK_LINKS = [
    {"label": "Call Helpline 1950 / 1950 அழைக்கவும்", "url": "tel:1950",                    "type": "phone"},
    {"label": "voters.eci.gov.in",                      "url": "https://voters.eci.gov.in",   "type": "web"},
]

# ── Out-of-scope (live data) keyword patterns ─────────────────────────────────
# Queries matching these patterns ask for live/specific data the KB cannot give.
# They must NOT be routed to FAISS — that would cause hallucinations.
_RESULT_KEYWORDS = [
    # English result/winner keywords
    r"\b(election|tn|tamilnadu|tamil\s*nadu|state|lok\s*sabha|assembly)\b.{0,40}\b(result|results|winner|winners|won|outcome|outcome|tally|tallies|score|scores|seat|seats|standing|standings|verdict|elected)\b",
    r"\b(result|results|winner|winners|won|who\s+won|who\s+won\s+the|winning|elected|victory|victories)\b.{0,40}\b(election|constituency|seat|seats|ward|tn|tamilnadu|tamil\s*nadu)\b",
    r"\b(how\s+many\s+seats?|seat\s+count|party\s+won|bjp|dmk|aiadmk|admk|congress|tvk)\b.{0,40}\b(won|win|seat|result|election)\b",
    r"\b(current\s+)?(mla|mp|cm|chief\s+minister|minister)\b.{0,30}\b(of|for|in|at)\b.{0,30}\b(tn|tamilnadu|tamil\s*nadu|tamil)\b",
    # standalone short result queries
    r"^(tn|tamilnadu|tamil\s*nadu|tamilnadu)\s+(election|lok\s*sabha|assembly)\s+(result|results|winner|won|seat|tally)s?$",
    r"^(election|assembly|lok\s*sabha)\s+(result|results)\s*(of|for|in|2021|2024|2026)?\s*(tn|tamilnadu|tamil\s*nadu)?$",
    r"^who\s+won\b",
    r"^(which|what)\s+(party|candidate)\b.{0,40}\b(won|win|winner|elected)\b",
    # Tamil result keywords
    r"தேர்தல்\s*முடிவு",
    r"வெற்றி\s*பெற்றவர்",
    r"யார்\s*வென்றார்",
    r"DMK|AIADMK|ADMK|TVK\b.{0,20}வென்ற",
]
_RESULT_RE = re.compile(
    "|".join(_RESULT_KEYWORDS),
    re.IGNORECASE | re.UNICODE,
)

# ── Global state ──────────────────────────────────────────────────────────────
state: dict = {}


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    import sys
    print(f"[STARTUP] App initialization started. PORT={os.getenv('PORT', 8080)}", flush=True)
    sys.stdout.flush()
    
    print("[STARTUP] Loading bilingual TF-IDF models ...", flush=True)
    t0 = time.time()

    state["vectorizers"] = {}
    state["matrices"] = {}
    state["stores"]  = {}

    for lang in SUPPORTED_LANGS:
        vec_path   = MODEL_DIR / f"tfidf_vectorizer_{lang}.pkl"
        mat_path   = MODEL_DIR / f"tfidf_matrix_{lang}.npy"
        store_path = MODEL_DIR / f"qa_store_{lang}.json"

        print(f"[STARTUP] Checking files for {lang}: vec={vec_path.exists()}, mat={mat_path.exists()}, store={store_path.exists()}", flush=True)

        if not vec_path.exists() or not mat_path.exists() or not store_path.exists():
            raise RuntimeError(
                f"Trained TF-IDF files for lang='{lang}' not found.\n"
                f"    Run: python train.py  (or  python train.py --lang {lang})"
            )

        with open(vec_path, "rb") as f:
            state["vectorizers"][lang] = pickle.load(f)
        state["matrices"][lang] = np.load(mat_path)

        with open(store_path, "r", encoding="utf-8") as f:
            state["stores"][lang] = json.load(f)

        print(f"[STARTUP] [{lang.upper()}] {state['matrices'][lang].shape[0]} vectors loaded", flush=True)

    state["sessions"] = {}

    # Load fetched QA pairs from scrapers and RSS
    print("[STARTUP] Loading fetched data (RSS and scrapers)...", flush=True)
    fetched_qa = []
    try:
        print("[STARTUP] Calling get_all_cached_results()...", flush=True)
        cached = get_all_cached_results()
        fetched_qa.extend(cached)
        print(f"[STARTUP] Cached results: {len(cached)} entries", flush=True)
        
        print("[STARTUP] Calling get_rss_qa_pairs()...", flush=True)
        rss = get_rss_qa_pairs()
        fetched_qa.extend(rss)
        print(f"[STARTUP] RSS data: {len(rss)} entries", flush=True)
        print(f"[STARTUP] Total fetched: {len(fetched_qa)} entries", flush=True)
    except Exception as e:
        print(f"[STARTUP] Warning: Failed to load fetched data: {type(e).__name__}: {e}", flush=True)
        fetched_qa = []

    # Add fetched QA to English store (assuming they are in English)
    if "en" in state["stores"] and fetched_qa:
        state["stores"]["en"].extend(fetched_qa)
        print(f"[STARTUP] Added {len(fetched_qa)} fetched entries to EN store", flush=True)

    state["fetched_qa"] = fetched_qa

    elapsed = round(time.time() - t0, 2)
    print(f"[STARTUP] ✓ All models ready in {elapsed}s. App is ready to accept requests.", flush=True)
    sys.stdout.flush()

    yield
    print("[SHUTDOWN] Shutting down.", flush=True)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="India Election Chatbot API — Bilingual",
    description="NLP chatbot for EN + Tamil. No API key. Fully local. OWASP-compliant.",
    version="4.0.0",
    lifespan=lifespan,
)

# NOTE: Static files are mounted AFTER all API routes (at the bottom of this file)
# to prevent StaticFiles from intercepting POST /api/chat with a 405.

# ── Security Middleware ───────────────────────────────────────────────────────
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"],  # Use specific domains in production
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific origins in production
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
    max_age=600,
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https:; "
        "connect-src 'self';"
    )
    return response


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str = Field(..., min_length=MIN_MSG_LEN, max_length=MAX_MSG_LEN)
    lang:        Literal["en", "ta"] = "en"
    session_id:  Optional[str] = "default"
    use_history: bool = True

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()

        # Length
        if len(v) < MIN_MSG_LEN:
            raise ValueError(f"Message too short (min {MIN_MSG_LEN} chars)")
        if len(v) > MAX_MSG_LEN:
            raise ValueError(f"Message too long (max {MAX_MSG_LEN} chars)")

        # Allowed character whitelist
        # Tamil Unicode block U+0B80–U+0BFF is allowed explicitly
        # Also allow standard ASCII letters, digits, spaces, .,?'-
        if not ALLOWED_CHARS_RE.match(v):
            # Strip disallowed chars instead of rejecting outright (better UX)
            v = re.sub(r"[^\w\s\u0B80-\u0BFF?.@,$'\-]", "", v, flags=re.UNICODE).strip()
            if len(v) < MIN_MSG_LEN:
                raise ValueError("Message contained only disallowed characters")

        # Prevent prompt injection / XSS
        lowered = v.lower()
        blocked_patterns = [
            "<script", "javascript:", "onerror=", "onclick=",
            "drop table", "select *", "--",
            "ignore previous", "disregard instructions",
        ]
        for pat in blocked_patterns:
            if pat in lowered:
                raise ValueError("Message contains disallowed content")

        return v

    @field_validator("session_id")
    @classmethod
    def validate_session(cls, v: str) -> str:
        if v and not re.match(r"^[a-zA-Z0-9_\-]{1,64}$", v):
            return "default"
        return v


class LinkItem(BaseModel):
    label: str
    url:   str
    type:  str


class ChatResponse(BaseModel):
    answer:           str
    confidence:       float
    matched_question: Optional[str] = None
    tags:             list[str] = []
    links:            list[LinkItem] = []
    lang:             str
    session_id:       str


# ── NLP Helpers ───────────────────────────────────────────────────────────────
def normalise(text: str) -> str:
    """Lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def is_irrelevant_message(message: str) -> bool:
    """Detect noisy or repeated input that should not influence history-based retrieval."""
    if len(message) < 3:
        return False
    if re.fullmatch(r"(.)\1{2,}", message):
        return True
    if re.fullmatch(r"([a-z])\1{2,}", message):
        return True
    if re.fullmatch(r"(.+?)\1{2,}", message):
        return True
    if re.search(r"\b(dummy|unreal|not\s+real|irrelevant|nonsense|random|blah|xxxxx|asdf+)\b", message):
        return True
    # Detect short random strings (e.g., "sASA", "xcmcds")
    if len(message) <= 10 and re.match(r"^[a-zA-Z]{3,10}$", message) and not re.search(r"[aeiouAEIOU]", message):
        return True  # No vowels, likely gibberish
    if re.match(r"^[a-zA-Z]{1,10}$", message) and message.isupper() and len(message) < 5:
        return True  # Short all caps
    return False


def encode_query(query: str, lang: str) -> np.ndarray:
    vectorizer = state["vectorizers"].get(lang)
    if vectorizer is None:
        return np.array([])
    return vectorizer.transform([normalise(query)]).toarray().astype("float32")


def tfidf_search(query: str, lang: str) -> list[dict]:
    """Search the language-specific TF-IDF matrix."""
    vectorizer = state["vectorizers"].get(lang)
    matrix = state["matrices"].get(lang)
    store = state["stores"].get(lang)

    if vectorizer is None or matrix is None or store is None:
        return []

    query_vec = encode_query(query, lang)
    if query_vec.size == 0:
        return []

    similarities = cosine_similarity(query_vec, matrix)[0]
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]  # top k in descending order

    results = []
    for idx in top_indices:
        score = float(similarities[idx])
        if score < LOW_CONFIDENCE_THRESHOLD:
            continue
        entry = store[idx].copy()
        entry["score"] = score
        results.append(entry)

    return results


def enrich_query(message: str, history: list[dict]) -> str:
    """Prepend recent user context to improve pronoun resolution."""
    if not history:
        return message
    recent = [h["content"] for h in history[-4:] if h["role"] == "user"]
    recent.append(message)
    return " ".join(recent[-3:])


def clean_answer(text: str) -> str:
    """
    Post-process the raw answer text:
    - Remove duplicate lines
    - Strip leading/trailing whitespace per line
    - Collapse more-than-2 consecutive blank lines
    """
    lines   = text.splitlines()
    seen    = set()
    cleaned = []
    blanks  = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            blanks += 1
            if blanks <= 2:
                cleaned.append("")
        else:
            blanks = 0
            # deduplicate bullet points and step lines
            key = stripped.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(stripped)

    return "\n".join(cleaned).strip()


def is_out_of_scope(message: str) -> bool:
    """Return True if the query asks for live election results/winners that the
    static knowledge base cannot reliably answer. Returning live data from a
    static FAISS store would be hallucination."""
    return bool(_RESULT_RE.search(message))


def resolve_special_query(message: str, lang: str) -> Optional[str]:
    msg = normalise(message)
    if is_irrelevant_message(msg):
        return "__irrelevant__"

    # ── Live-data guard: block before FAISS search ─────────────────────────────
    # Must come early — before any FAISS retrieval — so the bot never serves a
    # generic process answer as if it were actual result data.
    if is_out_of_scope(msg):
        return "__out_of_scope__"

    if re.match(r'^(h+i+|hello+|hey+|good\s+(morning|afternoon|evening)|greetings?)\b', msg):
        return "__greeting__"

    if "what is election" in msg or "election means" in msg or "meaning of election" in msg:
        return "__what_is_election__"

    if "vvpat" in msg and "evm" in msg:
        return "__evm_and_vvpat__"
    if "vvpat" in msg or "voter verifiable paper audit trail" in msg:
        return "What is VVPAT voter verifiable paper audit trail?"

    # ── EPIC / Voter ID — intent-aware routing ─────────────────────────────────
    # OLD: one-line catch-all routed everything to "How do I apply for voter ID?"
    # which is NOT in the FAISS store → closest match was Form-6 answer (wrong!).
    # NEW: each sub-intent maps to the exact question string present in the store.

    # 1. E-EPIC / digital voter ID — what is it, how to download
    if re.search(r"\be-?epic\b", msg) or (
        "epic" in msg and any(w in msg for w in ["what", "download", "digital", "pdf", "get it", "is it"])
    ):
        return "What is E-EPIC and how do I download my digital voter ID?"

    # 2. Voter ID / EPIC — apply / new / lost / damaged
    if re.search(r"\b(voter[\s_-]?id|epic[\s_-]?card|voter[\s_-]?card)\b", msg) and any(
        w in msg for w in ["apply", "get", "lost", "damage", "replace", "new", "how", "obtain"]
    ):
        return "How do I register as a voter in India?"

    # 3. Bare "epic" mention with no download/apply context → digital card info
    if "epic" in msg and "voter id" not in msg:
        return "What is E-EPIC and how do I download my digital voter ID?"

    # 4. Voter ID / status check
    if (
        "application status" in msg
        or ("status" in msg and any(word in msg for word in ["application", "registration", "register", "form 6", "epic", "voter id"]))
    ):
        return "How to check voter ID status and electoral roll?"

    if "find my polling booth" in msg or ("polling booth" in msg and "find" in msg):
        return "How do I find my polling booth?"
    return None


# ── Core inference ────────────────────────────────────────────────────────────
def generate_answer(message: str, lang: str, session_id: str, use_history: bool = True) -> ChatResponse:
    session_key = f"{session_id}:{lang}"
    history     = state["sessions"].setdefault(session_key, [])
    special     = resolve_special_query(message, lang)

    if special == "__irrelevant__":
        fallback_text = clean_answer(FALLBACK.get(lang, FALLBACK["en"]))
        history.append({"role": "assistant", "content": fallback_text})
        return ChatResponse(
            answer=fallback_text,
            confidence=0.0,
            matched_question=None,
            tags=[],
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    # ── Out-of-scope: live result queries ──────────────────────────────────────
    if special == "__out_of_scope__":
        oos_text = clean_answer(OOS_FALLBACK.get(lang, OOS_FALLBACK["en"]))
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": oos_text})
        return ChatResponse(
            answer=oos_text,
            confidence=0.0,
            matched_question=None,
            tags=["out-of-scope", "live-results"],
            links=[LinkItem(**lk) for lk in OOS_LINKS],
            lang=lang,
            session_id=session_id,
        )

    if special == "__greeting__":
        greeting = (
            "Hello! I am the Election Commission of India assistant. You can ask me about voter registration, EPIC, polling booths, election procedures, or helpline information."
            if lang == "en"
            else "வணக்கம்! நான் இந்திய தேர்தல் ஆணைய உதவியாளன். வாக்காளர் பதிவு, EPIC, வாக்குச்சாவடி, தேர்தல் செயல்முறை அல்லது உதவி எண்ணை பற்றி கேளுங்கள்."
        )
        history.append({"role": "assistant", "content": greeting})
        return ChatResponse(
            answer=greeting,
            confidence=1.0,
            matched_question="Greeting",
            tags=[],
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    if special == "__what_is_election__":
        election_def = (
            "An election is the formal process by which citizens choose representatives or decide on public issues. In India, elections are conducted by the Election Commission, and voters cast ballots at assigned polling booths following a regulated schedule."
            if lang == "en"
            else "ஒரு தேர்தல் என்பது குடிமக்கள் பிரதிநிதிகளை தேர்வு செய்வதற்கோ அல்லது பொது விஷயங்களில் முடிவு செய்வதற்கோ பயன்படுத்தப்படும் அதிகாரப்பூர்வ செயல்முறை. இந்தியாவில், இந்திய தேர்தல் ஆணையம் தேர்தல்களை நடத்துகிறது, மற்றும் வாக்காளர்கள் ஒதுக்கப்பட்ட வாக்குச்சாவடிகளில் சங்கடப்பட்ட அட்டைகளை செலுத்துகிறார்கள்."
        )
        history.append({"role": "assistant", "content": election_def})
        return ChatResponse(
            answer=election_def,
            confidence=1.0,
            matched_question="What is an election?",
            tags=[],
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    if special == "__evm_and_vvpat__":
        combined_answer = (
            "EVM (Electronic Voting Machine) is the device used to cast and record votes in Indian elections. VVPAT (Voter Verifiable Paper Audit Trail) is attached to the EVM and prints a paper slip that lets the voter verify their vote before it is sealed. Together, EVM and VVPAT provide both electronic vote recording and a physical verification record."
            if lang == "en"
            else "EVM (மின்னணு வாக்குப்பதிவு இயந்திரம்) இந்திய தேர்தலில் வாக்குகளை பதிவு செய்தும், EVM க்கு இணைக்கப்படும் VVPAT (வாக்காளர் செப்திகரிக்கும் காகித ஆடிட் ட்ரெயில்) வாக்கை வைத்திருப்பதற்கு ஒரு தொடக்கக் காகிதத்தை அச்சிடுகிறது. இணைந்து, EVM மற்றும் VVPAT மின்னணு பதிவீடு மற்றும் உண்மையான உறுதிப்படுத்தல் பதிவை வழங்குகின்றன."
        )
        history.append({"role": "assistant", "content": combined_answer})
        return ChatResponse(
            answer=combined_answer,
            confidence=1.0,
            matched_question="What is EVM and VVPAT?",
            tags=[],
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    query_text = special or message
    enriched    = enrich_query(query_text, history) if use_history else query_text

    # Search
    candidates = tfidf_search(enriched, lang)

    # Update history
    history.append({"role": "user", "content": message})
    if len(history) > MAX_HISTORY * 2:
        history[:] = history[-(MAX_HISTORY * 2):]

    if not candidates:
        fallback_text = clean_answer(FALLBACK.get(lang, FALLBACK["en"]))
        history.append({"role": "assistant", "content": fallback_text})
        return ChatResponse(
            answer=fallback_text,
            confidence=0.0,
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    best_score = candidates[0]["score"]
    best = candidates[0]
    answer_text = clean_answer(best["answer"])

    if best_score < CONFIDENCE_THRESHOLD:
        if best_score >= LOW_CONFIDENCE_THRESHOLD and answer_text:
            answer_text = (
                answer_text
                + ("\n\n" + (" குறைந்த நம்பகத்தன்மை; தயவுசெய்து ECI போர்டல் மூலம் சரிபார்க்கவும்." if lang == "ta" else " Low confidence answer; please verify on the ECI portal."))
            )
            links = [LinkItem(**lk) for lk in best.get("links", [])] or [LinkItem(**lk) for lk in FALLBACK_LINKS]
            history.append({"role": "assistant", "content": answer_text})
            return ChatResponse(
                answer=answer_text,
                confidence=round(best_score, 4),
                matched_question=best.get("question"),
                tags=best.get("tags", []),
                links=links,
                lang=lang,
                session_id=session_id,
            )
        fallback_text = clean_answer(FALLBACK.get(lang, FALLBACK["en"]))
        history.append({"role": "assistant", "content": fallback_text})
        return ChatResponse(
            answer=fallback_text,
            confidence=best_score,
            links=[LinkItem(**lk) for lk in FALLBACK_LINKS],
            lang=lang,
            session_id=session_id,
        )

    # Strong answer
    answer_text = clean_answer(best["answer"])

    # Do not append a second result into the same chatbot response.
    # This prevents mixed-topic replies for short or ambiguous queries.
    links = [LinkItem(**lk) for lk in best.get("links", [])]
    history.append({"role": "assistant", "content": answer_text})

    return ChatResponse(
        answer=answer_text,
        confidence=round(best["score"], 4),
        matched_question=best.get("question"),
        tags=best.get("tags", []),
        links=links,
        lang=lang,
        session_id=session_id,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        return {
            "status":   "ok",
            "model":    "tfidf",
            "languages": {
                lang: state["matrices"][lang].shape[0]
                for lang in SUPPORTED_LANGS
                if lang in state.get("matrices", {})
            },
        }
    except:
        return {"status": "starting", "message": "Models loading..."}

@app.get("/ready")
def ready():
    """Simple readiness check that doesn't depend on models"""
    return {"status": "ready"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Chat endpoint.
    Body: { "message": "...", "lang": "en" | "ta", "session_id": "abc", "use_history": true }
    """
    try:
        return generate_answer(req.message, req.lang, req.session_id, req.use_history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/stats")
def stats():
    return {
        "languages": SUPPORTED_LANGS,
        "vectors": {
            lang: state["matrices"][lang].shape[0]
            for lang in SUPPORTED_LANGS
            if lang in state.get("matrices", {})
        },
        "active_sessions":      len(state.get("sessions", {})),
        "model":                "tfidf",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "top_k":                TOP_K,
    }


@app.delete("/api/session/{session_id}")
def clear_session(session_id: str):
    state["sessions"].pop(session_id, None)
    return {"cleared": session_id}


# ── Static frontend (mounted LAST so API routes are never blocked) ────────────


@app.get("/", include_in_schema=False)
def root():
    """Serve the SPA index.html for the root path."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "Election Chatbot API running. POST /api/chat"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return 204 for favicon requests to suppress 404 log noise."""
    from fastapi.responses import Response
    return Response(status_code=204)


# Mount static assets (CSS, JS, images) — must be AFTER all @app.* route decorators
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static_assets")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, workers=1, reload=False)
