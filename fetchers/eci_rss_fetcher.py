"""
fetchers/eci_rss_fetcher.py
───────────────────────────
Fetches ECI press releases and notifications from official RSS/XML feeds
and converts them into QA pairs compatible with the chatbot's qa_data.json format.

Usage (standalone):
    python fetchers/eci_rss_fetcher.py

Usage (from app.py background task):
    from fetchers.eci_rss_fetcher import fetch_rss_qa_pairs
"""

import json
import re
import time
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import feedparser
    import requests
    from bs4 import BeautifulSoup
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── RSS Feed Sources ──────────────────────────────────────────────────────────
RSS_FEEDS = {
    "eci_main": {
        "url":   "https://eci.gov.in/feed/",
        "label": "ECI Official",
        "tags":  ["ECI", "press release", "notification"],
    },
    "eci_news": {
        "url":   "https://eci.gov.in/category/media-corner/news/feed/",
        "label": "ECI News",
        "tags":  ["ECI", "news", "election"],
    },
}

# Fallback: manually maintained list of important ECI notification URLs
# (used if RSS is unavailable)
FALLBACK_NOTICES = [
    "https://eci.gov.in/press-releases/",
    "https://eci.gov.in/model-code-of-conduct/",
]

CACHE_FILE   = Path("data/rss_cache.json")
MAX_ENTRIES  = 30          # max feed items to convert to QA pairs
MIN_SUMMARY  = 40          # minimum characters for a useful summary


def _clean_html(raw: str) -> str:
    """Strip HTML tags from feed summary text."""
    text = BeautifulSoup(raw, "html.parser").get_text(separator=" ") if raw else ""
    return re.sub(r"\s+", " ", text).strip()


def _make_id(title: str) -> str:
    """Generate a stable ID from title hash."""
    return "rss_" + hashlib.md5(title.encode()).hexdigest()[:10]


def _parse_date(entry) -> str:
    """Extract a human-readable date from a feed entry."""
    try:
        t = entry.get("published_parsed") or entry.get("updated_parsed")
        if t:
            return datetime(*t[:6], tzinfo=timezone.utc).strftime("%d %b %Y")
    except Exception:
        pass
    return datetime.now().strftime("%d %b %Y")


def fetch_rss_qa_pairs(max_items: int = MAX_ENTRIES) -> list[dict]:
    """
    Fetch RSS feeds from ECI and return a list of QA-formatted dicts.
    Each entry becomes a question like "What is the latest ECI notification about <title>?"
    """
    if not DEPS_AVAILABLE:
        logger.warning("feedparser / beautifulsoup4 not installed. Run: pip install feedparser beautifulsoup4")
        return []

    qa_pairs = []

    for feed_key, feed_info in RSS_FEEDS.items():
        try:
            logger.info(f"Fetching RSS: {feed_info['url']}")
            parsed = feedparser.parse(
                feed_info["url"],
                agent="Mozilla/5.0 (ECI-Civic-Chatbot/1.0; educational)",
                request_headers={"Accept": "application/rss+xml, application/xml, text/xml"},
            )

            for entry in parsed.entries[:max_items]:
                title   = entry.get("title", "").strip()
                summary = _clean_html(entry.get("summary", entry.get("description", "")))
                link    = entry.get("link", "https://eci.gov.in")
                date    = _parse_date(entry)

                if not title or len(summary) < MIN_SUMMARY:
                    continue

                qa_pairs.append({
                    "id":       _make_id(title),
                    "lang":     "en",
                    "question": f"What is the latest ECI update about {title}?",
                    "answer": (
                        f"ECI Update ({date}):\n\n"
                        f"{summary}\n\n"
                        f"Source: Election Commission of India\n"
                        f"Read more: {link}"
                    ),
                    "tags":  feed_info["tags"] + ["latest", "notification"],
                    "links": [{"label": f"Read: {title[:40]}", "url": link, "type": "web"}],
                    "_source":    feed_key,
                    "_fetched_at": datetime.now().isoformat(),
                })

        except Exception as e:
            logger.error(f"RSS fetch failed for {feed_key}: {e}")
            continue

    logger.info(f"RSS fetcher: collected {len(qa_pairs)} entries")
    return qa_pairs


def save_rss_cache(qa_pairs: list[dict]) -> None:
    """Save fetched RSS QA pairs to local cache file."""
    CACHE_FILE.parent.mkdir(exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"fetched_at": datetime.now().isoformat(), "items": qa_pairs}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(qa_pairs)} RSS entries to {CACHE_FILE}")


def load_rss_cache() -> list[dict]:
    """Load cached RSS QA pairs (used when live fetch isn't needed)."""
    if not CACHE_FILE.exists():
        return []
    with open(CACHE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items", [])


def is_cache_stale(max_age_hours: int = 6) -> bool:
    """Return True if the cache is older than max_age_hours."""
    if not CACHE_FILE.exists():
        return True
    with open(CACHE_FILE, encoding="utf-8") as f:
        data = json.load(f)
    fetched_at_str = data.get("fetched_at", "")
    if not fetched_at_str:
        return True
    fetched_at = datetime.fromisoformat(fetched_at_str)
    age_hours = (datetime.now() - fetched_at).total_seconds() / 3600
    return age_hours > max_age_hours


def get_rss_qa_pairs(force_refresh: bool = False) -> list[dict]:
    """
    Main entry point: return RSS-based QA pairs.
    Uses cache if fresh; fetches live if stale or force_refresh=True.
    """
    if not force_refresh and not is_cache_stale():
        logger.info("Using cached RSS data")
        return load_rss_cache()

    pairs = fetch_rss_qa_pairs()
    if pairs:
        save_rss_cache(pairs)
    else:
        # Fall back to cache even if stale
        logger.warning("Live RSS fetch returned nothing — using stale cache")
        return load_rss_cache()
    return pairs


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pairs = get_rss_qa_pairs(force_refresh=True)
    print(f"\nFetched {len(pairs)} RSS-based QA pairs\n")
    for p in pairs[:3]:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer'][:120]}...\n")
