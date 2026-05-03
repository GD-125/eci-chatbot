"""
fetchers/eci_results_scraper.py
────────────────────────────────
Scrapes ECI election results pages (results.eci.gov.in) and converts
constituency-level results into QA pairs for the chatbot.

Supports:
  - Tamil Nadu Assembly 2021
  - Lok Sabha 2024 (Tamil Nadu constituencies)
  - Extensible to other states/elections

Usage:
    python fetchers/eci_results_scraper.py --election tn2021
    python fetchers/eci_results_scraper.py --election ls2024_tn

Fairness / Legal notes:
  - Data is sourced exclusively from ECI official portal
  - Every answer includes "Source: Election Commission of India"
  - Every answer includes the fetch date (transparency)
  - Data is NOT altered or inferred — stored verbatim from ECI
"""

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import requests
    from bs4 import BeautifulSoup
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Election definitions ──────────────────────────────────────────────────────
ELECTIONS = {
    "tn2021": {
        "name":        "Tamil Nadu Assembly Election 2021",
        "year":        2021,
        "state":       "Tamil Nadu",
        "state_code":  "S22",
        "url_base":    "https://results.eci.gov.in/ResultAcGenMay2021/",
        "index_url":   "https://results.eci.gov.in/ResultAcGenMay2021/index.htm",
        "total_seats": 234,
    },
    "ls2024": {
        "name":       "Lok Sabha General Election 2024",
        "year":       2024,
        "state":      "India",
        "url_base":   "https://results.eci.gov.in/ResultAcGenJune2024/",
        "index_url":  "https://results.eci.gov.in/ResultAcGenJune2024/index.htm",
        "total_seats": 543,
    },
    "ls2024_tn": {
        "name":       "Lok Sabha 2024 — Tamil Nadu",
        "year":       2024,
        "state":      "Tamil Nadu",
        "state_code": "S22",
        "url_base":   "https://results.eci.gov.in/ResultAcGenJune2024/",
        "index_url":  "https://results.eci.gov.in/ResultAcGenJune2024/statewiseS22.htm",
        "total_seats": 39,
    },
}

RESULTS_CACHE_DIR = Path("data/results_cache")
REQUEST_DELAY     = 1.5   # seconds between requests — be polite to ECI servers
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (ECI-Civic-Chatbot/1.0; educational civic tool; "
        "contact: your@email.com)"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


# ── Scraper core ──────────────────────────────────────────────────────────────
def _safe_get(url: str, retries: int = 2) -> Optional[str]:
    """GET a URL with retry + delay. Returns HTML text or None."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.text
        except requests.RequestException as e:
            logger.warning(f"GET {url} attempt {attempt+1} failed: {e}")
            time.sleep(REQUEST_DELAY * 2)
    return None


def _parse_number(text: str) -> int:
    """Parse '1,23,456' or '123456' to int."""
    cleaned = re.sub(r"[^\d]", "", text)
    return int(cleaned) if cleaned else 0


def scrape_constituency_results(html: str, election_name: str, year: int) -> list[dict]:
    """
    Parse a constituency result table from ECI results HTML.
    Returns a list of result dicts.
    """
    soup    = BeautifulSoup(html, "html.parser")
    results = []

    # ECI uses various table structures — try multiple selectors
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cols) < 4:
                continue

            # Heuristic: look for rows with a candidate name + party + votes
            # ECI tables typically: Candidate | Party | Votes | ... | Status
            candidate_col = cols[0] if cols else ""
            party_col     = cols[1] if len(cols) > 1 else ""
            votes_col     = cols[2] if len(cols) > 2 else ""
            status_col    = cols[-1] if cols else ""

            votes = _parse_number(votes_col)
            if votes < 100:   # skip header/footer rows
                continue

            results.append({
                "candidate": candidate_col,
                "party":     party_col,
                "votes":     votes,
                "status":    status_col,   # "Won" / "Lost" etc.
            })

    return results


def build_result_qa(
    constituency:  str,
    winner:        str,
    party:         str,
    votes:         int,
    margin:        int,
    runner_up:     str,
    runner_party:  str,
    election_name: str,
    year:          int,
    url:           str,
) -> dict:
    """Convert a single constituency result into a QA pair."""
    fetch_date = datetime.now().strftime("%d %b %Y")

    answer = (
        f"{election_name} — {constituency} Constituency Result:\n\n"
        f"🏆 WINNER: {winner} ({party})\n"
        f"   Votes:  {votes:,}\n"
        f"   Margin: {margin:,} votes over {runner_up} ({runner_party})\n\n"
        f"Source: Election Commission of India (results.eci.gov.in)\n"
        f"Data fetched: {fetch_date}"
    )

    return {
        "id":       f"result_{year}_{re.sub(r'[^a-z0-9]', '_', constituency.lower())}",
        "lang":     "en",
        "question": f"Who won {constituency} in the {year} election?",
        "answer":   answer,
        "tags":     ["election result", constituency, party, str(year), "winner"],
        "links":    [{"label": "ECI Results", "url": url, "type": "web"}],
        "_source":  "eci_results_scraper",
        "_fetched_at": datetime.now().isoformat(),
    }


def scrape_election(election_key: str) -> list[dict]:
    """
    Scrape a full election result page and return QA pairs.
    election_key: one of the keys in ELECTIONS dict.
    """
    if not DEPS_AVAILABLE:
        logger.error("requests / beautifulsoup4 not installed. Run: pip install requests beautifulsoup4")
        return []

    config = ELECTIONS.get(election_key)
    if not config:
        logger.error(f"Unknown election key: {election_key}. Available: {list(ELECTIONS.keys())}")
        return []

    logger.info(f"Scraping: {config['name']} from {config['index_url']}")
    html = _safe_get(config["index_url"])
    if not html:
        logger.error(f"Failed to fetch results page for {election_key}")
        return []

    # Parse the index page for constituency links
    soup  = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", href=True)
    constituency_urls = [
        config["url_base"] + a["href"]
        for a in links
        if re.search(r"Constituency|cons|ac\d+", a["href"], re.I)
    ]

    if not constituency_urls:
        logger.warning("No constituency links found — will try parsing the index page directly")
        # Some ECI pages show all results on the index page
        raw_results = scrape_constituency_results(html, config["name"], config["year"])
        logger.info(f"Parsed {len(raw_results)} candidate rows from index page")
        return []

    logger.info(f"Found {len(constituency_urls)} constituency pages to scrape")

    qa_pairs = []
    for i, url in enumerate(constituency_urls[:config.get("total_seats", 50)]):
        logger.info(f"  [{i+1}/{len(constituency_urls)}] {url}")
        c_html = _safe_get(url)
        if not c_html:
            continue

        # Extract constituency name from URL or page title
        c_soup = BeautifulSoup(c_html, "html.parser")
        title  = c_soup.find("title")
        cname  = title.get_text(strip=True).split("-")[0].strip() if title else f"Constituency {i+1}"

        rows = scrape_constituency_results(c_html, config["name"], config["year"])
        if not rows:
            continue

        # Sort by votes descending — top 2 = winner and runner-up
        rows.sort(key=lambda r: r["votes"], reverse=True)
        winner    = rows[0] if rows else {}
        runner_up = rows[1] if len(rows) > 1 else {}

        if not winner:
            continue

        margin = winner["votes"] - runner_up.get("votes", 0) if runner_up else 0

        qa = build_result_qa(
            constituency=cname,
            winner=winner["candidate"],
            party=winner["party"],
            votes=winner["votes"],
            margin=margin,
            runner_up=runner_up.get("candidate", "N/A"),
            runner_party=runner_up.get("party", ""),
            election_name=config["name"],
            year=config["year"],
            url=url,
        )
        qa_pairs.append(qa)

    logger.info(f"Scraped {len(qa_pairs)} constituency results for {election_key}")
    return qa_pairs


def save_results_cache(election_key: str, qa_pairs: list[dict]) -> None:
    """Save scraped results to local cache."""
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RESULTS_CACHE_DIR / f"{election_key}.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"scraped_at": datetime.now().isoformat(), "items": qa_pairs}, f,
                  ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(qa_pairs)} result QA pairs to {cache_file}")


def load_results_cache(election_key: str) -> list[dict]:
    """Load cached results for an election."""
    cache_file = RESULTS_CACHE_DIR / f"{election_key}.json"
    if not cache_file.exists():
        return []
    with open(cache_file, encoding="utf-8") as f:
        return json.load(f).get("items", [])


def get_all_cached_results() -> list[dict]:
    """Return all cached result QA pairs across all elections."""
    RESULTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    for cache_file in RESULTS_CACHE_DIR.glob("*.json"):
        with open(cache_file, encoding="utf-8") as f:
            all_pairs.extend(json.load(f).get("items", []))
    return all_pairs


# ── Standalone run ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Scrape ECI election results")
    parser.add_argument(
        "--election", choices=list(ELECTIONS.keys()), default="tn2021",
        help="Which election to scrape"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and print without saving"
    )
    args = parser.parse_args()

    pairs = scrape_election(args.election)
    print(f"\nScraped {len(pairs)} result QA pairs for '{args.election}'\n")

    if not args.dry_run:
        save_results_cache(args.election, pairs)
        print(f"Saved to data/results_cache/{args.election}.json")

    for p in pairs[:3]:
        print(f"  Q: {p['question']}")
        print(f"  A: {p['answer'][:200]}\n")
