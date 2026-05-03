"""
train.py — Improved bilingual training for India Election Chatbot.

Improvements over v1:
  - Separate FAISS indexes per language (en, ta) for precision
  - Question augmentation (aliases + paraphrases) to boost recall
  - Robust error handling and validation before saving
  - Normalisation pipeline: strip, lowercase, collapse whitespace
  - Progress reporting with confidence self-test after training

Usage:
  python train.py           # train both languages
  python train.py --lang en # train English only
  python train.py --lang ta # train Tamil only
  python train.py --test    # self-test after training

"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data/qa_data.json")
MODEL_DIR   = Path("models")
EMBED_MODEL = "all-mpnet-base-v2"   #420M params, better quality, 768-dim vectors

SUPPORTED_LANGS = ["en", "ta"]

# ── Question augmentations (paraphrases added at training time) ───────────────

AUGMENTATIONS: dict[str, list[str]] = {
    "What is the Election Commission of India?": [
        "Tell me about ECI",
        "ECI details",
        "What does ECI do?",
        "Election Commission information",
        "What is Article 324?",
    ],
    "How do I register as a voter in India?": [
        "How to register voter",
        "Voter registration process",
        "How to get voter ID",
        "How to apply for voter card",
        "How to enroll as voter",
        "Voter enrolment steps",
    ],
    "What is Form 6 and how to fill it?": [
        "Form 6 voter registration",
        "Download Form 6",
        "Voter registration form",
        "How to fill voter form",
        "Form 6 details",
    ],
    "What is EVM electronic voting machine?": [
        "What is EVM?",
        "Electronic Voting Machine explained",
        "How does EVM work?",
        "EVM details",
        "Voting machine India",
    ],
    "What is VVPAT voter verifiable paper audit trail?": [
        "What is VVPAT?",
        "VVPAT paper slip",
        "How VVPAT works",
        "Vote verification paper",
        "Paper trail voting",
    ],
    "What is the complete election process in India step by step?": [
        "Election process India",
        "How elections work in India",
        "India election steps",
        "What happens in Indian elections",
        "Election procedure",
    ],
    "What is NOTA none of the above?": [
        "What is NOTA?",
        "None of the above option",
        "NOTA voting",
        "How to vote NOTA",
    ],
    "What is the voter helpline number?": [
        "Election helpline",
        "Voter helpline",
        "1950 helpline",
        "Contact ECI",
        "Voter support number",
    ],
    "How do I find my polling booth?": [
        "Find polling booth",
        "Where to vote",
        "My voting center",
        "Polling station location",
        "Where is my booth",
    ],
    "Can I vote without Voter ID card?": [
        "Vote without voter ID",
        "Alternative documents for voting",
        "Documents needed to vote",
        "No voter card can I vote",
        "Aadhaar card for voting",
    ],
    # Tamil augmentations
    "இந்திய தேர்தல் ஆணையம் என்றால் என்ன?": [
        "ECI என்றால் என்ன",
        "தேர்தல் ஆணையம் பற்றி சொல்லுங்கள்",
        "தேர்தல் ஆணையம் என்ன செய்கிறது",
        "324வது பிரிவு என்றால் என்ன",
    ],
    "வாக்காளராக பதிவு செய்வது எப்படி?": [
        "வாக்காளர் பதிவு செய்வது எப்படி",
        "வாக்காளர் அட்டை எப்படி பெறுவது",
        "வாக்காளர் பதிவு செயல்முறை",
        "படிவம் 6 எப்படி பெறுவது",
    ],
    "வாக்குப்பதிவு மையத்தை எவ்வாறு கண்டுபிடிப்பது?": [
        "வாக்குப்பதிவு மையம் எங்கே",
        "என் மையத்தை கண்டுபிடி",
        "வாக்கு போடும் இடம்",
    ],
}


# ── Helper: normalise text ────────────────────────────────────────────────────
def normalise(text: str) -> str:
    import re
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # For English: lowercase. Tamil characters are case-insensitive by default.
    return text.lower()


# ── Load + validate data ──────────────────────────────────────────────────────
def load_and_validate(path: Path) -> list[dict]:
    if not path.exists():
        print(f" Data file not found: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON parse error in {path}: {e}")
            sys.exit(1)

    errors = []
    for i, item in enumerate(data):
        for field in ("id", "lang", "question", "answer"):
            if field not in item or not str(item[field]).strip():
                errors.append(f"  Row {i}: missing or empty field '{field}'")
        if item.get("lang") not in SUPPORTED_LANGS:
            errors.append(f"  Row {i} id={item.get('id')}: unsupported lang '{item.get('lang')}'")

    if errors:
        print("Validation errors found in qa_data.json:")
        for e in errors:
            print(e)
        sys.exit(1)

    print(f"Loaded and validated {len(data)} Q&A pairs")
    return data


# ── Build augmented question list ─────────────────────────────────────────────
def augment(data: list[dict]) -> list[dict]:
    """
    For each Q&A pair, add extra paraphrase entries pointing to the same answer.
    This boosts recall without needing more real Q&A data.
    """
    augmented = list(data)
    added = 0
    for item in data:
        extras = AUGMENTATIONS.get(item["question"], [])
        for alt_q in extras:
            augmented.append({
                "id":       item["id"] + "_aug",
                "lang":     item["lang"],
                "question": alt_q,
                "answer":   item["answer"],
                "tags":     item.get("tags", []),
                "links":    item.get("links", []),
                "_augmented": True,
            })
            added += 1
    print(f"Augmented dataset: {len(data)} original + {added} paraphrases = {len(augmented)} total")
    return augmented


# ── Build TF-IDF vectors ─────────────────────────────────────────────────────
def build_tfidf(questions: list[str]) -> tuple[TfidfVectorizer, np.ndarray]:
    normalised = [normalise(q) for q in questions]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(normalised).toarray().astype("float32")
    return vectorizer, tfidf_matrix


# ── Save TF-IDF model ─────────────────────────────────────────────────────────
def save_tfidf(vectorizer: TfidfVectorizer, matrix: np.ndarray, lang: str) -> None:
    vec_path = MODEL_DIR / f"tfidf_vectorizer_{lang}.pkl"
    mat_path = MODEL_DIR / f"tfidf_matrix_{lang}.npy"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    np.save(mat_path, matrix)
    print(f"TF-IDF [{lang}] saved → {vec_path} & {mat_path}  ({matrix.shape[0]} vectors, dim={matrix.shape[1]})")


# ── Save QA store ─────────────────────────────────────────────────────────────
def save_store(data: list[dict], lang: str) -> None:
    store = [
        {
            "id":       item.get("id", ""),
            "question": item["question"],
            "answer":   item["answer"],
            "tags":     item.get("tags", []),
            "links":    item.get("links", []),
        }
        for item in data
    ]
    out_path = MODEL_DIR / f"qa_store_{lang}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    print(f"QA store [{lang}] saved → {out_path}")


# ── Self-test: run a few sample queries after training ────────────────────────
def self_test() -> None:
    print("\nRunning self-test ...")
    tests = [
        ("en", "How do I register as a voter?",        0.20),
        ("en", "What is EVM?",                          0.20),
        ("en", "Tell me about ECI",                     0.20),
        ("ta", "வாக்காளராக பதிவு செய்வது எப்படி?",    0.15),
        ("ta", "EVM என்றால் என்ன",                     0.15),
    ]

    all_passed = True
    for lang, query, min_score in tests:
        vec_path = MODEL_DIR / f"tfidf_vectorizer_{lang}.pkl"
        mat_path = MODEL_DIR / f"tfidf_matrix_{lang}.npy"
        store_path = MODEL_DIR / f"qa_store_{lang}.json"
        if not vec_path.exists() or not mat_path.exists():
            print(f"Skipping [{lang}] test — TF-IDF files not found")
            continue

        with open(vec_path, "rb") as f:
            vectorizer = pickle.load(f)
        matrix = np.load(mat_path)
        with open(store_path, encoding="utf-8") as f:
            store = json.load(f)

        query_vec = vectorizer.transform([normalise(query)]).toarray().astype("float32")
        similarities = cosine_similarity(query_vec, matrix)[0]
        best_idx = np.argmax(similarities)
        score = float(similarities[best_idx])
        result = store[best_idx]["question"] if best_idx >= 0 else "N/A"
        status = "PASS" if score >= min_score else "LOW"
        if score < min_score:
            all_passed = False
        print(f"  {status}  [{lang}] score={score:.3f}  query='{query[:40]}'")
        print(f"            → matched: '{result[:60]}'")

    if all_passed:
        print("\nAll self-tests passed.")
    else:
        print("\nSome tests scored below threshold — consider adding more Q&A pairs.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Election Chatbot NLP model")
    parser.add_argument("--lang",  choices=SUPPORTED_LANGS + ["all"], default="all",
                        help="Language to train (default: all)")
    parser.add_argument("--test",  action="store_true", help="Run self-test after training")
    parser.add_argument("--no-augment", action="store_true", help="Skip question augmentation")
    args = parser.parse_args()

    langs_to_train = SUPPORTED_LANGS if args.lang == "all" else [args.lang]

    print("\n" + "━" * 52)
    print("     India Election Chatbot - Training")
    print("━" * 52 + "\n")

    # 1. Load + validate
    data = load_and_validate(DATA_PATH)

    # 2. Augment
    if not args.no_augment:
        data = augment(data)
    else:
        print("ℹugmentation skipped (--no-augment)")

    MODEL_DIR.mkdir(exist_ok=True)

    # 3. Train per language
    for lang in langs_to_train:
        lang_data = [item for item in data if item["lang"] == lang]
        if not lang_data:
            print(f"\n No data found for lang='{lang}', skipping.")
            continue

        print(f"\n{'─'*40}")
        print(f"  Training [{lang.upper()}]  —  {len(lang_data)} entries")
        print(f"{'─'*40}")

        questions  = [item["question"] for item in lang_data]
        t1         = time.time()
        vectorizer, tfidf_matrix = build_tfidf(questions)
        print(f"  TF-IDF built in {round(time.time()-t1,2)}s  —  shape {tfidf_matrix.shape}")

        save_tfidf(vectorizer, tfidf_matrix, lang)
        save_store(lang_data, lang)

    # 4. Optional self-test
    if args.test:
        self_test()

    print("\n" + "━" * 52)
    print("🎉  Training complete!")
    print("    Start server:  uvicorn app:app --host localhost --port 8080")
    print("━" * 52 + "\n")


if __name__ == "__main__":
    main()
