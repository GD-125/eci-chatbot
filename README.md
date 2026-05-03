# 🗳️ ECI Bilingual AI Chatbot

> ⚠️ **This model is under active development.** Features, accuracy, and language coverage are being continuously improved.

An AI-powered, bilingual election assistant built for the **Election Commission of India (ECI)**. It answers voter queries instantly in **English and Tamil**, covering voter registration, EPIC cards, polling booths, and the full election process — deployed serverlessly on **Google Cloud Run**.

---

## 📌 Overview

This project is a domain-specific retrieval-based chatbot purpose-built for ECI topics. It uses TF-IDF vector search with cosine similarity to match user questions to a curated bilingual Q&A knowledge base and returns precise, trusted responses — with no hallucination risk.

**Designed for:**
- Voter registration guidance
- EPIC / Voter ID queries and downloads
- Polling booth lookup support
- Election process and timeline information
- ECI-related FAQs and helpline information

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🌐 Bilingual Support | English (`en`) and Tamil (`ta`) with separate models |
| 🔍 Retrieval-Based NLP | TF-IDF + cosine similarity — fully offline capable |
| 🧠 Session Memory | Per-session conversation history for natural dialogue |
| 📡 Live RSS Integration | Real-time ECI news fetched via RSS feed |
| 🚫 Out-of-Scope Guard | Detects live result queries and redirects to official portals |
| 🔐 OWASP-Compliant | XSS, SQL injection, and prompt injection protection built-in |
| 📦 Containerized | Docker + Google Cloud Run for auto-scaling serverless deployment |
| ⚡ Fast API | FastAPI + Uvicorn async stack for high-throughput responses |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn |
| NLP Engine | TF-IDF Vectorizer + scikit-learn |
| ML / Math | NumPy, scikit-learn |
| Data Validation | Pydantic v2 |
| Web Scraping | BeautifulSoup4, feedparser, requests |
| Containerization | Docker (multi-stage build) |
| Cloud Deployment | Google Cloud Run + Cloud Build |

---

## 🏗️ Architecture

### Retrieval Pipeline

```
User Query
    │
    ▼
Input Validation (OWASP guards, character whitelist)
    │
    ▼
Special Query Resolver (greetings, out-of-scope, intents)
    │
    ▼
TF-IDF Encoding → Cosine Similarity Search (language-specific index)
    │
    ▼
Confidence Scoring
    ├── High confidence  → Return best match answer
    ├── Low confidence   → Return answer with warning note
    └── No match        → Return fallback + ECI helpline links
    │
    ▼
Session History Updated → Response Returned
```

### Security

- **Input whitelist** — allows only Unicode letters (including Tamil block U+0B80–U+0BFF), digits, and safe symbols
- **Blocklist** — catches `<script>`, `onclick=`, SQL injection patterns (`DROP TABLE`, `SELECT *`), and prompt injection phrases
- **Security headers** — `X-Frame-Options`, `Content-Security-Policy`, `X-Content-Type-Options` on all responses

---

## 📂 Repository Structure

```
/
├── app.py                  # Main FastAPI application
├── train.py                # Offline TF-IDF training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── cloudbuild.yaml         # Google Cloud Build + Cloud Run deployment
├── .dockerignore
├── data/
│   └── qa_data.json        # Curated bilingual Q&A dataset
├── models/
│   ├── tfidf_vectorizer_en.pkl
│   ├── tfidf_vectorizer_ta.pkl
│   ├── tfidf_matrix_en.npy
│   ├── tfidf_matrix_ta.npy
│   ├── qa_store_en.json
│   └── qa_store_ta.json
├── fetchers/
│   ├── eci_results_scraper.py
│   └── eci_rss_fetcher.py
└── static/
    └── index.html          # Frontend UI
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Docker (optional, for container deployment)
- A Google Cloud project (optional, for Cloud Run)

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or: source venv/Scripts/activate  # Windows

pip install -r requirements.txt
```

### 2. Train the TF-IDF indexes

The pre-trained model files are included in `models/`. To retrain from the dataset:

```bash
python train.py          # trains both English and Tamil
python train.py --lang en   # train English only
python train.py --lang ta   # train Tamil only
```

### 3. Run locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open: [http://localhost:8080](http://localhost:8080)

---

## 🔌 API Reference

### `POST /api/chat`

```json
{
  "message": "How do I find my polling booth?",
  "lang": "en",
  "session_id": "session123",
  "use_history": true
}
```

**Response:**

```json
{
  "answer": "You can find your polling booth by...",
  "confidence": 0.82,
  "matched_question": "How do I find my polling booth?",
  "tags": ["polling-booth", "voter"],
  "links": [{"label": "voters.eci.gov.in", "url": "https://voters.eci.gov.in", "type": "web"}],
  "lang": "en",
  "session_id": "session123"
}
```

### Other Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check with model status |
| `GET` | `/ready` | Simple readiness probe |
| `GET` | `/api/stats` | Model stats and active sessions |
| `DELETE` | `/api/session/{id}` | Clear a session's conversation history |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t eci-chatbot .

# Run
docker run -p 8080:8080 eci-chatbot
```

---

## ☁️ Google Cloud Run Deployment

```bash
gcloud builds submit --config cloudbuild.yaml --project YOUR_PROJECT_ID
```

The `cloudbuild.yaml` handles build → push → deploy in one command. The service is deployed with:
- **2 vCPU / 2 GiB RAM**
- **Port 8080**, public access enabled (`--allow-unauthenticated`)
- **Region:** `us-central1`

---

## 🗺️ Roadmap

- [ ] Add more Indian regional languages (Hindi, Telugu, Kannada)
- [ ] Admin panel for live dataset editing and QA review
- [ ] Cross-encoder re-ranker for improved retrieval precision
- [ ] Automated CI/CD tests for retrieval pipeline and API
- [ ] RAG (Retrieval-Augmented Generation) for low-confidence queries

---

## 📝 Notes

- This is a **retrieval-based system**, not a general-purpose LLM. It will not fabricate answers — it falls back to official ECI helpline links when confidence is low.
- Ensure all `models/` files are present before starting the server. Run `train.py` if any are missing.
- For live election results, the bot safely redirects users to [results.eci.gov.in](https://results.eci.gov.in).

---

## 📞 ECI Voter Helpline

📱 **1950** (Toll-free, all states)
🌐 [voters.eci.gov.in](https://voters.eci.gov.in)
🌐 [eci.gov.in](https://eci.gov.in)
