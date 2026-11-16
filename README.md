# 🧭 Tourism RAG Assistant

A fully local, citation-grounded **Retrieval-Augmented Generation (RAG)** assistant for
tourists exploring India — currently scoped to **West Bengal**, **Rajasthan**, and
**Uttar Pradesh** — built entirely on official guidebooks and government tourism reports.

Everything runs on your machine: local embeddings, a local vector database, and a local
LLM served via [Ollama](https://ollama.com). No API keys, no cloud calls, no data leaves
your laptop.

---

## ✨ Features

- **Grounded answers only** — the LLM is instructed (via a strict system prompt) to answer
  *only* from retrieved context, cite its sources, and explicitly say when it doesn't know.
- **Section-aware chunking** — PDF text is split using guidebook section headings
  ("Places to Visit", "How to reach", "Food of...", etc.) rather than blind fixed-size windows.
- **Metadata-filtered retrieval** — filter by state, region, city, or document type from the
  Streamlit sidebar.
- **Fully local stack** — `sentence-transformers` for embeddings, `chromadb` for vector
  storage, `ollama` (`llama3.2:1b`) for generation.
- **Evaluation harness** — a CSV of factual / transit / out-of-scope test questions with an
  automated pass/fail keyword check.

---

## 📚 Knowledge Base

| File | Coverage |
|---|---|
| `Kolkata_Travel_Guide.pdf` | Kolkata city guide (food, shopping, places, transport) |
| `Darjeeling_pocket.pdf` | Darjeeling, Kurseong, Mirik, Kalimpong pocket guide |
| `Harnessing_the_Tourism_Potential_of_Bengal.pdf` | West Bengal Dept. of Tourism policy report |
| `Rajasthan-Through-Districts.pdf` | Rajasthan district-by-district guide |
| `UP-Book-opt.pdf` | Uttar Pradesh state tourism guide |

Place these five PDFs in `data/raw/` before running ingestion. Source-to-metadata mapping
(state, region, city, document type) is declared in `configs/domain_config.yaml`.

---

## 🏗️ Architecture

```
                ┌──────────────────┐
   PDFs  ──────▶│  ingest_all.py   │──▶ cleaned, section-aware chunks
                └──────────────────┘
                         │  embed (sentence-transformers)
                         ▼
                ┌──────────────────┐
                │    ChromaDB      │  persistent vector store
                └──────────────────┘
                         ▲
                         │ similarity search + metadata filter
                ┌──────────────────┐
   question ───▶│  retriever.py    │
                └──────────────────┘
                         │ top-k chunks
                         ▼
                ┌──────────────────┐
                │ prompt_builder.py│  grounding + citation + refusal rules
                └──────────────────┘
                         │ system + user prompt
                         ▼
                ┌──────────────────┐
                │  rag_engine.py   │──▶ Ollama /api/chat (llama3.2:1b)
                └──────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │     app.py       │  Streamlit UI, citations, filters
                └──────────────────┘
```

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally

```bash
# Install & start Ollama, then pull the model used by this project
ollama pull llama3.2:1b
ollama serve
```

### 2. Set up the project

```bash
git clone <this-repo>
cd tourism_rag_assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

python setup_project.py          # verifies structure, copies .env, checks deps/Ollama
pip install -r requirements.txt
```

Place the 5 raw PDFs in `data/raw/` (see table above) if they aren't already there.

### 3. Ingest the knowledge base

```bash
make ingest
# or: python -m src.ingestion.ingest_all
```

This extracts text, cleans it, chunks it section-aware, embeds it with
`all-MiniLM-L6-v2`, and upserts everything into a persistent ChromaDB collection at
`data/chroma_db/`.

Re-run ingestion any time you add/update PDFs — chunk IDs are content-hashed, so
re-ingestion safely upserts rather than duplicating.

```bash
# Ingest a single file only
python -m src.ingestion.ingest_all --file Kolkata_Travel_Guide.pdf

# Wipe and rebuild the whole collection
python -m src.ingestion.ingest_all --reset
```

### 4. Launch the app

```bash
make run
# or: streamlit run app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`).

### 5. Run the evaluation harness

```bash
make eval
# or: python -m src.services.rag_engine --eval data/evaluation/eval_questions.csv
```

Produces `data/evaluation/eval_questions_results.csv` with answers, sources, latency,
and whether an expected keyword was found in each answer.

---

## ⚙️ Configuration

| File | Purpose |
|---|---|
| `.env` (from `.env.example`) | Infra settings — Ollama URL/model, embedding model, Chroma paths, chunking/retrieval tuning |
| `configs/domain_config.yaml` | Domain knowledge — source-to-metadata mapping, chunking heading hints, answer policy (grounding/citation/refusal rules), out-of-scope safety boundaries |

Key environment variables (see `.env.example` for the full list):

```
OLLAMA_MODEL=llama3.2:1b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma_db
RETRIEVAL_TOP_K=6
RETRIEVAL_SCORE_THRESHOLD=0.25
```

---

## 🧪 Project Layout

```text
tourism_rag_assistant/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── Makefile
├── setup_project.py
├── configs/
│   └── domain_config.yaml
├── data/
│   ├── raw/                       # place the 5 PDFs here
│   ├── chroma_db/                 # generated by ingestion (gitignored)
│   └── evaluation/
│       └── eval_questions.csv
├── src/
│   ├── common/
│   │   ├── config_loader.py
│   │   └── logger.py
│   ├── ingestion/
│   │   ├── cleaner.py
│   │   └── ingest_all.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── generation/
│   │   └── prompt_builder.py
│   └── services/
│       └── rag_engine.py
└── app.py
```

---

## 🛡️ Answer Policy & Safety

The assistant is instructed, via `configs/domain_config.yaml` → `answer_policy` and
`safety`, to:

- Answer **only** from retrieved context — never from the LLM's general knowledge.
- Explicitly say "I don't know" (rather than guess) when context is insufficient.
- Cite every answer with source titles / pages / sections.
- Politely refuse and redirect for out-of-scope topics (visas, live prices/weather,
  bookings, unrelated general questions).

Edit `configs/domain_config.yaml` to add new states/sources, adjust tone, or expand the
refusal boundaries — no code changes required.

---

## 🧰 Makefile Commands

```bash
make setup      # run setup_project.py
make install    # pip install -r requirements.txt
make ingest     # run the ingestion pipeline
make run        # launch the Streamlit app
make eval       # run the evaluation harness
make reset-db   # delete the persisted Chroma vector store
make clean      # remove caches/logs
```

---

## 🩺 Troubleshooting

- **"Collection not found" / empty knowledge base** → run `make ingest` first.
- **"Could not reach Ollama"** → run `ollama serve` in a separate terminal and confirm
  `ollama pull llama3.2:1b` completed.
- **Slow first query** → the embedding model and LLM are loaded lazily on first use;
  subsequent queries are much faster.
- **Answers seem to ignore filters** → verify the sidebar filter values actually exist in
  your ingested metadata (state/region/city are case-sensitive exact matches sourced from
  `domain_config.yaml`).

---

## 📄 License

Internal / educational use. Source PDFs remain the property of their respective
publishers (West Bengal Tourism Dept., WBIDC, and state tourism guide authors).
