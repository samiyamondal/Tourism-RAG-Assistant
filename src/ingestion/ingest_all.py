"""
ingest_all.py
-------------
End-to-end ingestion pipeline for the Tourism RAG Assistant.

Steps:
  1. Discover raw PDFs in `RAW_DATA_DIR` (matched against `configs/domain_config.yaml`).
  2. Extract text per page via `pdfplumber` (fallback to `pypdf` on failure).
  3. Clean text (strip running headers/footers, page numbers, whitespace).
  4. Chunk text with a section-aware sliding window.
  5. Embed chunks with `sentence-transformers`.
  6. Upsert chunks + embeddings + metadata into a persistent ChromaDB collection.

Run:
    python -m src.ingestion.ingest_all
    python -m src.ingestion.ingest_all --file Kolkata_Travel_Guide.pdf --reset
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.common.config_loader import get_domain_config, get_settings
from src.common.logger import get_logger
from src.ingestion.cleaner import clean_text, remove_repeated_running_headers

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# PDF text extraction
# -----------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> List[str]:
    """
    Extract raw text for each page of a PDF. Tries `pdfplumber` first (better
    layout handling for tables/columns); falls back to `pypdf` if it fails
    or returns nothing.
    """
    pages: List[str] = []

    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        if any(p.strip() for p in pages):
            return pages
    except Exception as exc:  # noqa: BLE001
        logger.warning("pdfplumber failed for %s (%s); falling back to pypdf.", pdf_path.name, exc)

    # Fallback: pypdf
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001
        logger.error("pypdf also failed for %s: %s", pdf_path.name, exc)
        pages = []

    return pages


# -----------------------------------------------------------------------------
# Section-aware chunking
# -----------------------------------------------------------------------------
def _approx_token_count(text: str) -> int:
    """Cheap token estimate (~4 chars/token) to avoid a hard tiktoken dependency at chunk time."""
    return max(1, len(text) // 4)


def _split_into_sections(page_text: str, heading_hints: List[str]) -> List[Tuple[Optional[str], str]]:
    """
    Split a page's text into (heading, body) tuples using configured heading
    hints as anchors. Text before the first recognized heading is returned
    with heading=None.
    """
    if not heading_hints:
        return [(None, page_text)]

    # Build a regex that matches any hint at the start of a line (case-insensitive).
    escaped = [re.escape(h) for h in heading_hints]
    pattern = re.compile(rf"(?m)^({'|'.join(escaped)})", re.IGNORECASE)

    matches = list(pattern.finditer(page_text))
    if not matches:
        return [(None, page_text)]

    sections: List[Tuple[Optional[str], str]] = []
    if matches[0].start() > 0:
        sections.append((None, page_text[: matches[0].start()]))

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page_text)
        heading = match.group(1).strip()
        body = page_text[start:end]
        sections.append((heading, body))

    return sections


def chunk_document(
    doc_id: str,
    pages: List[str],
    target_tokens: int,
    overlap_tokens: int,
    heading_hints: List[str],
) -> List[Dict[str, Any]]:
    """
    Produce section-aware, overlapping chunks from a list of cleaned page
    texts. Each chunk dict contains: text, page_number, section_heading,
    chunk_index.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for page_num, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue

        sections = _split_into_sections(page_text, heading_hints)

        for heading, body in sections:
            body = body.strip()
            if not body:
                continue

            words = body.split()
            if not words:
                continue

            # Sliding window over words, sized by approximate token budget.
            # ~0.75 words per token on average for English prose -> convert.
            words_per_chunk = max(20, int(target_tokens * 0.75))
            words_overlap = max(0, int(overlap_tokens * 0.75))

            start = 0
            while start < len(words):
                end = min(start + words_per_chunk, len(words))
                chunk_text = " ".join(words[start:end])

                if _approx_token_count(chunk_text) >= 15:  # skip tiny fragments
                    chunks.append(
                        {
                            "text": chunk_text,
                            "page_number": page_num,
                            "section_heading": heading,
                            "chunk_index": chunk_index,
                        }
                    )
                    chunk_index += 1

                if end == len(words):
                    break
                start = end - words_overlap

    return chunks


# -----------------------------------------------------------------------------
# Embedding + persistence
# -----------------------------------------------------------------------------
class Embedder:
    """Thin wrapper around a sentence-transformers model, loaded lazily."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()


def get_chroma_collection(persist_dir: str, collection_name: str):
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic, content-aware ID so re-ingestion upserts cleanly."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}__{chunk_index:05d}__{digest}"


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def ingest_single_pdf(
    pdf_path: Path,
    source_meta: Dict[str, Any],
    embedder: Embedder,
    collection,
    chunking_cfg: Dict[str, Any],
) -> int:
    logger.info("Ingesting %s (%s)", pdf_path.name, source_meta.get("title"))

    raw_pages = extract_pages(pdf_path)
    if not raw_pages:
        logger.warning("No extractable text found in %s — skipping.", pdf_path.name)
        return 0

    # Strip repeated running headers/footers across the whole document first,
    # then clean each page individually.
    deduped_pages = remove_repeated_running_headers(raw_pages)
    cleaned_pages = [clean_text(p) for p in deduped_pages]

    chunks = chunk_document(
        doc_id=source_meta["doc_id"],
        pages=cleaned_pages,
        target_tokens=chunking_cfg.get("target_tokens", 350),
        overlap_tokens=chunking_cfg.get("overlap_tokens", 60),
        heading_hints=chunking_cfg.get("section_heading_hints", []),
    )

    if not chunks:
        logger.warning("No chunks produced for %s — skipping.", pdf_path.name)
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed(texts)

    ids = [make_chunk_id(source_meta["doc_id"], c["chunk_index"], c["text"]) for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append(
            {
                "doc_id": source_meta["doc_id"],
                "title": source_meta.get("title", ""),
                "state": source_meta.get("state") or "",
                "region": source_meta.get("region") or "",
                "city": source_meta.get("city") or "",
                "doc_type": source_meta.get("doc_type", ""),
                "source_file": source_meta.get("file", pdf_path.name),
                "page_number": c["page_number"],
                "section_heading": c["section_heading"] or "",
                "chunk_index": c["chunk_index"],
            }
        )

    # Chroma upserts in batches to avoid oversized single calls.
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=texts[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )

    logger.info("  -> %d chunks upserted for %s", len(ids), pdf_path.name)
    return len(ids)


def run_ingestion(
    only_file: Optional[str] = None,
    reset_collection: bool = False,
) -> None:
    settings = get_settings()
    domain = get_domain_config()

    raw_dir = settings.resolved_path(settings.raw_data_dir)
    persist_dir = str(settings.resolved_path(settings.chroma_persist_dir))

    if not raw_dir.exists():
        logger.error("Raw data directory does not exist: %s", raw_dir)
        sys.exit(1)

    client, collection = get_chroma_collection(persist_dir, settings.chroma_collection_name)

    if reset_collection:
        logger.info("Resetting collection '%s'.", settings.chroma_collection_name)
        client.delete_collection(settings.chroma_collection_name)
        collection = client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    embedder = Embedder(settings.embedding_model)
    chunking_cfg = domain.get("chunking", {})

    sources = domain.get("sources", [])
    if only_file:
        sources = [s for s in sources if s.get("file") == only_file]
        if not sources:
            logger.error("No configured source matches --file '%s'.", only_file)
            sys.exit(1)

    total_chunks = 0
    for source_meta in sources:
        pdf_path = raw_dir / source_meta["file"]
        if not pdf_path.exists():
            logger.warning("Configured source not found on disk, skipping: %s", pdf_path)
            continue
        total_chunks += ingest_single_pdf(
            pdf_path=pdf_path,
            source_meta=source_meta,
            embedder=embedder,
            collection=collection,
            chunking_cfg=chunking_cfg,
        )

    logger.info(
        "Ingestion complete. %d total chunks in collection '%s' at %s",
        collection.count(),
        settings.chroma_collection_name,
        persist_dir,
    )
    logger.info("This run added/updated %d chunks.", total_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest tourism PDFs into ChromaDB.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Only ingest a single configured source filename (e.g. Kolkata_Travel_Guide.pdf).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting.",
    )
    args = parser.parse_args()
    run_ingestion(only_file=args.file, reset_collection=args.reset)


if __name__ == "__main__":
    main()
