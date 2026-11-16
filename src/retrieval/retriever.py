"""
retriever.py
------------
Vector similarity retrieval over the ChromaDB collection populated by
`src/ingestion/ingest_all.py`. Supports optional metadata filtering by
state / region / city / doc_type, and returns structured, citation-ready
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from src.common.config_loader import get_settings
from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its similarity score and metadata."""

    text: str
    score: float  # cosine similarity, higher = more relevant (0..1 approx)
    doc_id: str
    title: str
    state: str
    region: str
    city: str
    doc_type: str
    source_file: str
    page_number: int
    section_heading: str
    chunk_index: int

    def citation_label(self) -> str:
        parts = [self.title]
        if self.page_number:
            parts.append(f"p.{self.page_number}")
        if self.section_heading:
            parts.append(f"§ {self.section_heading}")
        return " — ".join(parts)


@dataclass
class RetrievalResult:
    """The full result of a retrieval call, ready for prompt building."""

    query: str
    chunks: List[RetrievedChunk] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def unique_sources(self) -> List[str]:
        seen = []
        for c in self.chunks:
            label = c.citation_label()
            if label not in seen:
                seen.append(label)
        return seen


class Retriever:
    """
    Wraps a persistent ChromaDB collection + sentence-transformers embedder
    to perform semantic search with optional metadata filters.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        settings = get_settings()
        self.persist_dir = str(settings.resolved_path(persist_dir or settings.chroma_persist_dir))
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.default_top_k = settings.retrieval_top_k
        self.default_score_threshold = settings.retrieval_score_threshold

        self._client = None
        self._collection = None
        self._embedder = None

    # -- lazy resources ------------------------------------------------
    @property
    def collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            try:
                self._collection = self._client.get_collection(self.collection_name)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"Collection '{self.collection_name}' not found at {self.persist_dir}. "
                    "Did you run ingestion? (`make ingest`)"
                ) from exc
        return self._collection

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model for retrieval: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def collection_size(self) -> int:
        try:
            return self.collection.count()
        except Exception:  # noqa: BLE001
            return 0

    # -- core retrieval --------------------------------------------------
    def _build_where_filter(
        self,
        state: Optional[str] = None,
        region: Optional[str] = None,
        city: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        clauses = []
        if state and state.lower() != "all":
            clauses.append({"state": {"$eq": state}})
        if region and region.lower() != "all":
            clauses.append({"region": {"$eq": region}})
        if city and city.lower() != "all":
            clauses.append({"city": {"$eq": city}})
        if doc_type and doc_type.lower() != "all":
            clauses.append({"doc_type": {"$eq": doc_type}})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        state: Optional[str] = None,
        region: Optional[str] = None,
        city: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Run a semantic search against the vector store, with optional
        metadata filters. Results below `score_threshold` are dropped.
        """
        top_k = top_k or self.default_top_k
        score_threshold = (
            score_threshold if score_threshold is not None else self.default_score_threshold
        )

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        where = self._build_where_filter(state=state, region=region, city=city, doc_type=doc_type)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self.collection.query(**query_kwargs)

        result = RetrievalResult(query=query)

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        for doc_text, meta, distance in zip(documents, metadatas, distances):
            # Chroma with cosine space returns distance in [0, 2]; similarity ~= 1 - distance/2
            # but for normalized embeddings with cosine space, distance is typically 1 - cos_sim.
            similarity = max(0.0, 1.0 - distance)
            if similarity < score_threshold:
                continue

            result.chunks.append(
                RetrievedChunk(
                    text=doc_text,
                    score=round(similarity, 4),
                    doc_id=meta.get("doc_id", ""),
                    title=meta.get("title", ""),
                    state=meta.get("state", ""),
                    region=meta.get("region", ""),
                    city=meta.get("city", ""),
                    doc_type=meta.get("doc_type", ""),
                    source_file=meta.get("source_file", ""),
                    page_number=meta.get("page_number", 0),
                    section_heading=meta.get("section_heading", ""),
                    chunk_index=meta.get("chunk_index", 0),
                )
            )

        # Sort by score descending (Chroma already does this, but be explicit).
        result.chunks.sort(key=lambda c: c.score, reverse=True)
        return result

    def list_distinct_values(self, field_name: str) -> List[str]:
        """
        Return distinct non-empty values for a metadata field (e.g. 'state',
        'region', 'city') across the whole collection. Used to populate
        Streamlit sidebar filter dropdowns.
        """
        try:
            everything = self.collection.get(include=["metadatas"])
        except Exception:  # noqa: BLE001
            return []

        values = set()
        for meta in everything.get("metadatas", []):
            val = (meta or {}).get(field_name)
            if val:
                values.add(val)
        return sorted(values)
