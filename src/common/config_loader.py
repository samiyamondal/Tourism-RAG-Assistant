"""
config_loader.py
-----------------
Centralized configuration loader for the Tourism RAG Assistant.

Combines two configuration layers:
  1. `.env` (or real environment variables) — runtime / infra settings
     (paths, model names, ports, thresholds).
  2. `configs/domain_config.yaml` — domain knowledge: source-to-metadata
     mapping, answer policy, safety rules, chunking/retrieval defaults.

Usage:
    from src.common.config_loader import get_settings, get_domain_config

    settings = get_settings()
    domain = get_domain_config()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Resolve project root as the parent of the `src/` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_DOMAIN_CONFIG_PATH = PROJECT_ROOT / "configs" / "domain_config.yaml"


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key, default)
    return value


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Runtime / infrastructure settings sourced from environment variables."""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_timeout_seconds: int = 120

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "tourism_knowledge_base"

    # Ingestion
    raw_data_dir: str = "./data/raw"
    chunk_size_tokens: int = 350
    chunk_overlap_tokens: int = 60

    # Retrieval
    retrieval_top_k: int = 6
    retrieval_score_threshold: float = 0.25

    # App
    app_log_level: str = "INFO"
    app_log_dir: str = "./logs"

    project_root: Path = field(default=PROJECT_ROOT)

    def resolved_path(self, relative_or_absolute: str) -> Path:
        """Resolve a config path relative to the project root, if not absolute."""
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


@lru_cache(maxsize=1)
def get_settings(env_path: Optional[str] = None) -> Settings:
    """
    Load environment variables (from `.env` if present) and return a cached
    `Settings` instance. Safe to call repeatedly across modules.
    """
    dotenv_path = Path(env_path) if env_path else DEFAULT_ENV_PATH
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)
    else:
        # Fall back silently to whatever is already in the environment.
        load_dotenv(override=False)

    return Settings(
        ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_env("OLLAMA_MODEL", "llama3.2:3b"),
        ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 120),
        embedding_model=_env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chroma_persist_dir=_env("CHROMA_PERSIST_DIR", "./data/chroma_db"),
        chroma_collection_name=_env(
            "CHROMA_COLLECTION_NAME", "tourism_knowledge_base"
        ),
        raw_data_dir=_env("RAW_DATA_DIR", "./data/raw"),
        chunk_size_tokens=_env_int("CHUNK_SIZE_TOKENS", 350),
        chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 60),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 6),
        retrieval_score_threshold=_env_float("RETRIEVAL_SCORE_THRESHOLD", 0.25),
        app_log_level=_env("APP_LOG_LEVEL", "INFO"),
        app_log_dir=_env("APP_LOG_DIR", "./logs"),
    )


@lru_cache(maxsize=1)
def get_domain_config(
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and cache the domain configuration YAML file
    (`configs/domain_config.yaml` by default).
    """
    path = Path(config_path) if config_path else DEFAULT_DOMAIN_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Domain config not found at {path}. "
            "Did you run `make setup` / `setup_project.py`?"
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_source_metadata_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Look up the configured metadata block for a raw source filename."""
    domain = get_domain_config()
    for source in domain.get("sources", []):
        if source.get("file") == filename:
            return source
    return None


def list_covered_states() -> List[str]:
    domain = get_domain_config()
    return domain.get("safety", {}).get("covered_states", [])


def clear_config_cache() -> None:
    """Clear cached settings/config — useful in tests or after editing files."""
    get_settings.cache_clear()
    get_domain_config.cache_clear()
