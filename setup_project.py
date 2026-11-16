#!/usr/bin/env python3
"""
setup_project.py
-----------------
One-shot setup/verification script for the Tourism RAG Assistant.

What it does:
  1. Ensures the expected directory structure exists.
  2. Copies `.env.example` to `.env` if `.env` doesn't already exist.
  3. Verifies that the 5 raw PDFs referenced in `configs/domain_config.yaml`
     are present in `data/raw/` (warns if missing, doesn't fail hard).
  4. Checks that required Python packages are importable and prints guidance
     if not.
  5. Checks whether Ollama is reachable and whether the configured model is
     pulled.

Run:
    python setup_project.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/evaluation",
    "data/chroma_db",
    "logs",
    "src/common",
    "src/ingestion",
    "src/retrieval",
    "src/generation",
    "src/services",
]

REQUIRED_PACKAGES = [
    "yaml",
    "dotenv",
    "pandas",
    "pypdf",
    "pdfplumber",
    "sentence_transformers",
    "chromadb",
    "requests",
    "streamlit",
]


def _print_header(text: str) -> None:
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def ensure_directories() -> None:
    _print_header("1. Ensuring directory structure")
    for rel_dir in REQUIRED_DIRS:
        path = PROJECT_ROOT / rel_dir
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✔ {rel_dir}")


def ensure_env_file() -> None:
    _print_header("2. Checking .env file")
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"

    if env_path.exists():
        print("  ✔ .env already exists — leaving it untouched.")
        return

    if example_path.exists():
        shutil.copy(example_path, env_path)
        print("  ✔ Created .env from .env.example. Edit it to customize settings.")
    else:
        print("  ⚠ .env.example not found — skipping .env creation.")


def check_raw_pdfs() -> None:
    _print_header("3. Checking raw source PDFs")
    try:
        import yaml
    except ImportError:
        print("  ⚠ PyYAML not installed yet — skipping PDF presence check.")
        return

    config_path = PROJECT_ROOT / "configs" / "domain_config.yaml"
    if not config_path.exists():
        print("  ⚠ configs/domain_config.yaml not found — skipping check.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        domain_cfg = yaml.safe_load(f) or {}

    raw_dir = PROJECT_ROOT / "data" / "raw"
    sources = domain_cfg.get("sources", [])
    missing = []
    for source in sources:
        filename = source.get("file")
        if not filename:
            continue
        if (raw_dir / filename).exists():
            print(f"  ✔ {filename}")
        else:
            print(f"  ✘ MISSING: {filename}")
            missing.append(filename)

    if missing:
        print(
            f"\n  ⚠ {len(missing)} source file(s) missing from data/raw/. "
            "Place them there before running ingestion."
        )
    else:
        print("\n  All configured source PDFs are present.")


def check_python_packages() -> None:
    _print_header("4. Checking Python dependencies")
    missing = []
    for module_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
            print(f"  ✔ {module_name}")
        except ImportError:
            print(f"  ✘ MISSING: {module_name}")
            missing.append(module_name)

    if missing:
        print(
            "\n  ⚠ Some packages are missing. Install them with:\n"
            "      pip install -r requirements.txt"
        )
    else:
        print("\n  All required packages are importable.")


def check_ollama() -> None:
    _print_header("5. Checking Ollama availability")
    try:
        import requests
    except ImportError:
        print("  ⚠ `requests` not installed — skipping Ollama check.")
        return

    base_url = "http://localhost:11434"
    try:
        import os

        base_url = os.getenv("OLLAMA_BASE_URL", base_url)
    except Exception:  # noqa: BLE001
        pass

    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            print(f"  ✔ Ollama is reachable at {base_url}")
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            if models:
                print(f"    Installed models: {', '.join(models)}")
            else:
                print("    ⚠ No models pulled yet. Run: ollama pull llama3.2:1b")
        else:
            print(f"  ⚠ Ollama responded with status {resp.status_code}")
    except requests.exceptions.RequestException:
        print(
            f"  ⚠ Could not reach Ollama at {base_url}.\n"
            "    Start it with `ollama serve` and pull the model with "
            "`ollama pull llama3.2:1b`."
        )


def print_next_steps() -> None:
    _print_header("Next steps")
    print(
        "  1. (If needed) create a virtual environment and run:\n"
        "       pip install -r requirements.txt\n"
        "  2. Place the 5 source PDFs in data/raw/ (if not already there).\n"
        "  3. Run ingestion:\n"
        "       make ingest\n"
        "     or: python -m src.ingestion.ingest_all\n"
        "  4. Make sure Ollama is running with the model pulled:\n"
        "       ollama serve\n"
        "       ollama pull llama3.2:1b\n"
        "  5. Launch the app:\n"
        "       make run\n"
        "     or: streamlit run app.py\n"
    )


def main() -> None:
    ensure_directories()
    ensure_env_file()
    check_raw_pdfs()
    check_python_packages()
    check_ollama()
    print_next_steps()


if __name__ == "__main__":
    main()
