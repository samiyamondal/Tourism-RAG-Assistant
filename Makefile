.PHONY: help setup install ingest run eval clean reset-db

help:
	@echo "Tourism RAG Assistant — available commands"
	@echo "  make setup      - run setup_project.py to scaffold/verify the project"
	@echo "  make install    - install python dependencies"
	@echo "  make ingest     - run the ingestion pipeline (PDF -> chunks -> ChromaDB)"
	@echo "  make run        - launch the Streamlit app"
	@echo "  make eval       - run the evaluation harness against eval_questions.csv"
	@echo "  make reset-db   - delete the persisted Chroma vector store"
	@echo "  make clean      - remove caches, logs, and __pycache__ directories"

setup:
	python setup_project.py

install:
	pip install -r requirements.txt

ingest:
	python -m src.ingestion.ingest_all

run:
	streamlit run app.py

eval:
	python -m src.services.rag_engine --eval data/evaluation/eval_questions.csv

reset-db:
	rm -rf data/chroma_db

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf logs/*.log
