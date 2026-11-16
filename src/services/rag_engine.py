"""
rag_engine.py
-------------
The orchestrator that ties everything together:

    user question -> Retriever -> prompt_builder -> Ollama chat API -> answer

Also exposes a small CLI evaluation mode:

    python -m src.services.rag_engine --eval data/evaluation/eval_questions.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from src.common.config_loader import get_settings, get_domain_config
from src.common.logger import get_logger
from src.generation.prompt_builder import build_prompt
from src.retrieval.retriever import Retriever, RetrievalResult

logger = get_logger(__name__)

# Common conversational phrases that are NOT real tourism questions.
# Matched against the whole (lowercased, punctuation-stripped) message so
# short pleasantries don't get sent through retrieval + query rewriting.
_CHITCHAT_REPLIES = {
    "thank you": "You're welcome! Let me know if you have more questions about your trip.",
    "thanks": "You're welcome! Let me know if you have more questions about your trip.",
    "thank you so much": "You're very welcome! Happy to help with anything else about your trip.",
    "thanks a lot": "You're very welcome! Happy to help with anything else about your trip.",
    "thank you for your help": "You're welcome! Feel free to ask more about your trip anytime.",
    "ok": "Sounds good! Let me know if you'd like to know anything else.",
    "okay": "Sounds good! Let me know if you'd like to know anything else.",
    "ok thanks": "You're welcome!",
    "great": "Glad that helped! Anything else you'd like to know?",
    "great thanks": "You're welcome! Anything else you'd like to know?",
    "cool": "Glad that helped! Anything else you'd like to know?",
    "hi": "Hi! Ask me anything about tourism in India — places to visit, how to get there, food, and more.",
    "hello": "Hello! Ask me anything about tourism in India — places to visit, how to get there, food, and more.",
    "hey": "Hey! Ask me anything about tourism in India — places to visit, how to get there, food, and more.",
    "bye": "Safe travels! Come back anytime you have more questions.",
    "goodbye": "Safe travels! Come back anytime you have more questions.",
}

# Maps well-known cities/regions to their state, so state filtering also
# works for city-level questions (which never contain the state's own
# name, e.g. "how do I reach Kolkata by train?" never says "West Bengal").
# This is the key fix that prevents unrelated-state chunks (e.g. a
# Darjeeling passage) from leaking into a Kolkata-only question.
_CITY_TO_STATE = {
    "kolkata": "West Bengal",
    "darjeeling": "West Bengal",
    "kalimpong": "West Bengal",
    "kurseong": "West Bengal",
    "siliguri": "West Bengal",
    "digha": "West Bengal",
    "sundarbans": "West Bengal",
    "jaipur": "Rajasthan",
    "udaipur": "Rajasthan",
    "jodhpur": "Rajasthan",
    "jaisalmer": "Rajasthan",
    "pushkar": "Rajasthan",
    "ajmer": "Rajasthan",
    "mount abu": "Rajasthan",
    "lucknow": "Uttar Pradesh",
    "agra": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh",
    "mathura": "Uttar Pradesh",
    "ayodhya": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh",
    # extend this list as your knowledge base grows
}


def _chitchat_reply(question: str) -> Optional[str]:
    """Return a canned reply if the message is small talk, not a real query."""
    normalized = question.strip().lower().strip(".!? ")
    return _CHITCHAT_REPLIES.get(normalized)


def _detect_state_filter(query: str) -> Optional[str]:
    """
    If the query text names one (and only one) of the covered states/UTs
    -- or a well-known city/region belonging to one of them -- return that
    state name so retrieval can be restricted to it. This compensates for
    there being no state filter in the UI: with many similarly-worded
    state guides in the knowledge base, unfiltered search can otherwise
    pull in the wrong state's chunk for generic topics like 'local food'
    or 'how to reach'.
    """
    domain_cfg = get_domain_config()
    covered_states = domain_cfg.get("safety", {}).get("covered_states", [])
    query_lower = query.lower()

    matches = {s for s in covered_states if s.lower() in query_lower}

    # Fall back to city/region aliases when no state name is present in
    # the query itself (this is the common case for real user questions).
    for city, state in _CITY_TO_STATE.items():
        if city in query_lower and state in covered_states:
            matches.add(state)

    if len(matches) == 1:
        return next(iter(matches))
    return None


@dataclass
class RagResponse:
    question: str
    answer: str
    sources: List[str] = field(default_factory=list)
    retrieval: Optional[RetrievalResult] = None
    latency_seconds: float = 0.0
    context_used: bool = False
    error: Optional[str] = None
    search_query: str = ""  # the (possibly rewritten) query actually sent to retrieval


class OllamaClient:
    """Minimal HTTP client for the local Ollama `/api/chat` endpoint."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Is `ollama serve` running and is the model pulled "
                f"(`ollama pull {self.model}`)?"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama returned an error: {exc}") from exc

        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError(f"Ollama returned an empty response: {data}")
        return content.strip()

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False


class RagEngine:
    """High-level facade: ask a question, get a grounded answer."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.retriever = retriever or Retriever()
        self.ollama = ollama_client or OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    def contextualize_query(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        If there is prior conversation, rewrite a possibly vague follow-up
        question (e.g. "how can I get there?") into a standalone question
        that names the actual place/topic, using recent chat history.
        Falls back to the original question on any failure.

        IMPORTANT: the rewrite must NOT touch questions that already name
        their own place/topic. Local models are unreliable at following
        "leave it unchanged" instructions, and left unchecked this step
        can silently merge unrelated place names from earlier turns into
        an otherwise-clear question (e.g. turning "how do I reach Kolkata
        by train?" into a Darjeeling-tainted query right after a
        Darjeeling question), which then corrupts retrieval.
        """
        if not conversation_history:
            return question

        recent = conversation_history[-4:]
        history_text = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in recent
        )

        system_prompt = (
            "You rewrite a follow-up travel question into a standalone "
            "question, ONLY if it relies on pronouns or vague references "
            "like 'there', 'that place', 'it', or 'the city' that require "
            "the conversation history to resolve. If the question already "
            "names a specific place or topic, or is otherwise "
            "understandable on its own, output it COMPLETELY UNCHANGED — "
            "do not add, merge, substitute, or 'helpfully' bring in any "
            "place names from the conversation history. When in doubt, "
            "leave the question unchanged. Output ONLY the rewritten "
            "question, nothing else, no explanation."
        )
        user_prompt = (
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"FOLLOW-UP QUESTION:\n{question}\n\n"
            "Rewritten standalone question:"
        )

        try:
            rewritten = self.ollama.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            rewritten = rewritten.strip().strip('"')
            if rewritten:
                if rewritten.lower() != question.strip().lower():
                    logger.info("Rewrote query %r -> %r", question, rewritten)
                return rewritten
        except RuntimeError as exc:
            logger.warning("Query rewrite failed, using original question: %s", exc)

        return question

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        state: Optional[str] = None,
        region: Optional[str] = None,
        city: Optional[str] = None,
        doc_type: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> RagResponse:
        start = time.time()
        question = (question or "").strip()

        if not question:
            return RagResponse(
                question=question,
                answer="Please ask a question about West Bengal, Rajasthan, or Uttar Pradesh tourism.",
                error="empty_question",
            )

        chitchat = _chitchat_reply(question)
        if chitchat is not None:
            return RagResponse(
                question=question,
                answer=chitchat,
                context_used=False,
                latency_seconds=round(time.time() - start, 2),
            )

        search_query = self.contextualize_query(question, conversation_history)

        if state is None:
            # Detect against the rewritten query first (it's what actually
            # gets sent to the retriever); this also now recognizes city
            # names, not just exact state names.
            state = _detect_state_filter(search_query)

        try:
            retrieval = self.retriever.retrieve(
                query=search_query,
                top_k=top_k,
                state=state,
                region=region,
                city=city,
                doc_type=doc_type,
            )
        except RuntimeError as exc:
            logger.error("Retrieval failed: %s", exc)
            return RagResponse(
                question=question,
                answer=(
                    "The knowledge base isn't ready yet. Please run ingestion "
                    "(`make ingest`) before asking questions."
                ),
                error=str(exc),
                search_query=search_query,
                latency_seconds=time.time() - start,
            )

        built_prompt = build_prompt(
            question=search_query,
            retrieval_result=retrieval,
            conversation_history=conversation_history,
        )

        try:
            raw_answer = self.ollama.chat(
                system_prompt=built_prompt.system_prompt,
                user_prompt=built_prompt.user_prompt,
            )
        except RuntimeError as exc:
            logger.error("Generation failed: %s", exc)
            return RagResponse(
                question=question,
                answer=f"I couldn't reach the local language model. Details: {exc}",
                sources=built_prompt.source_labels,
                retrieval=retrieval,
                context_used=built_prompt.context_used,
                error=str(exc),
                search_query=search_query,
                latency_seconds=time.time() - start,
            )

        return RagResponse(
            question=question,
            answer=raw_answer,
            sources=built_prompt.source_labels,
            retrieval=retrieval,
            context_used=built_prompt.context_used,
            search_query=search_query,
            latency_seconds=round(time.time() - start, 2),
        )


# -----------------------------------------------------------------------------
# CLI evaluation mode
# -----------------------------------------------------------------------------
def run_eval(csv_path: str) -> None:
    import pandas as pd

    engine = RagEngine()

    if not engine.ollama.health_check():
        logger.warning(
            "Ollama does not appear to be reachable at %s. "
            "Generation calls below may fail.",
            engine.settings.ollama_base_url,
        )

    df = pd.read_csv(csv_path)
    required_cols = {"question"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Eval CSV must contain columns: {required_cols}")

    results = []
    for _, row in df.iterrows():
        question = str(row["question"])
        category = row.get("category", "uncategorized")
        expected_keyword = row.get("expected_keyword", "")

        response = engine.answer(question)
        contains_keyword = (
            bool(expected_keyword)
            and isinstance(expected_keyword, str)
            and expected_keyword.lower() in response.answer.lower()
        )

        print("=" * 80)
        print(f"[{category}] Q: {question}")
        if response.search_query and response.search_query != question:
            print(f"Rewritten query: {response.search_query}")
        print(f"A: {response.answer}")
        print(f"Sources: {response.sources}")
        print(f"Latency: {response.latency_seconds}s | Context used: {response.context_used}")
        if expected_keyword:
            print(f"Expected keyword '{expected_keyword}' found: {contains_keyword}")

        results.append(
            {
                "question": question,
                "category": category,
                "search_query": response.search_query,
                "answer": response.answer,
                "sources": "; ".join(response.sources),
                "latency_seconds": response.latency_seconds,
                "context_used": response.context_used,
                "expected_keyword": expected_keyword,
                "keyword_found": contains_keyword,
                "error": response.error or "",
            }
        )

    out_df = pd.DataFrame(results)
    out_path = str(csv_path).replace(".csv", "_results.csv")
    out_df.to_csv(out_path, index=False)
    print("=" * 80)
    print(f"Saved evaluation results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tourism RAG Engine CLI")
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Path to an evaluation CSV with a 'question' column.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        help="Ask a single question from the command line.",
    )
    args = parser.parse_args()

    if args.eval:
        run_eval(args.eval)
    elif args.ask:
        engine = RagEngine()
        response = engine.answer(args.ask)
        print(response.answer)
        print(f"\nSources: {response.sources}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()