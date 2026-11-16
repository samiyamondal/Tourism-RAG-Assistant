"""
prompt_builder.py
------------------
Builds the grounded system + user prompt sent to the local LLM (via Ollama),
enforcing the answer policy defined in `configs/domain_config.yaml`
(grounding rules, citation rules, refusal rules, tone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.config_loader import get_domain_config
from src.retrieval.retriever import RetrievalResult

MAX_CONTEXT_CHARS = 6000  # guardrail so we don't blow past a small local model's context window


@dataclass
class BuiltPrompt:
    system_prompt: str
    user_prompt: str
    context_used: bool
    source_labels: List[str]


def _format_context_block(result: RetrievalResult) -> str:
    """Render retrieved chunks into a numbered, citation-tagged context block."""
    if result.is_empty:
        return "(No relevant context was found in the knowledge base for this question.)"

    lines = []
    running_len = 0
    for i, chunk in enumerate(result.chunks, start=1):
        label = chunk.citation_label()
        entry = f"[{i}] Source: {label}\n{chunk.text.strip()}\n"
        if running_len + len(entry) > MAX_CONTEXT_CHARS and lines:
            break
        lines.append(entry)
        running_len += len(entry)

    return "\n".join(lines)


def _build_system_prompt(domain_cfg: Dict[str, Any]) -> str:
    policy = domain_cfg.get("answer_policy", {})
    safety = domain_cfg.get("safety", {})

    persona = policy.get("persona", "You are a helpful travel assistant.")
    grounding_rules = policy.get("grounding_rules", [])
    citation_rules = policy.get("citation_rules", [])
    refusal_rules = policy.get("refusal_rules", [])
    tone = policy.get("tone", [])

    covered_states = safety.get("covered_states", [])
    out_of_scope_topics = safety.get("out_of_scope_topics", [])
    out_of_scope_response = safety.get(
        "out_of_scope_response",
        "I can only help with tourism topics for the covered states.",
    )

    def bullet_block(title: str, items: List[str]) -> str:
        if not items:
            return ""
        body = "\n".join(f"- {item}" for item in items)
        return f"{title}:\n{body}\n"

    system_prompt = f"""{persona}

Your knowledge is strictly limited to the CONTEXT block provided in each
user message. That context comes from official tourism guidebooks and
government reports covering: {", ".join(covered_states) if covered_states else "the covered regions"}.

{bullet_block("GROUNDING RULES (must follow)", grounding_rules)}
{bullet_block("CITATION RULES", citation_rules)}
{bullet_block("REFUSAL RULES", refusal_rules)}
{bullet_block("TONE", tone)}

OUT-OF-SCOPE TOPICS you must decline (politely, briefly, and redirect to
travel planning): {", ".join(out_of_scope_topics) if out_of_scope_topics else "topics unrelated to travel"}.
If a question is out of scope, respond with something close to:
"{out_of_scope_response}"

ANSWER FORMAT:
1. Answer ONLY the specific question asked — nothing more. Do not volunteer extra
   facts, nearby attractions, historical background, or "related info" unless the
   user explicitly asked for it.
2. Be as short as possible while fully answering. A one-fact question (fee, distance,
   timing, yes/no) gets a one-sentence answer. Do not pad it into a paragraph.
3. If the CONTEXT does not contain the answer, say so in one sentence — do not invent details.
4. Do NOT include a "Sources:" line, citation numbers like [1], or any source
   list in your answer text — sources are displayed separately by the app.
   Just answer the question in plain prose.
""".strip()

    return system_prompt


def build_prompt(
    question: str,
    retrieval_result: RetrievalResult,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> BuiltPrompt:
    """
    Build the final (system_prompt, user_prompt) pair for the LLM call.

    `conversation_history` is optional list of {"role": "user"/"assistant", "content": str}
    for lightweight multi-turn context; it is summarized, not replayed verbatim,
    to keep the prompt small for a 1B local model.
    """
    domain_cfg = get_domain_config()
    system_prompt = _build_system_prompt(domain_cfg)
    context_block = _format_context_block(retrieval_result)

    history_block = ""
    if conversation_history:
        # Keep only the last 2 turns to control prompt size for a small model.
        recent = conversation_history[-4:]
        formatted = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in recent
        )
        history_block = f"\nRECENT CONVERSATION (for context only, do not repeat verbatim):\n{formatted}\n"

    user_prompt = f"""CONTEXT:
{context_block}
{history_block}
QUESTION:
{question}

Answer the QUESTION using only the CONTEXT above, following the grounding,
citation, and refusal rules from your instructions.""".strip()

    return BuiltPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_used=not retrieval_result.is_empty,
        source_labels=retrieval_result.unique_sources(),
    )