"""Grounded answer generation under the citation contract.

The model receives fenced, labelled context and must return structured output:
an answer, the chunk ids it relied on, a confidence, and whether a human should
take over. Structured output rather than free text is what makes the verification
step in ``verify.py`` possible at all - you cannot check citations that were never
declared.
"""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.config import settings
from app.graph.state import ConversationState
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry

log = get_logger(__name__)


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The answer, citing clauses as [chunk_id]")
    citations: list[str] = Field(
        default_factory=list, description="chunk_id values actually relied upon"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    needs_human: bool = Field(default=False, description="True if the question needs a human agent")


DISCLAIMER = (
    "\n\n*This explains what your policy document says. It is not a claims "
    "decision - your insurer determines the outcome of any claim.*"
)


async def generate_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = state.get("standalone_question") or state["question"]
    context = state.get("context_text") or ""
    route = state.get("route") or {}

    prompt = registry.load("generate")
    llm = get_llm()

    instruction = prompt.body
    if route.get("is_coverage_question"):
        # Restate the exclusion requirement at the point of generation. The
        # system prompt says it once; a coverage question is exactly where the
        # model is most likely to answer "yes" from the benefits section alone.
        instruction += (
            "\n\nThis is a coverage question. Before stating that something is "
            "covered, check the exclusions and waiting periods in the context and "
            "state any that apply. If the context has no exclusions section, say "
            "that you could not verify exclusions."
        )

    messages = [
        system(instruction),
        user(f"{context}\n\nQuestion: {question}"),
    ]

    try:
        result = await llm.structured(
            messages, GroundedAnswer, temperature=prompt.temperature or 0.1
        )
    except Exception as exc:
        log.error("Generation failed", error=str(exc))
        return {
            "answer": (
                "I ran into a problem answering that. Please try again, or I can "
                "connect you to an agent."
            ),
            "citations": [],
            "abstained": True,
            "needs_human": True,
            "error": str(exc),
            "timings_ms": {"generate": int((time.perf_counter() - started) * 1000)},
        }

    # Restore any PII the guard masked. The user supplied it; echoing it back to
    # them is correct, and the masked form only ever existed for the hop through
    # logs and the model.
    answer = _unmask(result.answer, state.get("pii_mapping") or {})
    answer = _merge_claim_answer(state.get("claim_answer"), answer)

    return {
        "answer": answer,
        "citations": _resolve_citations(result.citations, state.get("context_blocks") or []),
        "confidence": result.confidence,
        "needs_human": result.needs_human,
        "abstained": False,
        "model": llm.model,
        "prompt_version": prompt.qualified,
        "timings_ms": {"generate": int((time.perf_counter() - started) * 1000)},
    }


def _merge_claim_answer(claim_answer: str | None, clause_answer: str) -> str:
    """Re-attach the claim status a rejection hand-off left behind.

    `clause_handoff_node` rewrites the question so Lane A can retrieve the clause
    behind a rejection, and stashes the templated status in `claim_answer`. Without
    this merge the generated clause explanation simply replaces it, so a customer
    who asked "what is the status of CLM-2026-0004" was told what clause 4.11 says
    and never told their claim was rejected, for how much, or what to do next.

    The status goes first because it is what was asked, and it is the half rendered
    deterministically from the database rather than by a model.
    """
    if not claim_answer:
        return clause_answer
    return f"{claim_answer}\n\n**Why this was decided**\n\n{clause_answer}"


def _unmask(text: str, mapping: dict[str, str]) -> str:
    for placeholder in sorted(mapping, key=len, reverse=True):
        text = text.replace(placeholder, mapping[placeholder])
    return text


def _resolve_citations(cited_ids: list[str], blocks: list[dict]) -> list[dict]:
    """Turn declared chunk ids into full citation records.

    Ids the model invented are dropped here rather than shown to the user;
    ``verify.py`` separately treats their presence as a groundedness failure.
    """
    by_id = {str(b.get("chunk_id")): b for b in blocks}
    return [by_id[cid] for cid in cited_ids if cid in by_id]


def append_disclaimer(answer: str) -> str:
    return answer + DISCLAIMER if settings.app.is_production else answer
