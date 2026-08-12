"""Query expansion: paraphrase, HyDE, step-back, decomposition.

Customers and policy documents do not share a vocabulary. Someone asks about
"teeth cleaning"; the wording says "dental prophylaxis". Dense retrieval bridges
some of that gap, expansion bridges more.

Pick per intent rather than running everything: each variant is an extra local
bge-m3 forward pass and an extra prefetch branch, so indiscriminate expansion
spends latency without buying recall (ARCHITECTURE 8).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field

from app.config import settings
from app.llm import LLMError, get_llm, system, user
from app.logging import get_logger

log = get_logger(__name__)


class ExpansionMode(StrEnum):
    PARAPHRASE = "paraphrase"
    HYDE = "hyde"
    STEP_BACK = "step_back"
    DECOMPOSE = "decompose"


@dataclass(slots=True)
class ExpandedQuery:
    """The original question plus its variants.

    ``original`` stays first and is never dropped - it is the only variant
    guaranteed to reflect what was actually asked.
    """

    original: str
    paraphrases: list[str] = field(default_factory=list)
    hypothetical: str | None = None
    step_back: str | None = None
    sub_questions: list[str] = field(default_factory=list)

    @property
    def all_variants(self) -> list[str]:
        variants = [self.original, *self.paraphrases]
        for extra in (self.hypothetical, self.step_back):
            if extra:
                variants.append(extra)
        variants.extend(self.sub_questions)
        # Dedupe case-insensitively while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for v in variants:
            key = v.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(v.strip())
        return out


class _Paraphrases(BaseModel):
    queries: list[str] = Field(description="Alternative phrasings using policy terminology")


class _SubQuestions(BaseModel):
    questions: list[str] = Field(description="Independent sub-questions")


_PARAPHRASE_PROMPT = (
    "You rewrite customer questions into the vocabulary used in insurance policy "
    "wordings, so they retrieve better against clause text.\n\n"
    "Rules:\n"
    "- Keep the original meaning exactly. Do not broaden or narrow it.\n"
    "- Prefer policy terminology (e.g. 'dental prophylaxis' for 'teeth cleaning', "
    "'pre-existing disease' for 'illness I already had').\n"
    "- Keep every identifier, product name and date unchanged.\n"
    "- Return only the rewritten questions."
)

_HYDE_PROMPT = (
    "Write a short passage as it would appear in an insurance policy document, "
    "answering the question below. Two to four sentences. Use formal clause "
    "language. It does not need to be factually correct for any real policy - it "
    "is used only as a retrieval probe, never shown to anyone."
)

_STEP_BACK_PROMPT = (
    "Given a specific question, write the single broader question whose answer "
    "would provide the context needed. Return only that question."
)

_DECOMPOSE_PROMPT = (
    "Split the question into independent sub-questions, each answerable on its "
    "own. If it is already a single question, return it unchanged."
)


async def expand(
    query: str,
    *,
    modes: list[ExpansionMode] | None = None,
    n_paraphrases: int | None = None,
) -> ExpandedQuery:
    """Run the requested expansions concurrently.

    Failures degrade rather than propagate: retrieval with the original query
    alone is worse than retrieval with variants, but far better than an error.
    """
    result = ExpandedQuery(original=query)
    if not settings.retrieval.query_expansion_enabled:
        return result

    modes = modes or default_modes(query)
    n = n_paraphrases if n_paraphrases is not None else settings.retrieval.query_expansion_variants

    tasks: dict[ExpansionMode, asyncio.Task] = {}
    async with asyncio.TaskGroup() as tg:
        if ExpansionMode.PARAPHRASE in modes and n > 0:
            tasks[ExpansionMode.PARAPHRASE] = tg.create_task(_paraphrase(query, n))
        if ExpansionMode.HYDE in modes and settings.retrieval.hyde_enabled:
            tasks[ExpansionMode.HYDE] = tg.create_task(_hyde(query))
        if ExpansionMode.STEP_BACK in modes:
            tasks[ExpansionMode.STEP_BACK] = tg.create_task(_step_back(query))
        if ExpansionMode.DECOMPOSE in modes:
            tasks[ExpansionMode.DECOMPOSE] = tg.create_task(_decompose(query))

    if task := tasks.get(ExpansionMode.PARAPHRASE):
        result.paraphrases = task.result()
    if task := tasks.get(ExpansionMode.HYDE):
        result.hypothetical = task.result()
    if task := tasks.get(ExpansionMode.STEP_BACK):
        result.step_back = task.result()
    if task := tasks.get(ExpansionMode.DECOMPOSE):
        result.sub_questions = task.result()

    return result


def default_modes(query: str) -> list[ExpansionMode]:
    """Cheap heuristics, not a model call.

    Deciding how to expand should not itself cost an LLM round trip.
    """
    lowered = query.lower()
    modes = [ExpansionMode.PARAPHRASE, ExpansionMode.HYDE]

    # Broad questions benefit from a more general framing.
    if any(p in lowered for p in ("what does", "what all", "everything", "overview", "summary")):
        modes.append(ExpansionMode.STEP_BACK)

    # Compound questions retrieve badly as one query.
    if " and " in lowered and "?" in query:
        modes.append(ExpansionMode.DECOMPOSE)

    return modes


async def _paraphrase(query: str, n: int) -> list[str]:
    try:
        out = await get_llm().structured(
            [system(_PARAPHRASE_PROMPT), user(f"Produce {n} rewritings of: {query}")],
            _Paraphrases,
        )
        return [q.strip() for q in out.queries[:n] if q.strip()]
    except (LLMError, Exception) as exc:
        log.warning("Paraphrase expansion failed", error=str(exc))
        return []


async def _hyde(query: str) -> str | None:
    try:
        completion = await get_llm().complete(
            [system(_HYDE_PROMPT), user(query)], temperature=0.0, max_tokens=200
        )
        return completion.text.strip() or None
    except Exception as exc:
        log.warning("HyDE expansion failed", error=str(exc))
        return None


async def _step_back(query: str) -> str | None:
    try:
        completion = await get_llm().complete(
            [system(_STEP_BACK_PROMPT), user(query)], temperature=0.0, max_tokens=80
        )
        text = completion.text.strip()
        return text if text and text.lower() != query.lower() else None
    except Exception as exc:
        log.warning("Step-back expansion failed", error=str(exc))
        return None


async def _decompose(query: str) -> list[str]:
    try:
        out = await get_llm().structured([system(_DECOMPOSE_PROMPT), user(query)], _SubQuestions)
        subs = [q.strip() for q in out.questions if q.strip()]
        # A single sub-question means it did not decompose - drop it as a duplicate.
        return subs if len(subs) > 1 else []
    except Exception as exc:
        log.warning("Decomposition failed", error=str(exc))
        return []
