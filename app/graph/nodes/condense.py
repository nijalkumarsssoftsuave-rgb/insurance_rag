"""Rewrites a multi-turn message into a standalone question.

Non-negotiable for multi-turn retrieval. "And what about dental?" embeds to
nothing useful on its own - it has no subject, no product, no policy. Retrieval
quality on follow-up turns collapses without this step (ARCHITECTURE 8, step 2).

Skipped on the first turn, where there is no history to fold in and the LLM call
would be pure latency.
"""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import ConversationState
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry

log = get_logger(__name__)

HISTORY_TURNS = 6  # three exchanges is enough to resolve a pronoun


async def condense_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = state["question"]
    history = state.get("messages") or []

    if not history:
        return {"standalone_question": question, "timings_ms": {"condense": 0}}

    transcript = _format_history(history[-HISTORY_TURNS:])
    if not transcript:
        return {"standalone_question": question, "timings_ms": {"condense": 0}}

    prompt = registry.load("condense")
    try:
        completion = await get_llm().complete(
            [
                system(prompt.body),
                user(f"Conversation so far:\n{transcript}\n\nLatest message: {question}"),
            ],
            temperature=prompt.temperature or 0.0,
            max_tokens=200,
        )
        standalone = completion.text.strip() or question
    except Exception as exc:
        # Degrade to the raw question. A worse retrieval query beats a failed turn.
        log.warning("Condensation failed, using raw question", error=str(exc))
        standalone = question

    if standalone != question:
        log.debug("Condensed question", original=question[:80], standalone=standalone[:80])

    return {
        "standalone_question": standalone,
        "timings_ms": {"condense": int((time.perf_counter() - started) * 1000)},
    }


def _format_history(messages: list) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"Customer: {message.content}")
        elif isinstance(message, AIMessage):
            # Truncate assistant turns: the model needs the topic, not the whole
            # previous answer, and full answers crowd the context.
            lines.append(f"Assistant: {str(message.content)[:300]}")
    return "\n".join(lines)
