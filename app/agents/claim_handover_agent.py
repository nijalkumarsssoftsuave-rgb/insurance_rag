"""Hand-built ReAct loop for the claim handover task (Week 7 · Module 4 · Track D).

think -> act -> observe -> repeat, until the model says `finish` or a budget
trips. Deliberately not a framework: this is the whole loop, so nothing about it
is magic. Compare against ``app/claims/handover.py``, which does the identical
task as a fixed sequence - ``eval/week7/race.py`` runs both against the same
claims and measures which one you'd actually ship.

Uses ``get_llm().complete()`` rather than ``structured()`` for the per-step
decision. ``structured()`` (see ``app/llm/openai_provider.py``) discards
``response.usage`` entirely, and token cost is exactly what this week measures -
so the loop prompts for JSON and parses it defensively instead, the same way a
provider without native structured output already has to.

Memory here is short-term only: the running transcript of past actions and
observations, replayed into every prompt so the model does not re-ask for
something it already fetched. There is no cross-session memory (mem0, a vector
store) - out of scope for a single investigate-and-report task.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.tools import TOOL_SPECS, ToolResult, run_tool
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry
from app.security.authz import AuthSubject

log = get_logger(__name__)

# Stop conditions - the loop must not be able to run forever or unboundedly.
MAX_STEPS = 6
MAX_SECONDS = 20.0
MAX_TOTAL_TOKENS = 4000


@dataclass(slots=True)
class StepLog:
    thought: str
    action: str
    action_input: dict
    observation: str
    ok: bool


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: list[StepLog] = field(default_factory=list)
    stop_reason: str = "finished"
    elapsed_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0

    @property
    def tool_calls(self) -> int:
        return len(self.steps)


def _parse_step(raw: str) -> dict:
    """Recover the JSON object from a reply that may still wrap it in a fence."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {raw[:200]!r}")
    return json.loads(text[start : end + 1])


async def run(session: AsyncSession, subject: AuthSubject, claim_number: str) -> AgentResult:
    started = time.perf_counter()
    prompt = registry.load("claim_handover_agent")
    tool_specs = "\n".join(f"- `{name}`: {desc}" for name, desc in TOOL_SPECS.items())
    system_msg = system(prompt.render(tool_specs=tool_specs))
    transcript = [f"Investigate claim {claim_number}."]

    result = AgentResult(answer="")
    seen_actions: set[str] = set()
    # Short-term memory: facts observed from tool results, carried across steps.
    # `product_name` / `date_of_loss` are filled in here rather than trusted from
    # the model's `action_input`, and `search_policy_clause` refuses outright
    # until `date_of_loss` is in memory rather than running unfiltered -
    # date_of_loss picks which wording version of a clause is in force
    # (ARCHITECTURE 5.3), so it gets the same "never from the model, and never
    # silently absent" treatment `subject` gets everywhere else in app/claims.
    memory: dict[str, object] = {}

    for _step_num in range(MAX_STEPS):
        if time.perf_counter() - started > MAX_SECONDS:
            result.stop_reason = "time_budget_exceeded"
            break
        if result.input_tokens + result.output_tokens > MAX_TOTAL_TOKENS:
            result.stop_reason = "token_budget_exceeded"
            break

        completion = await get_llm().complete(
            [system_msg, user("\n".join(transcript))], temperature=0.0
        )
        result.llm_calls += 1
        result.input_tokens += completion.input_tokens
        result.output_tokens += completion.output_tokens

        try:
            step = _parse_step(completion.text)
        except (ValueError, json.JSONDecodeError) as exc:
            # Not appended to `steps`: no tool ran, so it is not a tool call -
            # `stop_reason` alone carries this failure.
            log.warning("Agent step unparseable", claim_number=claim_number, error=str(exc))
            result.stop_reason = "malformed_output"
            break

        thought = str(step.get("thought", ""))
        action = str(step.get("action", ""))

        if action == "finish":
            result.answer = str(step.get("final_answer", "")).strip()
            result.stop_reason = "finished"
            break

        action_input = dict(step.get("action_input") or {})
        if action == "search_policy_clause" and "date_of_loss" not in memory:
            # Refuse rather than degrade to an unfiltered lookup: date_of_loss
            # picks which wording version is in force (the bug fixed in
            # retrieve_by_clause), so this tool never runs without it, the same
            # way `require_policy_holder` raises rather than running an
            # unscoped query. The model gets a chance to correct itself, not a
            # silent wrong answer.
            tool_result = ToolResult(
                ok=False,
                observation="Look up the claim with get_claim_status first - "
                "search_policy_clause needs its date of loss.",
            )
        else:
            if action == "search_policy_clause":
                # Overwrite, never trust: the model never actually knew these.
                action_input["product_name"] = memory.get("product_name")
                action_input["date_of_loss"] = memory.get("date_of_loss")

            fingerprint = f"{action}:{json.dumps(action_input, sort_keys=True, default=str)}"
            if fingerprint in seen_actions:
                # The model asked the same question twice - looping, not
                # progressing.
                result.stop_reason = "repeated_action"
                break
            seen_actions.add(fingerprint)

            tool_result = await run_tool(action, action_input, session=session, subject=subject)
            if action == "get_claim_status" and tool_result.ok and tool_result.data:
                claim = tool_result.data["claim"]
                memory["product_name"] = claim.product_name
                memory["date_of_loss"] = claim.date_of_loss

        result.steps.append(
            StepLog(thought, action, action_input, tool_result.observation, tool_result.ok)
        )
        transcript.append(f"Action: {action}({action_input})")
        transcript.append(f"Observation: {tool_result.observation}")
    else:
        result.stop_reason = "max_steps_exceeded"

    if not result.answer:
        # A budget tripped before `finish` - answer from what was actually
        # observed rather than returning nothing.
        observations = "; ".join(s.observation for s in result.steps if s.ok)
        result.answer = (
            f"Investigation stopped early ({result.stop_reason}). "
            f"What was found: {observations or 'nothing yet'}."
        )

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result
