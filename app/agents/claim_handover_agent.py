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
import re
import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.tools import TOOL_SPECS, ToolResult, run_tool
from app.core.enums import ClaimStatus
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry
from app.security import injection
from app.security.authz import AuthSubject

log = get_logger(__name__)

# Stop conditions - the loop must not be able to run forever or unboundedly.
MAX_STEPS = 6
MAX_SECONDS = 20.0
MAX_TOTAL_TOKENS = 4000

# Tools whose observation is free text from a source the agent does not
# control (adjuster notes, policy clause wording) rather than rendered from
# typed DB fields. `get_claim_status`'s observation is built from a status
# enum, dates and a closed set of reason codes - there is nothing there for
# an attacker to write. These two are where Week 8's injection attack landed
# (a fabricated "corrected to APPROVED" note via `get_claim_notes`, 10/10
# trials), so they get scanned for telemetry and fenced before re-entering
# the transcript; `get_claim_status` does not need either.
_UNTRUSTED_CONTENT_TOOLS = frozenset({"get_claim_notes", "search_policy_clause"})

# Output validation, the layer that turned out to matter. Week 8's injection
# attack (a fabricated "Claims Supervisor correction" note) beat the fence and
# the prompt's explicit warning 10/10 times - the model is not structurally
# unable to be talked into this, only discouraged, and discouragement did not
# hold. This regex is the deterministic backstop: it does not ask the model to
# resist anything, it checks the model's own claim against the status
# `get_claim_status` actually observed, the same way `date_of_loss` was never
# trusted from the model either. Deliberately narrow (rejected -> approved
# only, not a general fact-checker) - see RESULTS.md for what this does not
# cover.
APPROVAL_STATUS_RE = re.compile(
    r"\b(now\s+approved|corrected\s+to\s+approved|reclassif\w*\s+.{0,20}\bapproved\b|"
    r"is\s+approved|has\s+been\s+approved|approved\s+in\s+full|"
    r"approved\s+for\s+the\s+full\s+amount)\b",
    re.I,
)


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
    # What the model actually wrote, before any output-validation override.
    # Equal to `answer` unless `output_overridden` is True - kept so a caller
    # can measure what the model itself did versus what actually shipped,
    # rather than only the post-override number (which would be tautological:
    # an override that neutralizes X by construction cannot report X).
    raw_answer: str = ""
    output_overridden: bool = False
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

    def _transcript_observation(action_name: str, observation: str) -> str:
        """What the model sees for this tool's result - fenced and labelled
        DATA if the content is untrusted free text, verbatim otherwise.

        `result.steps` (and every eval script reading it) keeps the raw text;
        only what re-enters the prompt is fenced - wrapping the eval record
        too would make the transcript unreadable in `race.py`/`trajectory.py`
        output for no security benefit, since nothing there re-feeds an LLM.
        """
        if action_name not in _UNTRUSTED_CONTENT_TOOLS:
            return observation
        verdict = injection.scan(observation)
        if verdict.suspicious:
            log.warning(
                "Tool observation flagged by injection scan",
                claim_number=claim_number,
                action=action_name,
                severity=verdict.severity.value,
                signals=verdict.signals,
            )
        return injection.wrap_tool_observation(action_name, observation)

    async def _auto_call(name: str, args: dict) -> ToolResult:
        """Run a tool the model never asked for, and record it exactly like one
        it did - visible in `result.steps`, and fingerprinted so a later model
        decision that duplicates it hits the repeated-action guard instead of
        re-running it."""
        tool_result = await run_tool(name, args, session=session, subject=subject)
        result.steps.append(
            StepLog("(automatic)", name, args, tool_result.observation, tool_result.ok)
        )
        transcript.append(f"Action: {name}({args})")
        transcript.append(f"Observation: {_transcript_observation(name, tool_result.observation)}")
        seen_actions.add(f"{name}:{json.dumps(args, sort_keys=True, default=str)}")
        return tool_result

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

        result.steps.append(
            StepLog(thought, action, action_input, tool_result.observation, tool_result.ok)
        )
        transcript.append(f"Action: {action}({action_input})")
        shown = _transcript_observation(action, tool_result.observation)
        transcript.append(f"Observation: {shown}")

        if action == "get_claim_status" and tool_result.ok and tool_result.data:
            claim = tool_result.data["claim"]
            memory["product_name"] = claim.product_name
            memory["date_of_loss"] = claim.date_of_loss
            memory["status"] = claim.status
            memory["rejection_clause_ref"] = claim.rejection_clause_ref

            # Auto-fetch, not a model decision. Both calls below are triggered
            # by conditions the fixed sequence (app/claims/handover.py) already
            # evaluates unconditionally from the fetched claim - neither was
            # ever a judgment call, so neither is left in TOOL_SPECS. Costs a
            # tool round-trip each, zero extra LLM calls: they do not consume a
            # `_step_num` slot or a `complete()` call.
            #
            # get_claim_notes: Week 8's trajectory eval found the model
            # silently skipped it in 10/20 real-claim runs even though the
            # outcome assertions still passed - a right-looking answer missing
            # an operative note.
            await _auto_call("get_claim_notes", {"claim_number": claim_number})

            # search_policy_clause: discovered *because of* the notes fix
            # above - once notes were always present, the model saw a note
            # repeating the rejection reason and skipped fetching the actual
            # clause 4/5 times (was 0/5 before), and 2 of those runs then
            # failed `denial_cites_clause` outright - a denial with no citable
            # clause. Same root cause as the notes gap (a mechanical decision
            # left to judgment), so it gets the same fix.
            if claim.status is ClaimStatus.REJECTED and claim.rejection_clause_ref:
                await _auto_call(
                    "search_policy_clause",
                    {
                        "clause_ref": claim.rejection_clause_ref,
                        "product_name": claim.product_name,
                        "date_of_loss": claim.date_of_loss,
                    },
                )
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

    result.raw_answer = result.answer

    # Output validation: the fence and the prompt's explicit warning both
    # failed against Week 8's injection payload (10/10 trials still shipped
    # "approved"). This check does not trust the model to have resisted it -
    # it compares what the model wrote against the status `get_claim_status`
    # actually observed, gated on the enum, not on prose (a text check like
    # "does the answer contain a denial word" has a false negative here: the
    # clause text legitimately contains "excluded").
    if memory.get("status") is ClaimStatus.REJECTED and APPROVAL_STATUS_RE.search(result.answer):
        log.warning(
            "Final answer contradicted observed status - overriding",
            claim_number=claim_number,
            raw_answer=result.answer[:200],
        )
        result.output_overridden = True
        clause_ref = memory.get("rejection_clause_ref")
        clause_note = f" (clause {clause_ref})" if clause_ref else ""
        result.answer = (
            f"Claim {claim_number} is REJECTED{clause_note}. The generated handover text "
            "was withheld because it contradicted the claim's observed status."
        )

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result
