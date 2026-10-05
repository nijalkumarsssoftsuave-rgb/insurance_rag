"""Manager + 2 specialists: an orchestrator-worker squad for the claim
handover task (Week 10 · Module 5 · Track D).

Same task as `app/agents/claim_handover_agent.py` (Week 7-8's single agent) -
`eval/week10/race.py` runs both against the same claims and reports quality,
speed, tokens and cost for each, so "would a team beat one agent here" is
measured, not assumed.

Two specialists, each with one narrow job and its own tools:

    status-specialist   get_claim_status, get_claim_notes -> a status report
    clause-specialist    search_policy_clause             -> a clause explanation

...and a manager that hands work to them and synthesizes their reports into
one handover note. Every specialist call is a *fresh* LLM context - its own
system prompt, no shared transcript - and every report a specialist writes is
re-sent in full into the manager's own prompt. That re-send is the "hidden
cost" this week's brief names, and it is what this design pays, honestly, to
find out whether splitting the work is worth it here.

The routing decision (does this claim need the clause specialist at all) is
*not* an LLM call, on purpose, carried forward from Week 8: the fixed
sequence (`app/claims/handover.py`) already proves that decision is
mechanical, not a judgment call, for this task. Encoding it in Python is the
same lesson applied to the manager that was already applied to the single
agent - not a shortcut around what a manager does, since the two
architectures should be compared with the same already-known optimizations,
not a hand-built agent against a deliberately-naive one. See
`eval/week10/RESULTS.md` for where a *real* orchestration decision would
actually need the manager's judgment.

Output validation carries forward too: the fence on specialist reports is not
sufficient on its own (Week 8 measured this directly, 10/10), so the manager's
final answer is checked against the status actually observed before it ships.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.claim_handover_agent import APPROVAL_STATUS_RE
from app.claims.tools import get_claim_notes, get_claim_status, search_policy_clause
from app.core.enums import ClaimStatus
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry
from app.security import injection
from app.security.authz import AuthSubject

log = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AgentCard:
    """A minimal, in-process nod to A2A's discovery card: a name, a one-line
    purpose, and the skill(s) offered. Not network-exposed - these
    specialists are plain async functions called directly in the same
    process, so there is nothing to discover over the wire and no AgentCard
    JSON to publish. See RESULTS.md for when a real AgentCard + A2A handshake
    (cross-process, cross-team, or cross-org agents) would actually earn its
    keep over a direct call like this one.
    """

    name: str
    description: str
    skills: tuple[str, ...]


STATUS_SPECIALIST_CARD = AgentCard(
    name="status-specialist",
    description="Investigates one claim's status, dates, amounts and adjuster notes.",
    skills=("get_claim_status", "get_claim_notes"),
)
CLAUSE_SPECIALIST_CARD = AgentCard(
    name="clause-specialist",
    description="Explains the policy clause behind a claim rejection in plain language.",
    skills=("search_policy_clause",),
)


@dataclass(slots=True)
class SpecialistReport:
    agent: str
    report: str
    ok: bool
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # Facts the manager validates against later, taken from the typed record -
    # never re-derived from a specialist's prose (the same "gate on the enum,
    # not the text" lesson Week 8 learned the hard way).
    status: ClaimStatus | None = None
    rejection_clause_ref: str | None = None
    product_name: str | None = None
    date_of_loss: object = None


async def run_status_specialist(
    session: AsyncSession, subject: AuthSubject, claim_number: str
) -> SpecialistReport:
    status_result = await get_claim_status(session, subject, claim_number=claim_number)
    if not status_result.ok or not status_result.data:
        return SpecialistReport(
            agent=STATUS_SPECIALIST_CARD.name,
            report=status_result.observation,
            ok=False,
            tool_calls=1,
        )
    claim = status_result.data["claim"]

    notes_result = await get_claim_notes(session, subject, claim_number=claim_number)
    notes_shown = injection.wrap_tool_observation("get_claim_notes", notes_result.observation)

    prompt = registry.load("squad_status_specialist")
    completion = await get_llm().complete(
        [system(prompt.body), user(f"Status: {status_result.observation}\n\n{notes_shown}")],
        temperature=0.0,
    )
    return SpecialistReport(
        agent=STATUS_SPECIALIST_CARD.name,
        report=completion.text.strip(),
        ok=True,
        tool_calls=2,
        llm_calls=1,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        status=claim.status,
        rejection_clause_ref=claim.rejection_clause_ref,
        product_name=claim.product_name,
        date_of_loss=claim.date_of_loss,
    )


async def run_clause_specialist(
    session: AsyncSession,
    subject: AuthSubject,
    *,
    clause_ref: str,
    product_name: str | None,
    date_of_loss: object,
) -> SpecialistReport:
    clause_result = await search_policy_clause(
        session,
        subject,
        clause_ref=clause_ref,
        product_name=product_name,
        date_of_loss=date_of_loss,
    )
    if not clause_result.ok:
        return SpecialistReport(
            agent=CLAUSE_SPECIALIST_CARD.name,
            report=clause_result.observation,
            ok=False,
            tool_calls=1,
        )

    clause_shown = injection.wrap_tool_observation(
        "search_policy_clause", clause_result.observation
    )
    prompt = registry.load("squad_clause_specialist")
    completion = await get_llm().complete(
        [system(prompt.body), user(clause_shown)], temperature=0.0
    )
    return SpecialistReport(
        agent=CLAUSE_SPECIALIST_CARD.name,
        report=completion.text.strip(),
        ok=True,
        tool_calls=1,
        llm_calls=1,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


@dataclass(slots=True)
class SquadResult:
    answer: str
    raw_answer: str = ""
    output_overridden: bool = False
    reports: list[SpecialistReport] = field(default_factory=list)
    elapsed_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0


def _accumulate(result: SquadResult, report: SpecialistReport) -> None:
    result.tool_calls += report.tool_calls
    result.llm_calls += report.llm_calls
    result.input_tokens += report.input_tokens
    result.output_tokens += report.output_tokens


async def run(session: AsyncSession, subject: AuthSubject, claim_number: str) -> SquadResult:
    started = time.perf_counter()
    result = SquadResult(answer="")

    status_report = await run_status_specialist(session, subject, claim_number)
    result.reports.append(status_report)
    _accumulate(result, status_report)

    if not status_report.ok:
        result.answer = result.raw_answer = status_report.report
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    clause_report: SpecialistReport | None = None
    if status_report.status is ClaimStatus.REJECTED and status_report.rejection_clause_ref:
        clause_report = await run_clause_specialist(
            session,
            subject,
            clause_ref=status_report.rejection_clause_ref,
            product_name=status_report.product_name,
            date_of_loss=status_report.date_of_loss,
        )
        result.reports.append(clause_report)
        _accumulate(result, clause_report)

    # Manager: synthesize. Both reports are re-sent in full - the hand-off
    # cost this week measures.
    manager_input = f"Claim {claim_number}.\nStatus specialist report: {status_report.report}"
    if clause_report is not None:
        manager_input += f"\nClause specialist report: {clause_report.report}"

    prompt = registry.load("squad_manager")
    completion = await get_llm().complete(
        [system(prompt.body), user(manager_input)], temperature=0.0
    )
    result.llm_calls += 1
    result.input_tokens += completion.input_tokens
    result.output_tokens += completion.output_tokens
    result.answer = result.raw_answer = completion.text.strip()

    if status_report.status is ClaimStatus.REJECTED and APPROVAL_STATUS_RE.search(result.answer):
        log.warning(
            "Squad final answer contradicted observed status - overriding",
            claim_number=claim_number,
            raw_answer=result.answer[:200],
        )
        result.output_overridden = True
        clause_ref = status_report.rejection_clause_ref
        clause_note = f" (clause {clause_ref})" if clause_ref else ""
        result.answer = (
            f"Claim {claim_number} is REJECTED{clause_note}. The generated handover text "
            "was withheld because it contradicted the claim's observed status."
        )

    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result
