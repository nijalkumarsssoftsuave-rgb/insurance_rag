"""Race the claim-handover agent against the fixed sequence, on real claims.

    python -m eval.week7.race

Runs both `app.agents.claim_handover_agent` and `app.claims.handover` against
every claim `scripts/seed_claims.py` creates (one of each branch: settled,
under_review, info_required, rejected-with-clause) plus one claim number that
does not exist. Reliability is checked with the two Week-6 assertions that
transfer directly to this task: the claim number must be echoed correctly, and
a stated rejection must cite a clause. Writes `eval/week7/race_results.json`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents import claim_handover_agent  # noqa: E402
from app.claims import handover, repository  # noqa: E402
from app.claims.summary import from_claim_view  # noqa: E402
from app.core.enums import UserRole  # noqa: E402
from app.db.session import async_session_scope  # noqa: E402
from app.logging import configure_logging  # noqa: E402
from app.security.authz import AuthSubject  # noqa: E402
from eval.assertions import claim_number_echoed, denial_cites_clause  # noqa: E402

CLAIM_NUMBERS = [
    "CLM-2026-0001",
    "CLM-2026-0002",
    "CLM-2026-0003",
    "CLM-2026-0004",
    "CLM-9999-9999",
]

STAFF_SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
)


def _check(answer: str, record: dict | None) -> dict[str, bool]:
    if record is None:
        return {}
    ok_number, _ = claim_number_echoed(answer, record)
    ok_clause, _ = denial_cites_clause(answer, record)
    return {"claim_number_echoed": ok_number, "denial_cites_clause": ok_clause}


async def _race_one(claim_number: str) -> dict:
    async with async_session_scope() as session:
        claim = await repository.get_claim(session, STAFF_SUBJECT, claim_number)
        record = from_claim_view(claim) if claim else None

    async with async_session_scope() as session:
        t0 = time.perf_counter()
        fixed = await handover.run(session, STAFF_SUBJECT, claim_number)
        fixed_wall_ms = int((time.perf_counter() - t0) * 1000)

    async with async_session_scope() as session:
        t0 = time.perf_counter()
        agent = await claim_handover_agent.run(session, STAFF_SUBJECT, claim_number)
        agent_wall_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "claim_number": claim_number,
        "fixed": {
            "answer": fixed.answer,
            "wall_ms": fixed_wall_ms,
            "tool_calls": fixed.tool_calls,
            "llm_calls": fixed.llm_calls,
            "input_tokens": fixed.input_tokens,
            "output_tokens": fixed.output_tokens,
            "checks": _check(fixed.answer, record),
        },
        "agent": {
            "answer": agent.answer,
            "wall_ms": agent_wall_ms,
            "tool_calls": agent.tool_calls,
            "llm_calls": agent.llm_calls,
            "input_tokens": agent.input_tokens,
            "output_tokens": agent.output_tokens,
            "stop_reason": agent.stop_reason,
            "steps": [
                {"action": s.action, "action_input": s.action_input, "ok": s.ok}
                for s in agent.steps
            ],
            "checks": _check(agent.answer, record),
        },
    }


async def main() -> int:
    configure_logging(json_output=False)
    results = [await _race_one(c) for c in CLAIM_NUMBERS]

    out_path = Path(__file__).parent / "race_results.json"
    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    print(f"{'claim':<16}{'fixed ms':>10}{'agent ms':>10}{'fixed tok':>11}{'agent tok':>11}"
          f"{'agent calls':>13}{'stop_reason':>18}")
    for r in results:
        f, a = r["fixed"], r["agent"]
        f_tok = f["input_tokens"] + f["output_tokens"]
        a_tok = a["input_tokens"] + a["output_tokens"]
        print(
            f"{r['claim_number']:<16}{f['wall_ms']:>10}{a['wall_ms']:>10}"
            f"{f_tok:>11}{a_tok:>11}{a['llm_calls']:>13}{a['stop_reason']:>18}"
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
