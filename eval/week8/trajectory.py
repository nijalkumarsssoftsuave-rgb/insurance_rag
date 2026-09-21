"""Trajectory eval: does the agent take the right *path*, not just land on a
right-looking answer?

    python -m eval.week8.trajectory --out eval/week8/trajectory_before.json

Runs the agent `TRIALS` times against each of the 4 real seeded claims (even
at temperature 0.0, the JSON-mode decision step varies run to run - this is a
genuine batch, not a formality). The oracle for "which tools should have run"
is `app.claims.handover.expected_tools()` - the same fixed-sequence logic Week
7 already proved correct, not a second, driftable copy of the branch
condition. For each run:

  - outcome: the two Week-6/7 assertions that transfer to this task
    (`claim_number_echoed`, `denial_cites_clause`)
  - trajectory: does the agent's actual tool-call set cover the oracle's set?
  - the gap: outcome PASSED but trajectory was INCOMPLETE - a right-looking
    answer reached by skipping a step the fixed sequence never skips.

Also reports cost per task (mean & p99 of tokens and wall time) across the
whole batch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
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

REAL_CLAIMS = ["CLM-2026-0001", "CLM-2026-0002", "CLM-2026-0003", "CLM-2026-0004"]
TRIALS = 5

STAFF_SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
)


def _outcome_pass(answer: str, record: dict) -> bool:
    ok_number, _ = claim_number_echoed(answer, record)
    ok_clause, _ = denial_cites_clause(answer, record)
    return ok_number and ok_clause


async def _run_trials(claim_number: str) -> list[dict]:
    async with async_session_scope() as session:
        claim = await repository.get_claim(session, STAFF_SUBJECT, claim_number)
    assert claim is not None, f"{claim_number} must be a real seeded claim"
    record = from_claim_view(claim)
    expected = sorted(handover.expected_tools(claim))

    runs = []
    for _ in range(TRIALS):
        async with async_session_scope() as session:
            t0 = time.perf_counter()
            result = await claim_handover_agent.run(session, STAFF_SUBJECT, claim_number)
            wall_ms = int((time.perf_counter() - t0) * 1000)

        # `called` counts a step regardless of outcome; `called_successfully`
        # requires `ok=True`. A tool that ran but failed (e.g. a real retrieval
        # miss) must not count as "the trajectory covered it" - that would let
        # a broken tool call masquerade as a complete path.
        called = sorted({s.action for s in result.steps})
        called_ok = sorted({s.action for s in result.steps if s.ok})
        outcome_pass = _outcome_pass(result.answer, record)
        trajectory_complete = set(expected) <= set(called_ok)
        runs.append(
            {
                "claim_number": claim_number,
                "expected_tools": expected,
                "called_tools": called,
                "called_tools_ok": called_ok,
                "outcome_pass": outcome_pass,
                "trajectory_complete": trajectory_complete,
                "outcome_pass_trajectory_incomplete": outcome_pass and not trajectory_complete,
                "wall_ms": wall_ms,
                "total_tokens": result.input_tokens + result.output_tokens,
                "llm_calls": result.llm_calls,
                "stop_reason": result.stop_reason,
            }
        )
    return runs


def _p99(values: list[int]) -> float:
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=100, method="inclusive")[98]


def _summarize(runs: list[dict]) -> dict:
    n = len(runs)
    n_gap = sum(1 for r in runs if r["outcome_pass_trajectory_incomplete"])
    n_complete = sum(1 for r in runs if r["trajectory_complete"])
    n_outcome_pass = sum(1 for r in runs if r["outcome_pass"])
    tokens = [r["total_tokens"] for r in runs]
    wall = [r["wall_ms"] for r in runs]
    return {
        "n_runs": n,
        "outcome_pass_rate": n_outcome_pass / n,
        "trajectory_complete_rate": n_complete / n,
        "outcome_pass_trajectory_incomplete_rate": n_gap / n,
        "outcome_pass_trajectory_incomplete_count": n_gap,
        "cost_tokens_mean": statistics.mean(tokens),
        "cost_tokens_p99": _p99(tokens),
        "wall_ms_mean": statistics.mean(wall),
        "wall_ms_p99": _p99(wall),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "trajectory_results.json"))
    args = parser.parse_args()

    configure_logging(json_output=False)

    all_runs: list[dict] = []
    for claim_number in REAL_CLAIMS:
        all_runs.extend(await _run_trials(claim_number))

    summary = _summarize(all_runs)
    out = {"summary": summary, "runs": all_runs}

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    print(f"{TRIALS} trials x {len(REAL_CLAIMS)} claims = {len(all_runs)} runs")
    print(f"  outcome pass rate:               {summary['outcome_pass_rate']:.0%}")
    print(f"  trajectory complete rate:        {summary['trajectory_complete_rate']:.0%}")
    print(
        "  outcome-pass-but-trajectory-gap:  "
        f"{summary['outcome_pass_trajectory_incomplete_count']}/{summary['n_runs']} "
        f"({summary['outcome_pass_trajectory_incomplete_rate']:.0%})"
    )
    tok_mean, tok_p99 = summary["cost_tokens_mean"], summary["cost_tokens_p99"]
    ms_mean, ms_p99 = summary["wall_ms_mean"], summary["wall_ms_p99"]
    print(f"  cost tokens  mean/p99:  {tok_mean:.0f} / {tok_p99:.0f}")
    print(f"  wall ms      mean/p99:  {ms_mean:.0f} / {ms_p99:.0f}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
