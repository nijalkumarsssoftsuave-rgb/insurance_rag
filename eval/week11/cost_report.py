"""Cost per request, broken down by step - the baseline this week's "make one
improvement" is measured against.

    python -m eval.week11.cost_report --out eval/week11/cost_before.json

Runs the graph directly (same code path `/api/v1/chat` calls, no server
needed) against a representative mix of real questions: coverage questions,
a waiting-period question, a general policy question, and claim-status
questions (Lane B is templated and mostly free - included so the report
does not overstate the typical request). Each question runs `TRIALS` times.
Reports mean and p99 cost per request, overall and by step, using the same
`app.llm.pricing.cost_usd` the live app now logs with - not a second,
drifting copy of the pricing table.

Must be run with caching OFF (or not yet built) to mean anything as a
"before" baseline, per the same before/after discipline Week 8's assertion
fix and Week 10's race re-run both needed - a warm cache answering from the
cache instead of the model would make every number here read as "free" and
mean nothing.
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

from app.deps import DEMO_SUBJECT  # noqa: E402
from app.graph.builder import get_graph  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from app.llm.pricing import cost_usd  # noqa: E402
from app.logging import configure_logging  # noqa: E402

TRIALS = 3

QUESTIONS = [
    ("coverage_dental", "Is dental treatment covered under Family Health Optima?"),
    ("coverage_ped", "What is the waiting period for pre-existing diseases?"),
    ("coverage_maternity", "Is maternity covered and what is the waiting period?"),
    ("general_policy", "How do I file a cashless claim at a network hospital?"),
    ("claim_status", "What is the status of claim CLM-2026-0004?"),
]

# The real demo subject (app.deps), not a fresh random one - its user_id
# actually exists in the `users` table (scripts/seed_claims.py creates it),
# so the claim-status question resolves to a real claim and the audit write
# claim_lookup_node makes on every lookup does not fail the FK constraint a
# fabricated user_id would.
CUSTOMER_SUBJECT = DEMO_SUBJECT


async def _run_one(question: str) -> dict:
    graph = get_graph()
    conversation_id = str(uuid.uuid4())
    state = initial_state(
        question=question,
        subject=CUSTOMER_SUBJECT,
        conversation_id=conversation_id,
        thread_id=conversation_id,
    )
    t0 = time.perf_counter()
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": conversation_id}})
    wall_ms = int((time.perf_counter() - t0) * 1000)

    tokens_by_step = result.get("tokens_by_step") or {}
    cost_by_step = {
        step: cost_usd("gpt-4o-mini", t.get("input_tokens", 0), t.get("output_tokens", 0))
        for step, t in tokens_by_step.items()
    }
    total_input = sum(t.get("input_tokens", 0) for t in tokens_by_step.values())
    total_output = sum(t.get("output_tokens", 0) for t in tokens_by_step.values())
    return {
        "wall_ms": wall_ms,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": sum(cost_by_step.values()),
        "cost_by_step": cost_by_step,
        "abstained": bool(result.get("abstained")),
        "intent": result.get("intent").value if result.get("intent") else None,
    }


def _p99(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[98]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).parent / "cost_results.json"))
    args = parser.parse_args()

    configure_logging(json_output=False)

    by_question: dict[str, list[dict]] = {}
    for key, question in QUESTIONS:
        by_question[key] = [await _run_one(question) for _ in range(TRIALS)]

    all_runs = [r for runs in by_question.values() for r in runs]
    costs = [r["total_cost_usd"] for r in all_runs]
    summary = {
        "n_runs": len(all_runs),
        "cost_usd_mean": statistics.mean(costs),
        "cost_usd_p99": _p99(costs),
        "cost_usd_total": sum(costs),
        "wall_ms_mean": statistics.mean(r["wall_ms"] for r in all_runs),
    }

    # Per-step cost share of the total - this is the number that tells you
    # *what* to cache or route to a cheaper model, not just that it's pricey.
    step_totals: dict[str, float] = {}
    for r in all_runs:
        for step, c in r["cost_by_step"].items():
            step_totals[step] = step_totals.get(step, 0.0) + c
    grand_total = sum(step_totals.values()) or 1.0
    step_share = {s: round(c / grand_total, 3) for s, c in step_totals.items()}

    out = {
        "summary": summary,
        "step_totals_usd": step_totals,
        "step_share": step_share,
        "by_question": by_question,
    }
    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    print(f"{TRIALS} trials x {len(QUESTIONS)} questions = {len(all_runs)} runs\n")
    mean, p99 = summary["cost_usd_mean"], summary["cost_usd_p99"]
    print(f"cost/request  mean: ${mean:.6f}   p99: ${p99:.6f}")
    print(f"total cost (this run): ${summary['cost_usd_total']:.6f}")
    print("\nshare of cost by step:")
    for step, share in sorted(step_share.items(), key=lambda kv: -kv[1]):
        print(f"  {step:<18} {share:.0%}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
