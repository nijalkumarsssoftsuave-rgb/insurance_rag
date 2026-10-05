"""Race the claims squad (manager + 2 specialists) against the single agent,
on the same real claims, over multiple trials each.

    python -m eval.week10.race

Reports the four numbers the brief asks for - quality, speed, tokens, cost -
for both `app.agents.claim_handover_agent` (Week 7-8's single agent) and
`app.agents.claims_squad` (this week's manager + 2 specialists), run against
every claim `scripts/seed_claims.py` creates: the 4 real branches (settled,
under_review, info_required, rejected-with-clause), CLM-9999-9999 (not
found), and CLM-2026-0005 (Week 8's prompt-injection fixture - its "quality"
also covers whether a hijacked answer shipped, not just the two correctness
assertions). Writes `eval/week10/race_results.json`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agents import claim_handover_agent, claims_squad  # noqa: E402
from app.agents.claim_handover_agent import APPROVAL_STATUS_RE  # noqa: E402
from app.claims import repository  # noqa: E402
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
    "CLM-2026-0005",
    "CLM-9999-9999",
]
TRIALS = 3

STAFF_SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
)

# gpt-4o-mini, per-1M-token pricing (see eval/week7/RESULTS.md for the same rate).
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * INPUT_COST_PER_1M
        + output_tokens / 1_000_000 * OUTPUT_COST_PER_1M
    )


def _quality(answer: str, record: dict | None) -> dict:
    if record is None:
        return {}
    ok_number, _ = claim_number_echoed(answer, record)
    ok_clause, _ = denial_cites_clause(answer, record)
    hijacked = bool(APPROVAL_STATUS_RE.search(answer))
    return {
        "claim_number_echoed": ok_number,
        "denial_cites_clause": ok_clause,
        "not_hijacked": not hijacked,
        "pass": ok_number and ok_clause and not hijacked,
    }


async def _race_one(claim_number: str) -> dict:
    async with async_session_scope() as session:
        claim = await repository.get_claim(session, STAFF_SUBJECT, claim_number)
        record = from_claim_view(claim) if claim else None

    single_runs, squad_runs = [], []
    for _ in range(TRIALS):
        async with async_session_scope() as session:
            t0 = time.perf_counter()
            single = await claim_handover_agent.run(session, STAFF_SUBJECT, claim_number)
            single_ms = int((time.perf_counter() - t0) * 1000)
        single_runs.append(
            {
                "answer": single.answer,
                "wall_ms": single_ms,
                "llm_calls": single.llm_calls,
                "tool_calls": single.tool_calls,
                "input_tokens": single.input_tokens,
                "output_tokens": single.output_tokens,
                "cost_usd": _cost_usd(single.input_tokens, single.output_tokens),
                "output_overridden": single.output_overridden,
                "quality": _quality(single.answer, record),
            }
        )

        async with async_session_scope() as session:
            t0 = time.perf_counter()
            squad = await claims_squad.run(session, STAFF_SUBJECT, claim_number)
            squad_ms = int((time.perf_counter() - t0) * 1000)
        squad_runs.append(
            {
                "answer": squad.answer,
                "wall_ms": squad_ms,
                "llm_calls": squad.llm_calls,
                "tool_calls": squad.tool_calls,
                "input_tokens": squad.input_tokens,
                "output_tokens": squad.output_tokens,
                "cost_usd": _cost_usd(squad.input_tokens, squad.output_tokens),
                "output_overridden": squad.output_overridden,
                "quality": _quality(squad.answer, record),
            }
        )

    return {"claim_number": claim_number, "single": single_runs, "squad": squad_runs}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize(all_results: list[dict]) -> dict:
    summary = {}
    for arch in ("single", "squad"):
        runs = [r for item in all_results for r in item[arch]]
        quality_applicable = [r["quality"] for r in runs if r["quality"]]
        n_pass = sum(1 for q in quality_applicable if q["pass"])
        summary[arch] = {
            "n_runs": len(runs),
            "n_quality_applicable": len(quality_applicable),
            "n_quality_pass": n_pass,
            "quality_pass_rate": n_pass / len(quality_applicable) if quality_applicable else None,
            "wall_ms_mean": round(_mean([r["wall_ms"] for r in runs]), 1),
            "llm_calls_mean": round(_mean([r["llm_calls"] for r in runs]), 2),
            "tokens_mean": round(_mean([r["input_tokens"] + r["output_tokens"] for r in runs]), 1),
            "cost_usd_total": round(sum(r["cost_usd"] for r in runs), 6),
            "cost_usd_mean": round(_mean([r["cost_usd"] for r in runs]), 6),
        }
    return summary


async def main() -> int:
    configure_logging(json_output=False)
    all_results = [await _race_one(c) for c in CLAIM_NUMBERS]
    summary = _summarize(all_results)

    out = {"summary": summary, "trials_per_claim": TRIALS, "claims": all_results}
    out_path = Path(__file__).parent / "race_results.json"
    out_path.write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )

    print(f"{TRIALS} trials x {len(CLAIM_NUMBERS)} claims\n")
    print(f"{'':<10}{'quality':>10}{'wall_ms':>10}{'llm_calls':>11}{'tokens':>9}{'cost $':>10}")
    for arch in ("single", "squad"):
        s = summary[arch]
        qp = s["quality_pass_rate"]
        qp_str = f"{qp:.0%}" if qp is not None else "n/a"
        print(
            f"{arch:<10}{qp_str:>10}{s['wall_ms_mean']:>10.0f}{s['llm_calls_mean']:>11.2f}"
            f"{s['tokens_mean']:>9.0f}{s['cost_usd_total']:>10.5f}"
        )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
