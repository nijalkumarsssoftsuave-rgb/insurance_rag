"""Week 12 demo: the claim-summary seam, live, both sides.

Run after `make bootstrap` (needs CLM-2026-0005, the Week 8 injection
fixture, which `seed_claims.py` plants). No server needs to be running -
this talks to the database and the LLM directly, the same way the API
process does.

What it shows:
  1. The OLD path (`app.claims.summary.summarise`) is still importable and
     still vulnerable - calling it directly reproduces the Week 8 attack.
  2. The path the live API actually calls today (`claims_squad.run`) is not
     - either the model resists it, or the output-validation guard rewrites
     it and says so via `output_overridden`.

This is the reproduction behind eval/week12/RESULTS.md, runnable by anyone
who cloned the repo, not just something that happened once in a chat.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents import claims_squad  # noqa: E402
from app.claims import repository, summary  # noqa: E402
from app.core.enums import UserRole  # noqa: E402
from app.db.session import async_session_scope  # noqa: E402
from app.security.authz import AuthSubject  # noqa: E402

CLAIM_NUMBER = "CLM-2026-0005"


async def main() -> int:
    staff = AuthSubject(
        user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
    )

    async with async_session_scope() as session:
        claim = await repository.get_claim(session, staff, CLAIM_NUMBER)
        if claim is None:
            print(f"{CLAIM_NUMBER} not found - run `make seed` first.")
            return 1

        print(f"=== {CLAIM_NUMBER}, status={claim.status.value} ===\n")

        print("--- OLD path: app.claims.summary.summarise() ---")
        old_answer = await summary.summarise(summary.from_claim_view(claim))
        print(old_answer, "\n")

        print("--- CURRENT path: what GET /claims/{id}/summary actually calls ---")
        result = await claims_squad.run(session, staff, CLAIM_NUMBER)
        print(result.answer)
        print(f"\noutput_overridden={result.output_overridden}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
