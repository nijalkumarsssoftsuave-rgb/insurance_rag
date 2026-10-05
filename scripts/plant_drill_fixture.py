"""Plants one deliberately bad historical answer for Week 11's support drill
(Track D: "Find the coverage answer that ignored an exclusion").

    python scripts/plant_drill_fixture.py

The brief asks to "practise finding a planted bad answer from a vague
complaint" - this is that plant, not a claim that the current pipeline
produces it today. Dental treatment is excluded under Family Health Optima
unless accident-related (clause 4.11, SECTION 4 - EXCLUSIONS) - confirmed
live this session (eval/week11/, and every coverage question asked through
the real API this session correctly states the exclusion). The planted row
answers "yes, covered" instead, with no mention of the exclusion: the shape
of failure this week's drill is for, not a claim about a bug that exists now.

Backdated ~3 weeks so "a customer complained last month" is literally true
of the data, and written through the same `Message`/`Conversation` models
and roughly the same `trace` shape `_persist()` writes - close enough that a
log search cannot tell it apart from a real row by its shape, only by
content. `trace.planted = true` is the one tell, present only so this
exact fixture can be identified and removed later; the drill below does not
use it, searching from the complaint text alone instead, the same way a
real support engineer would have to.

Idempotent: finds and replaces its own prior row by thread_id, like
scripts/seed_claims.py.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.core.enums import Intent, MessageRole  # noqa: E402
from app.db.models import Conversation, Message  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.logging import configure_logging  # noqa: E402

THREAD_ID = "week11-drill-fixture-dental-exclusion"
PLANTED_AT = datetime.now(UTC) - timedelta(days=23)

QUESTION = "Is dental treatment covered under my Family Health Optima policy?"
BAD_ANSWER = "Yes, dental treatment is covered under your Family Health Optima policy."


def main() -> int:
    configure_logging(json_output=False)
    with session_scope() as session:
        existing = session.scalar(select(Conversation).where(Conversation.thread_id == THREAD_ID))
        if existing:
            session.execute(delete(Message).where(Message.conversation_id == existing.id))
            session.execute(delete(Conversation).where(Conversation.id == existing.id))
            session.flush()

        conversation = Conversation(
            tenant_id="default",
            thread_id=THREAD_ID,
            title=QUESTION[:200],
            created_at=PLANTED_AT,
        )
        session.add(conversation)
        session.flush()

        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=QUESTION,
                created_at=PLANTED_AT,
            )
        )
        session.add(
            Message(
                id=uuid.uuid4(),
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=BAD_ANSWER,
                intent=Intent.POLICY_QA,
                model="gpt-4o-mini",
                prompt_version="generate@v1",
                retrieved_chunk_ids=[],
                citations=[],
                confidence=0.85,
                abstained=False,
                latency_ms=2100,
                input_tokens=620,
                output_tokens=24,
                created_at=PLANTED_AT + timedelta(seconds=2),
                trace={
                    "schema_version": 1,
                    "planted": True,
                    "planted_reason": "Week 11 Track D support-drill fixture - never shown "
                    "to a real customer. See eval/week11/RESULTS.md.",
                    "llm": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 1024,
                    },
                    "retrieval": {
                        "variants": [QUESTION],
                        "top_score": 0.41,
                        "below_threshold": False,
                        "broadened": False,
                        # No exclusion chunk present - the shape of the
                        # retrieval miss this answer is standing in for.
                        "chunks": [
                            {
                                "chunk_id": "planted-chunk-scope-of-cover",
                                "section_path": "SECTION 2 - SCOPE OF COVER",
                                "score": 0.41,
                            }
                        ],
                    },
                    "router": {
                        "intent": "policy_qa",
                        "product_name": "Family Health Optima",
                        "claim_number": None,
                        "is_coverage_question": True,
                    },
                    "guard": {
                        "injection_severity": "none",
                        "pii_placeholders": [],
                        "blocked": False,
                    },
                    "verification": {"verified": True, "notes": [], "needs_human": False},
                    "timings_ms": {
                        "route": 410,
                        "retrieve_total": 1340,
                        "generate": 980,
                        "verify": 710,
                    },
                    "tokens_by_step": {
                        "route": {"input_tokens": 180, "output_tokens": 20},
                        "generate": {"input_tokens": 340, "output_tokens": 18},
                        "verify": {"input_tokens": 100, "output_tokens": 6},
                    },
                },
            )
        )

    print(f"Planted drill fixture: thread_id={THREAD_ID}")
    print(f"  question:  {QUESTION}")
    print(f"  answer:    {BAD_ANSWER}")
    print(f"  created_at: {PLANTED_AT.isoformat()} (~23 days ago)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
