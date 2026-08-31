"""Draw a seeded random sample of traces for error analysis.

    python scripts/sample_traces.py --seed 20260901 --n 20 --before 2026-08-30

A trace is one assistant turn: the question that produced it, the answer, and the
replay bundle written alongside. The sample is drawn with `random.Random(seed)`,
so pasting the seed and the frame is enough for anyone to redraw exactly this set.

**Why a frame instead of sampling everything.** The point of a random sample is
that it is not the traces you remember breaking. Turns generated while debugging
a specific bug are exactly those traces, so a frame that includes them produces
frequencies that look measured and are not. `--before` exists to exclude them,
and whatever frame is used has to be stated in the write-up next to the seed - a
hidden frame is worse than no frame.

Nothing here writes to the database. Sampling must not perturb what it samples.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.db.session import get_sync_session_factory  # noqa: E402

# One row per assistant turn, paired with the user turn immediately before it in
# the same conversation. LATERAL keeps that pairing correct when a conversation
# has many turns - matching on conversation alone would attach the wrong question.
QUERY = """
SELECT a.id::text            AS trace_id,
       c.thread_id::text     AS conversation_id,
       a.created_at          AS created_at,
       u.content             AS question,
       a.content             AS answer,
       a.intent::text        AS intent,
       a.model               AS model,
       a.prompt_version      AS prompt_version,
       a.confidence          AS confidence,
       a.abstained           AS abstained,
       a.latency_ms          AS latency_ms,
       a.retrieved_chunk_ids AS retrieved_chunk_ids,
       a.citations           AS citations,
       a.trace               AS trace
FROM messages a
JOIN conversations c ON c.id = a.conversation_id
LEFT JOIN LATERAL (
    SELECT m.content
    FROM messages m
    WHERE m.conversation_id = a.conversation_id
      AND m.role = 'user'
      AND m.created_at <= a.created_at
    ORDER BY m.created_at DESC
    LIMIT 1
) u ON TRUE
WHERE a.role = 'assistant'
{frame}
ORDER BY a.created_at
"""


def load_frame(before: str | None, after: str | None) -> list[dict]:
    # Built conditionally rather than with `:p IS NULL OR ...`: Postgres cannot
    # infer a type for a parameter that only ever appears in an IS NULL test.
    clauses, params = [], {}
    if before:
        clauses.append("  AND a.created_at <  CAST(:before AS timestamptz)")
        params["before"] = before
    if after:
        clauses.append("  AND a.created_at >= CAST(:after AS timestamptz)")
        params["after"] = after
    sql = QUERY.format(frame="\n".join(clauses))

    factory = get_sync_session_factory()
    with factory() as session:
        rows = session.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, help="paste this into the write-up")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--before", help="exclude turns at/after this timestamp")
    ap.add_argument("--after", help="exclude turns before this timestamp")
    ap.add_argument("--out", type=Path, help="write the drawn sample here as JSON")
    args = ap.parse_args()

    frame = load_frame(args.before, args.after)
    if not frame:
        raise SystemExit("empty sampling frame - check --before/--after")

    n = min(args.n, len(frame))
    drawn = random.Random(args.seed).sample(frame, n)
    drawn.sort(key=lambda r: r["created_at"])

    distinct_questions = len({r["question"] for r in drawn})
    print(f"seed        : {args.seed}")
    print(f"frame       : {len(frame)} assistant turns"
          f"{f', before {args.before}' if args.before else ''}"
          f"{f', after {args.after}' if args.after else ''}")
    print(f"frame distinct questions : {len({r['question'] for r in frame})}")
    print(f"drawn       : {n}")
    # A sample of 20 that collapses onto 4 questions has 20 traces and 4
    # behaviours; the frequencies in any taxonomy built on it would be fiction.
    print(f"drawn distinct questions : {distinct_questions}")
    print()
    print(f"| # | trace_id | when | intent | abstained | question |")
    print(f"|---|---|---|---|---|---|")
    for i, r in enumerate(drawn, 1):
        q = (r["question"] or "").replace("|", "\\|")[:58]
        when = r["created_at"].strftime("%m-%d %H:%M")
        print(f"| {i} | `{r['trace_id']}` | {when} | {r['intent'] or '-'} "
              f"| {'yes' if r['abstained'] else 'no'} | {q} |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "frame_size": len(frame),
                    "frame_before": args.before,
                    "frame_after": args.after,
                    "drawn_distinct_questions": distinct_questions,
                    "traces": drawn,
                },
                indent=2,
                default=str,
            ),
            encoding="utf8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
