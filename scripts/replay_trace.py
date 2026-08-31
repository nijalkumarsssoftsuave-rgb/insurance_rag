"""Replay one recorded turn from its trace row alone, and diff it against the original.

    python scripts/replay_trace.py <trace_id>
    python scripts/replay_trace.py <trace_id> --retrieval-only

"From the trace alone" is the point: this reads the stored row and reconstructs
the inputs from it - the question, the router's filters, the decoding parameters -
rather than re-deriving them by running the router again. Anything the row does
not carry is reported as unreconstructable instead of being quietly re-inferred,
because a replay that silently re-derives its own inputs proves nothing about
whether the trace was sufficient.

Two honest sources of drift the diff will show, neither of them a replay bug:

* **Corpus drift.** Traces written before a document was added or removed ran
  against a corpus that no longer exists, so retrieval legitimately differs.
* **Prompt drift.** `prompt_version` records which version answered; the registry
  only holds the current one. If they differ, the replay used a different prompt
  and says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from sqlalchemy import text  # noqa: E402

from app.db.session import get_sync_session_factory  # noqa: E402

ROW = """
SELECT a.id::text AS trace_id, a.created_at, a.content AS answer, a.intent::text AS intent,
       a.model, a.prompt_version, a.confidence, a.abstained, a.latency_ms,
       a.retrieved_chunk_ids, a.citations, a.trace, a.input_tokens, a.output_tokens,
       (SELECT m.content FROM messages m
         WHERE m.conversation_id = a.conversation_id AND m.role = 'user'
           AND m.created_at <= a.created_at
         ORDER BY m.created_at DESC LIMIT 1) AS question
FROM messages a
WHERE a.id = CAST(:tid AS uuid)
"""

# Fields the task requires a trace to carry for replay. Each maps to where it
# lives on the row, so a missing one can be named rather than guessed at.
REQUIRED = {
    "question": lambda r: r["question"],
    "raw output": lambda r: r["answer"],
    "model": lambda r: r["model"],
    "prompt version": lambda r: r["prompt_version"],
    "retrieved chunk ids": lambda r: r["retrieved_chunk_ids"],
    "retrieved chunk scores": lambda r: ((r["trace"] or {}).get("retrieval") or {}).get("chunks"),
    "model params": lambda r: (r["trace"] or {}).get("llm"),
    "router filters": lambda r: (r["trace"] or {}).get("router"),
    "query variants": lambda r: ((r["trace"] or {}).get("retrieval") or {}).get("variants"),
    "input/output tokens": lambda r: r["input_tokens"] or r["output_tokens"],
}


def load(trace_id: str) -> dict:
    with get_sync_session_factory()() as session:
        row = session.execute(text(ROW), {"tid": trace_id}).mappings().first()
    if row is None:
        raise SystemExit(f"no trace {trace_id}")
    return dict(row)


async def replay(row: dict, retrieval_only: bool) -> dict:
    from app.retrieval import filters
    from app.retrieval.hybrid import RetrievalPipeline
    from app.security.injection import wrap_untrusted

    tr = row["trace"] or {}
    router = tr.get("router") or {}
    dol = router.get("date_of_loss")
    rf = filters.RetrievalFilter(
        product_name=router.get("product_name"),
        date_of_loss=date.fromisoformat(dol) if dol else None,
    )
    outcome = await RetrievalPipeline().retrieve(row["question"], retrieval_filter=rf)
    out = {
        "top_score": round(outcome.top_score, 4),
        "chunk_ids": outcome.context.chunk_ids,
        "sections": [b.section_path for b in outcome.context.blocks],
        "answer": None,
    }
    if retrieval_only:
        return out

    from app.graph.nodes.generate import GroundedAnswer
    from app.llm import get_llm, system, user
    from app.prompts import registry

    prompt = registry.load("generate")
    ctx = wrap_untrusted(outcome.context.as_pairs())
    result = await get_llm().structured(
        [system(prompt.body), user(f"{ctx}\n\nQuestion: {row['question']}")],
        GroundedAnswer,
        temperature=(tr.get("llm") or {}).get("temperature", prompt.temperature or 0.1),
    )
    out["answer"] = result.answer
    out["replayed_prompt_version"] = prompt.version
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace_id")
    ap.add_argument("--retrieval-only", action="store_true", help="skip the LLM call")
    args = ap.parse_args()

    row = load(args.trace_id)
    missing = [name for name, get in REQUIRED.items() if not get(row)]

    print(f"trace_id        : {row['trace_id']}")
    print(f"recorded at     : {row['created_at']}")
    print(f"question        : {row['question']}")
    print(f"model / prompt  : {row['model']} / {row['prompt_version']}")
    print(f"trace bundle    : {'present (v%s)' % row['trace'].get('schema_version') if row['trace'] else 'ABSENT - written before the trace column existed'}")
    print(f"unreconstructable fields ({len(missing)}): {', '.join(missing) if missing else 'none'}")

    replayed = asyncio.run(replay(row, args.retrieval_only))

    print("\n" + "=" * 78)
    print("ORIGINAL")
    print("=" * 78)
    print(f"  top_score : {((row['trace'] or {}).get('retrieval') or {}).get('top_score')}")
    print(f"  chunks    : {len(row['retrieved_chunk_ids'] or [])} -> {(row['retrieved_chunk_ids'] or [])[:3]}")
    print(f"  answer    : {(row['answer'] or '')[:400]}")

    print("\n" + "=" * 78)
    print("REPLAYED")
    print("=" * 78)
    print(f"  top_score : {replayed['top_score']}")
    print(f"  chunks    : {len(replayed['chunk_ids'])} -> {replayed['chunk_ids'][:3]}")
    print(f"  sections  : {replayed['sections'][:4]}")
    if replayed["answer"] is not None:
        print(f"  answer    : {replayed['answer'][:400]}")

    same_chunks = set(row["retrieved_chunk_ids"] or []) & set(replayed["chunk_ids"])
    print("\n" + "-" * 78)
    print(f"chunk overlap   : {len(same_chunks)} of {len(row['retrieved_chunk_ids'] or [])} original ids also retrieved on replay")
    if row["prompt_version"] and replayed.get("replayed_prompt_version") not in (None, row["prompt_version"]):
        print(f"prompt drift    : recorded {row['prompt_version']}, replayed {replayed['replayed_prompt_version']}")
    print(json.dumps({"trace_id": row["trace_id"], "unreconstructable": missing}, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
