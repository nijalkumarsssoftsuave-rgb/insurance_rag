"""Runs the golden set end to end and writes an eval_run record.

    python -m eval.harness --label baseline --out eval/runs/baseline.json

Calls ``RetrievalPipeline.retrieve`` directly rather than the HTTP API: the API
adds an LLM generation per question for a number that only measures retrieval,
and the whole point of separating retrieval failures from generation failures is
to be able to measure one without the other.

Every question is scored against the top-k of the **reranked** list, which is
what actually reaches the model. The per-question ``verdict`` column is the
Week-4 failure label:

    RETRIEVAL_FAIL   the right clause never made it into the top k. Nothing the
                     prompt does can fix this.
    RETRIEVED_OK     the right clause was there. Any wrong answer from here is a
                     generation failure, and is judged against reference_answer.
    ABSTAIN_OK       an unanswerable question that the confidence gate rejected.
    OVER_ANSWERED    an unanswerable question the gate let through - the
                     expensive failure, because it ends in a confident wrong
                     answer rather than "I could not find that".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.enums import DocType
from app.retrieval import filters
from app.retrieval.hybrid import RetrievalPipeline
from eval.metrics import retrieval as M

GOLDEN = Path("eval/golden/questions.jsonl")


def load(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids in the golden set")
    return rows


def build_filter(spec: dict[str, Any]) -> filters.RetrievalFilter:
    """Golden-set ``filters`` -> ``RetrievalFilter``.

    Unknown keys raise rather than being ignored: a typo'd filter key that is
    silently dropped produces a plausible number from the wrong query.
    """
    kwargs = dict(spec)
    if dol := kwargs.pop("date_of_loss", None):
        kwargs["date_of_loss"] = date.fromisoformat(dol)
    if doc_types := kwargs.pop("doc_types", None):
        kwargs["doc_types"] = [DocType(d) for d in doc_types]
    return filters.RetrievalFilter(**kwargs)


async def run_one(pipeline: RetrievalPipeline, row: dict[str, Any], k: int) -> dict[str, Any]:
    outcome = await pipeline.retrieve(row["question"], retrieval_filter=build_filter(row.get("filters", {})))
    payloads = [h.hit.payload for h in outcome.ranked]
    labels = row.get("relevant_sections", [])
    ranks = M.relevant_ranks(payloads, labels)

    if row.get("should_abstain"):
        verdict = "ABSTAIN_OK" if not outcome.should_answer else "OVER_ANSWERED"
        hit = float(not outcome.should_answer)
    else:
        hit = M.hit_rate_at_k(ranks, k)
        verdict = "RETRIEVED_OK" if hit else "RETRIEVAL_FAIL"

    return {
        "id": row["id"],
        "category": row["category"],
        "question": row["question"],
        "verdict": verdict,
        f"hit@{k}": hit,
        # k=3 saturates on a corpus this small, and saturated metrics stop
        # discriminating. hit@1 is what catches "right section, wrong product" -
        # a motor NCB question answered from a health wording still scores 1.0
        # at k=3 because the correct clause scraped in at rank 3.
        "hit@1": M.hit_rate_at_k(ranks, 1) if not row.get("should_abstain") else hit,
        f"recall@{k}": M.recall_at_k(payloads, labels, k),
        "mrr": M.mrr(ranks),
        "best_rank": min(ranks) + 1 if ranks else None,
        "top_score": round(outcome.top_score, 4),
        "below_threshold": outcome.below_threshold,
        "broadened": outcome.broadened,
        "variants": outcome.variants,
        "reference_answer": row.get("reference_answer"),
        # The FULL reranked list, not just the top k. The interesting evidence is
        # usually just outside the cut - the chunk that holds the answer sitting
        # at rank 5 is what tells you the retriever found it and the reranker
        # buried it. Storing only the top k makes that unauditable after the run.
        "retrieved": [
            {
                "rank": i + 1,
                "score": round(h.score, 4),
                "section_path": h.hit.payload.get("section_path"),
                "doc_type": h.hit.payload.get("doc_type"),
                "product_name": h.hit.payload.get("product_name"),
                "effective_from": h.hit.payload.get("effective_from"),
                "kind": h.hit.payload.get("kind"),
                "retrieval_rank": h.retrieval_rank,
            }
            for i, h in enumerate(outcome.ranked)
        ],
        "timings_ms": outcome.timings_ms,
    }


def aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    answerable = [r for r in rows if r["verdict"] in ("RETRIEVED_OK", "RETRIEVAL_FAIL")]
    unanswerable = [r for r in rows if r["verdict"] in ("ABSTAIN_OK", "OVER_ANSWERED")]
    # None, not 0.0, for an empty bucket: "no unanswerable questions were asked"
    # and "the gate answered all of them" are opposite results, and averaging an
    # empty list into 0.0 renders them identically.
    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else None  # noqa: E731
    return {
        f"hit_rate@{k}": mean([r[f"hit@{k}"] for r in answerable]),
        "hit_rate@1": mean([r["hit@1"] for r in answerable]),
        f"recall@{k}": mean([r[f"recall@{k}"] for r in answerable]),
        "mrr": mean([r["mrr"] for r in answerable]),
        "abstention_accuracy": mean([r["verdict"] == "ABSTAIN_OK" for r in unanswerable]),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval_failures": [r["id"] for r in answerable if r["verdict"] == "RETRIEVAL_FAIL"],
        "wrong_at_rank_1": [r["id"] for r in answerable if not r["hit@1"]],
        # Answerable questions the gate would refuse to answer anyway. hit@k
        # scores rank position and is blind to the threshold, so raising the
        # threshold can buy abstention accuracy while silently introducing false
        # abstentions here without moving hit_rate@k at all.
        "false_abstentions": [r["id"] for r in answerable if r["below_threshold"]],
        "over_answered": [r["id"] for r in unanswerable if r["verdict"] == "OVER_ANSWERED"],
    }


def config_snapshot() -> dict[str, Any]:
    """What the number is a number *of*. A before/after with no config is noise."""
    return {
        "candidates": settings.reranker.candidates,
        "top_n": settings.reranker.top_n,
        "score_threshold": settings.reranker.score_threshold,
        "reranker_model": settings.reranker.model,
        "query_expansion_enabled": settings.retrieval.query_expansion_enabled,
        "query_expansion_variants": settings.retrieval.query_expansion_variants,
        "hyde_enabled": settings.retrieval.hyde_enabled,
        "bm25_enabled": settings.retrieval.retrieval_bm25_enabled,
        "dense_top_k": settings.retrieval.retrieval_dense_top_k,
        "sparse_top_k": settings.retrieval.retrieval_sparse_top_k,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", type=Path, default=GOLDEN)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--label", default="unlabelled", help="name for this run, e.g. 'baseline'")
    ap.add_argument("--out", type=Path, help="write the full per-question report here")
    ap.add_argument("--only", nargs="*", help="run only these question ids")
    args = ap.parse_args()

    rows = load(args.questions)
    if args.only:
        rows = [r for r in rows if r["id"] in set(args.only)]
    if not rows:
        raise SystemExit("no questions selected")

    pipeline = RetrievalPipeline()
    # Sequential on purpose: the embedder and cross-encoder are CPU-bound torch
    # calls, so concurrency here buys contention, not throughput.
    results = [await run_one(pipeline, row, args.k) for row in rows]
    summary = aggregate(results, args.k)

    print(f"\n{args.label}  ({len(results)} questions, k={args.k})")
    print(f"{'id':<5} {'verdict':<14} {'rank':>4} {'score':>7}  category")
    for r in results:
        print(
            f"{r['id']:<5} {r['verdict']:<14} {str(r['best_rank'] or '-'):>4} "
            f"{r['top_score']:>7.4f}  {r['category']}"
        )

    print()
    for key, value in summary.items():
        print(f"  {key:<24} {value}")

    report = {"label": args.label, "k": args.k, "config": config_snapshot(),
              "metrics": summary, "per_question": results}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
