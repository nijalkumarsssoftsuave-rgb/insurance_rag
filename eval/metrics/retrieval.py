"""Recall@k, MRR, nDCG, context precision and recall.

Only the three the Week-4 deliverable needs are implemented: hit-rate@k,
recall@k and MRR. nDCG and the context metrics need graded relevance labels,
and the golden set is binary. Add them when the labels earn it.

**Ground truth is labelled on payload fields, not ``chunk_id``.** Chunk ids are
database row UUIDs regenerated on every re-ingest, so a golden set keyed on them
would die the first time chunking or the embedding model changed - which is
exactly what an ablation does. Labels are payload predicates instead; see
``eval/golden/README.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

Payload = Mapping[str, Any]
Label = Mapping[str, Any]


def matches(payload: Payload, label: Label) -> bool:
    """One retrieved chunk against one ground-truth label.

    Every key in ``label`` must match. ``section_path`` matches by prefix, so a
    label on ``SECTION 5 - WAITING PERIODS`` is satisfied by the child chunk
    ``SECTION 5 - WAITING PERIODS > 5.1 Pre-existing Diseases``. Everything else
    is exact equality.
    """
    for key, want in label.items():
        got = payload.get(key)
        if key == "section_path":
            if not str(got or "").startswith(str(want)):
                return False
        elif got != want:
            return False
    return True


def relevant_ranks(payloads: Sequence[Payload], labels: Sequence[Label]) -> list[int]:
    """0-based ranks of the retrieved chunks that satisfy any label."""
    return [i for i, p in enumerate(payloads) if any(matches(p, lbl) for lbl in labels)]


def hit_rate_at_k(ranks: Sequence[int], k: int) -> float:
    """1.0 if at least one relevant chunk landed in the top k, else 0.0.

    The headline Week-4 number. Averaged over questions it reads as "how often
    did the right clause make it in front of the model at all".
    """
    return 1.0 if any(r < k for r in ranks) else 0.0


def recall_at_k(payloads: Sequence[Payload], labels: Sequence[Label], k: int) -> float:
    """Fraction of *labels* covered by the top k.

    Different from hit-rate on multi-hop questions: an answer that needs both the
    wording and the endorsement amending it scores 1.0 on hit-rate and 0.5 on
    recall when only one of the two is retrieved.
    """
    if not labels:
        return 0.0
    top = payloads[:k]
    found = sum(1 for lbl in labels if any(matches(p, lbl) for p in top))
    return found / len(labels)


def mrr(ranks: Sequence[int]) -> float:
    """Reciprocal of the best relevant rank - 1.0 at position 1, 0.5 at 2."""
    return 1.0 / (min(ranks) + 1) if ranks else 0.0
