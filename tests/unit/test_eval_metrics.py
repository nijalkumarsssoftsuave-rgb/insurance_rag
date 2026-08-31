"""The golden-set scorer - the one place a silent bug fakes a passing grade.

A matcher that is too permissive reports hit-rate 1.0 on a broken retriever, and
nothing downstream would notice. These tests exist to make that impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.harness import build_filter, load
from eval.metrics import retrieval as M

V32 = {"doc_type": "policy_wording", "effective_from": "2026-04-01"}
LABEL = V32 | {"section_path": "SECTION 4 - EXCLUSIONS > 4.11"}


def _chunk(section_path: str, **over) -> dict:
    return V32 | {"section_path": section_path, "product_name": "Family Health Optima"} | over


def test_section_path_matches_by_prefix() -> None:
    assert M.matches(_chunk("SECTION 4 - EXCLUSIONS > 4.11 Dental Treatment"), LABEL)


def test_wrong_section_does_not_match() -> None:
    assert not M.matches(_chunk("SECTION 4 - EXCLUSIONS > 4.12 Cosmetic"), LABEL)


def test_wrong_document_version_does_not_match() -> None:
    """The whole point of date-of-loss filtering: v2.8 must not satisfy a v3.2 label."""
    assert not M.matches(_chunk("SECTION 4 - EXCLUSIONS > 4.11 Dental", effective_from="2024-04-01"), LABEL)


def test_hit_rate_respects_k() -> None:
    payloads = [_chunk("SECTION 1 - DEFINITIONS"), _chunk("SECTION 2 - SCOPE OF COVER"),
                _chunk("SECTION 3 - BENEFITS"), _chunk("SECTION 4 - EXCLUSIONS > 4.11 Dental")]
    ranks = M.relevant_ranks(payloads, [LABEL])
    assert ranks == [3]
    assert M.hit_rate_at_k(ranks, 3) == 0.0  # rank 4 is outside the top 3
    assert M.hit_rate_at_k(ranks, 4) == 1.0
    assert M.mrr(ranks) == 0.25


def test_recall_counts_labels_not_chunks() -> None:
    """A multi-hop question that finds one of its two sources scores 0.5, not 1.0."""
    labels = [LABEL, {"doc_type": "endorsement", "section_path": "SECTION 2 - AMENDMENTS"}]
    payloads = [_chunk("SECTION 4 - EXCLUSIONS > 4.11 Dental")]
    assert M.recall_at_k(payloads, labels, 3) == 0.5
    assert M.hit_rate_at_k(M.relevant_ranks(payloads, labels), 3) == 1.0


def test_no_labels_scores_zero() -> None:
    assert M.relevant_ranks([_chunk("SECTION 1 - DEFINITIONS")], []) == []
    assert M.mrr([]) == 0.0


def test_golden_set_is_wellformed() -> None:
    """Every row parses, ids are unique, and every filter maps onto RetrievalFilter."""
    rows = load(Path("eval/golden/questions.jsonl"))
    assert len(rows) >= 20
    for row in rows:
        assert row["question"] and row["reference_answer"]
        build_filter(row.get("filters", {}))  # raises on an unknown filter key
        if row.get("should_abstain"):
            assert not row["relevant_sections"], f"{row['id']} is unanswerable but has labels"
        else:
            assert row["relevant_sections"], f"{row['id']} has no ground truth"


def test_committed_reports_agree_with_their_own_per_question_rows() -> None:
    """Guards the failure an aggregate cannot show: a headline number that does
    not follow from the rows underneath it. Both committed runs, not one."""
    # eval/runs/ is gitignored - run artefacts, not fixtures. Nothing to check on
    # a fresh clone; everything to check on the machine that produced the report.
    paths = sorted(Path("eval/runs").glob("*_deterministic.json"))
    for path in paths:
        report = json.loads(path.read_text(encoding="utf8"))
        k = report["k"]
        answerable = [r for r in report["per_question"] if r["verdict"].startswith("RETRIEV")]
        expected = sum(r[f"hit@{k}"] for r in answerable) / len(answerable)
        # The report rounds to 4dp; compare like for like.
        assert report["metrics"][f"hit_rate@{k}"] == round(expected, 4), path
        assert report["metrics"]["retrieval_failures"] == [
            r["id"] for r in answerable if not r[f"hit@{k}"]
        ], path
