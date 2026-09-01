"""One command: score every claim summary, by mode, and check the judge against the labels.

    python -m eval.week6_eval                      # assertions + judge v1 + agreement
    python -m eval.week6_eval --judge eval/week6/judge_v2.txt
    python -m eval.week6_eval --no-judge           # assertions only, free and offline

Two scoring layers, deliberately separate:

* **Assertions** (`eval/assertions.py`) decide everything a rule can decide. Free,
  instant, deterministic.
* **One judged criterion** - groundedness - for the part no rule can reach.

They are reported separately because collapsing them hides which half moved. Pass
rate is printed **per mode**, never as a single number: an average will happily
hide a regression on the denial-citation mode while the clean cases carry it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel, Field

from app.llm import get_llm, system, user
from eval.assertions import ASSERTIONS, run_all

WEEK6 = pathlib.Path("eval/week6")
SUMMARIES = WEEK6 / "summaries.json"
LABELS = WEEK6 / "labels_25.json"


class Verdict(BaseModel):
    verdict: str = Field(description="SAFE or UNSAFE")
    why: str = Field(default="", description="one sentence")


def render_case(item: dict[str, Any]) -> str:
    record = {k: v for k, v in item["record"].items() if k != "notes"}
    notes = item["record"].get("notes") or []
    lines = [
        "<claim_record>",
        json.dumps(record, indent=2, sort_keys=True, default=str),
        "</claim_record>",
        "<adjuster_notes>",
    ]
    lines += [f"- [{n.get('at', '')}] {n.get('note', '')}" for n in notes] or ["(none)"]
    lines += ["</adjuster_notes>", "<summary>", item["summary"], "</summary>"]
    return "\n".join(lines)


async def judge_all(items: list[dict], prompt_path: pathlib.Path) -> dict[str, Verdict]:
    # Comment lines carry the provenance and the assertion/judge split; they are
    # documentation for the reader, not instructions for the model.
    body = "\n".join(
        line
        for line in prompt_path.read_text(encoding="utf8").splitlines()
        if not line.startswith("#")
    ).strip()
    llm = get_llm()
    out: dict[str, Verdict] = {}
    for item in items:
        try:
            out[item["case_id"]] = await llm.structured(
                [system(body), user(render_case(item))], Verdict, temperature=0.0
            )
        except Exception as exc:  # a judge outage must not look like an UNSAFE verdict
            out[item["case_id"]] = Verdict(verdict="ERROR", why=str(exc)[:120])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judge", type=pathlib.Path, default=WEEK6 / "judge_v1.txt")
    ap.add_argument("--no-judge", action="store_true", help="assertions only")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    items = json.loads(SUMMARIES.read_text(encoding="utf8"))
    for item in items:
        item["assertions"] = {
            name: {"passed": ok, "detail": detail}
            for name, (ok, detail) in run_all(item["summary"], item["record"]).items()
        }

    # ── assertions, per mode ─────────────────────────────────────────────
    per_mode: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        per_mode[item["mode"]].append(item)

    print(f"\n{len(items)} cases · {len(ASSERTIONS)} assertions · 1 judged criterion")
    print(f"\n{'mode':<34}{'n':>3}{'assert pass':>13}{'failing cases':>34}")
    print("-" * 84)
    for mode in sorted(per_mode):
        rows = per_mode[mode]
        bad = [r["case_id"] for r in rows if not all(a["passed"] for a in r["assertions"].values())]
        score = f"{len(rows) - len(bad)}/{len(rows)}"
        print(f"{mode:<34}{len(rows):>3}{score:>13}{(', '.join(bad) or '-'):>34}")
    all_bad = [
        r["case_id"] for r in items if not all(a["passed"] for a in r["assertions"].values())
    ]
    print("-" * 84)
    print(f"{'ALL':<34}{len(items):>3}{f'{len(items) - len(all_bad)}/{len(items)}':>13}")

    by_check = Counter(
        name for r in items for name, a in r["assertions"].items() if not a["passed"]
    )
    if by_check:
        print("\nassertion failures by check:")
        for name, n in by_check.most_common():
            print(f"  {name:<24} {n}")

    report: dict[str, Any] = {
        "n_cases": len(items),
        "n_assertions": len(ASSERTIONS),
        "n_judged_criteria": 1,
        "assertion_failures": all_bad,
    }

    # ── judge + agreement ────────────────────────────────────────────────
    if not args.no_judge:
        labels = {
            r["case_id"]: r["label"]
            for r in json.loads(LABELS.read_text(encoding="utf8"))["labels"]
        }
        verdicts = asyncio.run(judge_all(items, args.judge))

        agree = [c for c in labels if verdicts[c].verdict == labels[c]]
        disagree = [c for c in labels if verdicts[c].verdict != labels[c]]
        unsafe_labels = [c for c in labels if labels[c] == "UNSAFE"]
        caught = [c for c in unsafe_labels if verdicts[c].verdict == "UNSAFE"]

        print(f"\njudge: {args.judge}")
        pct = len(agree) / len(labels)
        print(f"{'agreement with hand labels':<34}{len(agree)}/{len(labels)} = {pct:.1%}")
        # The labels are 25 SAFE to 1 UNSAFE, so a judge that answered SAFE every
        # time would score 96%. Overall agreement alone cannot tell that apart
        # from a judge that works; recall on the minority class can.
        if unsafe_labels:
            print(f"{'  of which UNSAFE caught':<34}{len(caught)}/{len(unsafe_labels)}")
        print(
            f"{'  always-SAFE baseline':<34}"
            f"{sum(1 for c in labels if labels[c] == 'SAFE') / len(labels):.1%}"
        )
        if disagree:
            print("\ndisagreements:")
            for c in disagree:
                v = verdicts[c]
                print(f"  {c}  human={labels[c]:<7} judge={v.verdict:<7} :: {v.why[:84]}")

        report |= {
            "judge": str(args.judge),
            "agreement": round(len(agree) / len(labels), 4),
            "unsafe_recall": f"{len(caught)}/{len(unsafe_labels)}" if unsafe_labels else None,
            "disagreements": [
                {
                    "case_id": c,
                    "human": labels[c],
                    "judge": verdicts[c].verdict,
                    "why": verdicts[c].why,
                }
                for c in disagree
            ],
            "verdicts": {c: v.model_dump() for c, v in verdicts.items()},
        }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
