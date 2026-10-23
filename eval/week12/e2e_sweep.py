"""Week 12 end-to-end sweep: every lane, through the real HTTP API - the
single entry point a stranger would actually use after `make bootstrap`,
`make api`.

    python eval/week12/e2e_sweep.py [--base http://localhost:8010/api/v1]

Writes eval/week12/e2e_results.json. No assertions - this is the live
reproduction behind the "end-to-end sweep" table in RESULTS.md, not a test
(tests/ never makes a real LLM call; this script does, on purpose).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8010/api/v1")
    args = ap.parse_args()

    results = []
    with httpx.Client(timeout=90.0) as c:
        r = c.post(
            f"{args.base}/chat",
            json={"message": "Is dental treatment covered under Family Health Optima?"},
        )
        d = r.json()
        results.append(
            {
                "check": "lane_a_coverage",
                "status": r.status_code,
                "intent": d.get("intent"),
                "verified": d.get("verified"),
                "answer": d.get("answer", "")[:150],
            }
        )

        r = c.post(
            f"{args.base}/chat", json={"message": "What is the status of claim CLM-2026-0004?"}
        )
        d = r.json()
        results.append(
            {
                "check": "lane_b_claim_status",
                "status": r.status_code,
                "intent": d.get("intent"),
                "answer": d.get("answer", "")[:150],
            }
        )

        r = c.post(
            f"{args.base}/chat", json={"message": "What is the status of claim CLM-9999-9999?"}
        )
        d = r.json()
        results.append(
            {
                "check": "lane_b_not_found",
                "status": r.status_code,
                "answer": d.get("answer", "")[:150],
            }
        )

        t0 = time.perf_counter()
        r = c.post(
            f"{args.base}/chat",
            json={"message": "Is dental treatment covered under Family Health Optima?"},
        )
        ms = int((time.perf_counter() - t0) * 1000)
        results.append(
            {
                "check": "cache_hit_repeat",
                "status": r.status_code,
                "ms": ms,
                "answer": r.json().get("answer", "")[:150],
            }
        )

        r = c.post(f"{args.base}/chat", json={"message": "Hello"})
        cid = r.json()["conversation_id"]
        r2 = c.get(f"{args.base}/chat/{cid}/history")
        results.append({"check": "history", "status": r2.status_code, "n_messages": len(r2.json())})

    out = Path(__file__).resolve().parent / "e2e_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
