# Week 6 · Task Set D — validating the claim-summary judge

One command: `python -m eval.week6_eval` · 26 cases · **4 assertions / 1 judged criterion**

| | |
|---|---|
| **agreement_before** (judge v1) | **21/26 = 80.8%** |
| **agreement_after** (judge v2) | **20/26 = 76.9%** |
| UNSAFE recall | **0/2 → 1/2** |
| always-SAFE baseline | **92.3%** |
| assertions / judged criteria | 4 / 1 |

**The iteration made agreement worse, and both judge versions score below a judge
that answers SAFE every time.** That is the result, not a mishap in reporting it.
The reason is diagnosed below; the number that did move is UNSAFE recall.

## The feature had to be built first

The brief assumes *"the quality score for every claim summary your app writes from
adjuster notes"*. None of it existed. `service.render_status` is a deterministic
customer-facing template — nothing generated, nothing to judge. So this week starts
with `app/claims/summary.py` + `app/prompts/templates/claim_summary.md`: an
adjuster-facing handover summary written from the claim record and
`ClaimEvent.note`, which is where the adjuster notes actually live.

## Assertion / judge split — 4 assertions, 1 judged criterion

Moved **out** of the judge and into `eval/assertions.py`, and deleted from the
judge prompt (the deletion is listed at the top of `judge_v1.txt`):

| assertion | what a regex decides for free |
|---|---|
| `claim_number_echoed` | the number appears in canonical form **and is this claim's** |
| `date_of_loss_present` | stated and parses to the record's date, in any rendering |
| `amounts_numeric` | money is figures, and every figure came from the record |
| `denial_cites_clause` | a stated denial carries a clause reference |

The judge is left with exactly one binary criterion: *is every material claim
supported, with nothing invented, contradicted, or decided that the file leaves
open?* Binary on purpose — a 1–10 score with within-1 tolerance would inflate
agreement into meaninglessness.

**Format deviation, recorded not hidden.** The brief specifies `CLM-YYYY-NNNNN`
(5 digits). Live data is `CLM-2026-0001` — 4. The assertion matches what the system
issues; asserting the brief's shape would fail every real claim.

### Three bugs found in my own assertions, by running them against real output first

1. `amounts_numeric` only looked *right* of the currency marker, so it failed every
   summary written as "62,800.00 INR" — it fired on correct output.
2. The contradiction check matched a figure within 40 characters of "claim" and read
   the **2026 out of CLM-2026-0001** as the claimed amount, failing all four real summaries.
3. `[\d,]+` matched a bare comma, so "…41,200.00 INR, and the approved…" parsed the
   comma after INR as an amount.

All three failed on *correct* output — the dangerous direction, because an assertion
that cries wolf gets ignored. Pinned now by `tests/unit/test_assertions.py`.

## Blind protocol and the ordering proof

The brief accepts *"commit or timestamp"*. Timestamps and content hashes are used, so
nothing depends on git history:

```
14:48:18  e7124a2c9226a80c  cases.jsonl        26 cases built
14:49:42  b9f2ff707f99f9f8  summaries.json     26 summaries generated
15:29:57  80bf7d41f2e8c6b1  labels_25.json     26 hand labels FROZEN  <- no judge on disk
15:30:14  dbd50b866f3733be  judge_v1.txt       authored after the labels
15:31:01  25994b3d6f13c8b4  run_v1.json        agreement_before
15:31:25  8deb2ba5c6a1bf19  prediction.txt     filed before v2 existed
15:31:51  05e858f33e5264b2  judge_v2.txt       two disagreements added
15:32:37  3abffccf5ec396fd  run_v2.json        agreement_after
```

`labels_25.json` pins `summaries_sha256 b9f2ff70…`, so labels and the summaries they
were written against cannot be swapped after the fact.

**Two disclosures that belong here rather than in a footnote.**

*Concurrency.* A second session ran this task in parallel and wrote its own judge chain
into this directory. Its `cases.jsonl`, `summaries.json` and `labels_25.json` were
overwritten by this session's, which broke the hash its report pinned. Its chain could
not be recovered, so the whole protocol was **re-run from the labels forward**: the
stale judge files were removed, and no judge prompt existed on disk when these labels
were frozen or at any time after. The backup of the discarded chain is outside the repo.

*Vocabulary.* Labels were first recorded as PASS/FAIL and rewritten to SAFE/UNSAFE to
match the judge's output. One-to-one, no verdict changed; every `why` is the text
written at labelling time.

**Stated limitation.** The labeller and the judge author are the same process. This is
one rater, not independent human validation, and is not dressed up as such.

## Eval set — 26 cases, tagged by Week-5 mode

| mode | n | assertions pass |
|---|---|---|
| `clean` | 11 | 11/11 |
| `W5-M6-wrong-source` | 4 | 4/4 |
| `W5-M5-contradicts-record` | 4 | 4/4 |
| `W5-M3-hedges-instead-of-stating` | 4 | 4/4 |
| `W5-M1-uncitable-clause` | 3 | **2/3** (c08) |
| **all** | **26** | **25/26** |

Four of the six Week-5 modes transfer to claim summaries; *turn latency* and *refuses
on identical repeat* do not, and were not forced. **6 of 26 cases are real** — the four
live claims from the database, plus two regression cases traced to real claim-lane
turns (`CLM-99999` not-found, and `CLM-2026-0004`, the rejected claim behind the Week-5
dental hand-off).

## Before → after, and where the prediction was wrong

`prediction.txt` was filed after reading v1's disagreements and **before** v2 existed.
Scored honestly:

| prediction | outcome |
|---|---|
| agreement 80.8% → **≥92%** | **WRONG** — it fell to **76.9%** |
| UNSAFE recall 0/2 → **1/2** | **RIGHT** — c08 fixed by its own example |
| c18 **and** c19 both flip to SAFE | **HALF RIGHT** — c18 flipped; c19 still disagrees, on a new complaint |
| c16 will **not** be fixed | **RIGHT** — still SAFE to the judge |

**Why the headline number fell.** Example A taught the judge that calling a reason code
a "policy clause" is UNSAFE. It over-generalised to *any* mention of a clause: v2 now
flags **c06 (clause 4.11), c07 (5.1) and c15 (7.2)** — all three of which cite a clause
that **is** on the record. The example taught a keyword, not the distinction. That is
the concrete failure mode of few-shot judge iteration, and it cost four new false alarms
to buy one real catch.

The prediction was right about the mechanism and wrong about the arithmetic, which is
the useful way round: the part I said could be wrong (c19 generalising) is exactly the
part that broke.

## Two disagreements, and who was right

**c08 — v1 said SAFE, I said UNSAFE. I was right.** The summary reads "rejected due to
the DUI_EXCLUSION policy clause". `DUI_EXCLUSION` is a `rejection_reason_code`; the
record carries **no clause reference at all**. Every token came from the record, so the
judge's "every material claim is supported" was true term by term and wrong as a whole —
an adjuster would go looking for a clause that does not exist. The deterministic
`denial_cites_clause` assertion caught this independently, which is the argument for the
assertion/judge split in one case.

**c06 — v2 said UNSAFE, I said SAFE. The judge was wrong.** The summary cites clause
4.11 on a claim whose record carries `rejection_clause_ref: 4.11`. v2 flagged it purely
for resembling Example A. A judge that cannot tell a correct clause citation from a
fabricated one is worse than no judge on exactly the cases that matter most — denials.

**c13 is the honest borderline.** v2 says the summary "resolves a disagreement between
the assessors"; it actually ends *"The next step is to resolve the disagreement between
the assessors before proceeding."* — naming the disagreement, not settling it. I read
that as SAFE and still do, but the judge's misreading is a fair one and I would not
call this a clean win for either side.

## What this measurement is worth

Both judge versions score **below the 92.3% always-SAFE baseline**, so neither is fit to
route work on. Overall agreement is the wrong headline on a 24-to-2 label set: it is
dominated by the majority class and would rank a judge that never says UNSAFE above both
of these. UNSAFE recall — **0/2 → 1/2** — is the only figure here that moved in the
direction claims ops actually needs, and it is measured on two cases.

The honest next step is more UNSAFE cases before any further prompt work. A minority
class of two cannot validate a judge, and iterating a prompt against it optimises noise.
