# Week 4 — debugging retrieval: one change, measured

Golden set: `eval/golden/questions.jsonl`, 24 questions (21 answerable, 3 unanswerable),
labelled on payload predicates rather than chunk ids. Corpus: 166 chunks, 7 documents.
Metric: hit-rate@3 over the reranked list.

Runs referenced here: `eval/runs/{before,after}_deterministic.json`. `eval/runs/` is
gitignored - regenerate with the command below rather than expecting the JSON in git.

## Where everything lives

```
eval/
  RESULTS.md            this report - the Week-4 deliverable
  harness.py            runs the golden set, scores it, classifies each failure
  metrics/retrieval.py  hit-rate@k, recall@k, MRR, and the label matcher
  golden/
    README.md           the labelling contract - read before adding questions
    questions.jsonl     24 labelled questions (committed; the valuable artefact)
  runs/                 gitignored run output - regenerate, do not expect in git
    before_deterministic.json
    after_deterministic.json
  ablations.py          stub, unused this week
  report.py             stub, unused this week
  metrics/generation.py stub - faithfulness/citation accuracy, unused this week

app/retrieval/rerankers/bge_reranker.py   the one change (pair_text)
tests/unit/test_eval_metrics.py           metric + golden-set integrity tests
tests/unit/test_reranker_input.py         pair_text null-handling tests
```

## Read this first: the first measurement was not reproducible

The initial before/after was run with the pipeline as configured, which means query
expansion was on. Expansion calls an LLM to paraphrase each question
(`app/retrieval/expansion.py`), so **the paraphrase differs between runs and moves the
candidate pool underneath the reranker.** Re-running the identical code gave a different
answer:

| run | hit-rate@3 | retrieval failures |
|---|---|---|
| first "after" run | 1.0000 | `[]` |
| re-run, identical code | 0.9524 | `['q14']` |

q14 alternated between rank 2 and rank 4 depending on whether the paraphrase ended
"…resultant damage **indemnified**?" or "…damage **covered under the policy**?". A
before/after of ±0.048 measured through that stage is measuring the paraphrase.

Everything below is therefore measured with `QUERY_EXPANSION_ENABLED=false`. Both sides
were re-run from a fresh process and reproduced every aggregate exactly. One residual
source of variance survives and is worth knowing about: Qdrant's HNSW search is
approximate, so the *tail* of the candidate list can differ between runs. It showed up
once, on q17, where candidate 21-of-24 differed and the top rerank score moved
0.0013 → 0.0008. No rank, verdict or metric changed — the effect is confined to
questions with no signal, where every score is near zero. **This is itself a finding: the
eval harness is not trustworthy against a stochastic expansion stage, and any future
ablation has to disable it or average over repeats.**

Reproduce: `QUERY_EXPANSION_ENABLED=false python -m eval.harness -k 3 --label <name> --out eval/runs/<name>.json`

## The change

**Give the cross-encoder the section breadcrumb.** In
`app/retrieval/rerankers/bge_reranker.py`, `pairs = [(query, hit.text) ...]` became
`pairs = [(query, pair_text(hit)) ...]`, where `pair_text` prefixes
`"{product_name} | {section_path}\n"` onto the chunk text.

Ingestion already prefixes a breadcrumb onto `embedded_text` for the bi-encoder —
`_prefix()` in `app/ingestion/chunking/structure_aware.py` builds
`[insurer | product_name | breadcrumb]`, so this is the same idea in a different format,
not exact parity. The cross-encoder — the one model that reads query and chunk *together*
— was the only stage still scoring the bare `payload["text"]`.

## Before and after

| Metric | Before | After |
|---|---|---|
| **hit-rate@3** | **0.9524** | **1.0000** |
| recall@3 | 0.9286 | 1.0000 |
| MRR | 0.9381 | 0.9444 |
| hit-rate@1 | 0.9048 | 0.9048 |
| abstention accuracy | 0.3333 | 0.3333 |
| retrieval failures | `['q13']` | `[]` |
| wrong at rank 1 | `['q09','q13']` | `['q13','q14']` |
| false abstentions | `[]` | `[]` |

## Failure classification

`RETRIEVAL_FAIL` means the correct clause never entered the top 3 — no prompt change can
rescue it. `RETRIEVED_OK` means it was there, so any wrong answer from that point is a
generation failure. The classification is a column in the same run, not a separate pass.

The single retrieval failure was **q13**, "what no claim bonus do I get after three
claim-free years?". The correct chunk is the table holding
`| 3 consecutive claim-free years | 35% |`. Before, from `before_deterministic.json`
(`retr` = position in the hybrid-search result, 0-based):

```
ce-score  retr  kind           product                    section
 0.9801    0    child          Comprehensive Health Ins   17. No-Claim Bonus
 0.8213    2    child          Motor Shield Private Car   1.2 No Claim Bonus
 0.3732    5    child          Motor Shield Private Car   SECTION 4 - NO CLAIM BONUS
 0.2971    4    table_summary  Motor Shield Private Car   SECTION 4 - NO CLAIM BONUS
 0.0787    1    table          Motor Shield Private Car   SECTION 4 - NO CLAIM BONUS  <- the answer
```

Hybrid search placed that table **second** out of 12 candidates. The cross-encoder scored
it 0.0787 and dropped it to fifth — outside k=3 — while ranking a *health* policy's NCB
prose first at 0.9801 on a *motor* question. A table chunk is pipe-markdown with no prose
in it, so without its heading there was nothing to match the question against.

After: the same table scores **0.8748 at rank 2**, and hit-rate@3 reaches 1.0000.

## How much of this is one question?

Nearly all of it. The change was chosen after inspecting q13, and q13 is the whole of the
hit-rate@3 movement (20/21 → 21/21). hit-rate@1 does not move at all. MRR moves +0.0063.
Only recall@3 has a second question behind it (q10, 0.5 → 1.0).

A better test than sample size is whether the mechanism predicts anything out of sample.
It claims a chunk with no breadcrumb saturates on questions it cannot answer, which
predicts both unanswerable questions should drop. **q24 dropped 0.9985 → 0.0267 as
predicted; q16 moved the wrong way, 0.0939 → 0.1116.** One for two.

Treat this as a fix for one question with a plausible mechanism, not a validated retrieval
improvement.

## What the change did NOT fix

**1. Abstention. Unchanged at 0.3333 — two of three unanswerable questions still get
answered.** The scores moved a long way without fixing the problem:

```
before   answerable min 0.4194 (q14)   unanswerable max 0.9985 (q24)   bands overlap
after    answerable min 0.0118 (q14)   unanswerable max 0.1116 (q16)   bands overlap
```

In **both** configurations no value of `score_threshold` separates the two classes. Before,
q24 outscored every answerable question. After, q16 at 0.1116 outscores three answerable
questions (q14 0.0118, q08 0.0185, q06 0.0288). `RERANKER_SCORE_THRESHOLD=0.01` is still
uncalibrated and still cannot be calibrated on this evidence.

**2. Rank-1 precision, and q14 got quietly worse.** q09 gained rank 1; q14 lost it:

```
before   ce      product                   section
 rank1   0.4194  Motor Shield Private Car  SECTION 7 - EXCLUSIONS > 7.3 Driving Under the Influence
 rank2   0.0371  Comprehensive Health Ins  9. Emergency Hospitalization

after
 rank1   0.0118  Comprehensive Health Ins  12. Policy Exclusions
 rank2   0.0100  Comprehensive Health Ins  9. Emergency Hospitalization
 rank3   0.0094  Motor Shield Private Car  SECTION 7 - EXCLUSIONS > 7.3 Driving Under the Influence
```

q14 still counts as `RETRIEVED_OK` at k=3, so no metric in the table registers this. Three
costs it hides:

* The `top_score` column reads 0.0118, but that is the **wrong** chunk's score. The correct
  clause scores 0.0094, and it appears nowhere in the summary.
* `should_answer` compares `top_score < 0.01`. Before, the margin was **42× the threshold**;
  after, it is **1.2×**. Had the correct clause held rank 1 at 0.0094, the gate would have
  fired and triggered the broadening retry on a question with one unambiguous answer.
* `packing.pack` orders context "weakest-first so the best lands last", so the slot nearest
  the question — the one generation attends to most — now holds the wrong product's clause.

q08 (0.9850 → 0.0185), q06 (0.8078 → 0.0288) and q12 (0.4828 → 0.2030) kept their rank but
lost most of their score. The breadcrumb improves *ordering* and compresses *scores* toward
an uncalibrated gate. Ranking is the half being measured; the gate is the half that was
already broken, and this pushes it closer to the edge.

**3. Cross-product confusion.** q13 now passes at k=3, but rank 1 is still a health
policy's NCB section answering a motor question. Nothing filters by line of business when
the question does not name a product.

## Caveats

- 24 questions, 3 of them unanswerable. The abstention figure rests on those 3 — directional,
  not calibrated. The golden README targets 250 questions including 20 `adversarial_injection`
  and 20 `claim_status`; neither category exists yet.
- The corpus contains a stray document — `Comprehensive Health Insurance`, a synthetic
  personal policy schedule numbered `1. Policy Overview` rather than the `SECTION n` scheme
  every other document uses, with `insurer: null`. It is not one of the six PDFs in `pdf/`,
  and it takes rank 1 on q13, q14, q16 and q24 — every failure in this report. Deciding
  whether it belongs in the tenant is probably worth more than any reranker tuning.
- `hit_rate@3` is now saturated at 1.0000 and cannot measure the next change. Use
  hit-rate@1, MRR and abstention accuracy from here.
