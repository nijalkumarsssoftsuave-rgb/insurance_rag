# Week 4 · Task Set D — label the failures, then buy back hit-rate@3 with one change

Golden set: `eval/golden/questions.jsonl` — **30 questions** (27 answerable, 3 unanswerable),
of which **6 carry an exact token** dense retrieval is supposed to be structurally bad at
(rejection code, UIN, endorsement number, document reference, reference-number format).
The brief asks for 12; this set is a superset, and **the same set is used for every run**.

Corpus: **139 chunks, 6 documents** — the six PDFs in `pdf/`. Metric: hit-rate@3 over the
reranked list.

Runs: `eval/runs/{v0_baseline,v1_rerank_top25,v2_breadcrumb}.json` (gitignored — regenerate).

## A contaminating document was removed before these runs

The corpus previously held a seventh document, `Untitled document.pdf` ("Comprehensive
Health Insurance") — a synthetic personal policy schedule, uploaded manually on 2026-08-14,
numbered `1. Policy Overview` rather than the `SECTION n` scheme every other document uses,
with `insurer: null`. It was not one of the six PDFs in `pdf/`.

It took rank 1 on **every failure** in an earlier version of this report, and it caused a
live generation bug. Asked *"what no claim bonus do I get after three claim-free years?"* the
system answered:

> "a No-Claim Bonus of **35% of the original Sum Insured**"

35% came from the **motor** policy's NCB table (`Discount on own damage premium`);
"of the original Sum Insured" came from the **health** document's NCB clause sitting in the
same context. The model welded two products together. `verified: true`, `confidence: 1.0` —
the groundedness check passed it.

Measured on unmodified code, changing only which documents were visible:

| | merge bug |
|---|---|
| with `Untitled document.pdf` | **5/5** |
| without it | **0/5** |

It was deleted via `DELETE /api/v1/documents/6ca6d804-…` (7 docs / 166 points → 6 docs /
139 points). The raw upload is retained in the object store by design, so this is
reversible. All 30 golden rows still resolve: **32 labels, 0 unmatched.**

A code fix was attempted first and **rejected on measurement** — see "What I tried and threw
away" at the end.

## Reproducibility — read before trusting any number here

Query expansion calls an LLM to paraphrase each question, so its output differs run to run
and moves the candidate pool underneath the reranker. Measured through that stage, identical
code gave hit-rate@3 of 1.0000 on one run and 0.9524 on the next, with a *different* question
failing each time. That is the paraphrase moving, not the retriever.

**Every run below sets `QUERY_EXPANSION_ENABLED=false`.** Both sides were re-run from fresh
processes and reproduced every aggregate exactly. One residual: Qdrant's HNSW is approximate,
so the *tail* of the candidate list can shift; it moved one near-zero score and changed no
rank, verdict or metric.

```bash
QUERY_EXPANSION_ENABLED=false python -m eval.harness -k 3 --label v0 --out eval/runs/v0_baseline.json
```

## Ground truth is labelled on payload fields, not chunk_id

The brief asks for a `chunk_id` per question. Here `chunk_id` is a **database row UUID,
regenerated on every re-ingest** — a golden set keyed on it dies the first time chunking or
the embedding model changes, which is exactly what an ablation does. Each question instead
carries `relevant_sections`: payload predicates (`doc_type`, `product_name`,
`effective_from`, `section_path` matched by prefix, `kind`).

The questions are authored against the synthetic corpus rather than drawn from real adjuster
tickets — there are none for a generated corpus. That is a real weakness: the set cannot
contain a failure mode nobody thought to write down.

## Baseline (v0), written down before any change

| | |
|---|---|
| **hit-rate@3** | **0.9630** (26/27) |
| hit-rate@1 | 0.9259 |
| recall@3 | 0.9444 |
| MRR | 0.9537 |
| abstention accuracy | 0.3333 |
| p50 latency | 2114 ms |

## Failure tally: R / G / Not-In-Corpus

| Question | Label | Evidence |
|---|---|---|
| q13 "no claim bonus after three claim-free years" | **R** | The correct chunk (NCB table, `3 consecutive claim-free years \| 35%`) is pipe-markdown with no prose. The cross-encoder scored it 0.0787 and dropped it to rank 4 — outside k=3 — while ranking the motor policy's NCB *definition* higher. |
| q16 "surgery for my pet dog" | **Not-In-Corpus** | Regex `pet\|dog\|cat\|veterinar\|animal\|livestock` over all 139 chunks: **0 matches**. Answered anyway — top score 0.0939 against `score_threshold = 0.01`. |
| q24 "treatment taken outside India" | **Not-In-Corpus** | Regex `outside India\|overseas\|abroad\|worldwide\|geograph\|territor` over all 139 chunks: **0 matches**. Answered anyway at 0.9959. |

**Tally (retrieval harness): R = 1 · G = 0 · Not-In-Corpus = 2.** The G column reads zero only
because this harness cannot observe generation — see below.

### All 6 exact-token questions passed at rank 1

| Question | rank | ce | rank-1 chunk |
|---|---|---|---|
| q25 `PED_WAITING_PERIOD` | 1 | 0.8990 | sop · SECTION 2 - REJECTION REASON CODES |
| q26 `ACMEHLIP26032V032627` | 1 | 0.3489 | Family Health Optima · Policy Wording title |
| q27 `ACMEMOTP26014V011627` | 1 | 0.2782 | Motor Shield · Policy Wording title |
| q28 `PA-YYYY-NNNNNN` | 1 | 0.3032 | sop · 1.1 Receipt and Acknowledgement |
| q29 `AGI/SOP/CLAIMS/2026/04` | 1 | 0.9983 | sop · SOP title |
| q30 `AGI/END/2026/1187` | 1 | 0.1039 | endorsement · Endorsement title |

The premise behind the brief — *dense retrieval is structurally bad at exact tokens* — does
not bite on this system, because **the retriever is not dense**. bge-m3 emits a learned
sparse vector alongside the dense one, and Qdrant fuses both branches by RRF server-side.
There is already a lexical branch, and it resolves exact codes at rank 1.

**This is what rules out BM25.** Zero exact-token failures means a BM25 branch has nothing to
fix here, and it would cost a full re-ingest (`app/ingestion/pipeline.py` writes only the
learned-sparse vector). Choosing it would mean choosing a change the evidence does not support.

### G failures: the harness cannot see them, and live checks found two

The harness calls `RetrievalPipeline.retrieve` directly and never generates, so "model
misused good context" is *structurally unobservable* in every number above. Two were found by
querying the live API:

1. **The NCB merge bug** described at the top — now fixed by removing the contaminating
   document. Verified 4/4 clean afterwards.
2. **Caesarean sub-limit, still open.** Asked *"what is the maternity limit for a caesarean
   section?"* the system answers **Rs. 75,000**. The endorsement table reads
   `| Maternity - caesarean section | Rs. 75,000 | Rs. 1,00,000 |` under headers
   `Benefit | Existing sub-limit | Revised sub-limit`. The correct chunk is retrieved and
   cited; the model reads the **Existing** column instead of the **Revised** one. Right
   chunk, wrong column — a G failure no retrieval change can touch.

So the true tally is **R = 1 · G ≥ 2 · Not-In-Corpus = 2**, and G is a floor, not a count.
Measuring it properly needs the harness extended to run generation and grade against
`reference_answer` — the single most valuable thing to build next.

## Two candidate changes, each measured against the same v0 baseline

The tally says the one R failure is a **cross-encoder scoring** problem, not a candidate
recall problem: the right chunk was already in the pool and the reranker buried it. Two
candidates, measured separately against v0 — never stacked.

| | change | hit@3 | hit@1 | MRR | p50 | p50 rerank |
|---|---|---|---|---|---|---|
| **v0** | baseline: rerank top 12, bare chunk text | 0.9630 | 0.9259 | 0.9537 | 2114 ms | 1866 ms |
| **v1** | cross-encoder rerank over the **top 25** | 0.9630 | 0.9259 | 0.9537 | 4276 ms | 3975 ms |
| **v2** | **breadcrumb prefix on the cross-encoder input** | **1.0000** | **1.0000** | **1.0000** | 2299 ms | 2060 ms |

**v1 — rerank over the top 25: +2162 ms p50 (+102%) for zero gain.** Every metric identical
to baseline, recall@3 slightly *worse* (0.9444 → 0.9259). It fails for the reason the tally
predicted: q13's correct chunk was never missing from the candidate pool, so widening the
pool cannot help. It only hands the cross-encoder more chunks to mis-score, and cross-encoder
latency is linear in candidate count.

**v2 — the shipped change: +186 ms p50 (+8.8%) for a perfect ranking.** Every answerable
question's correct chunk at rank 1. One line in `app/retrieval/rerankers/bge_reranker.py`:

```python
- pairs = [(query, hit.text) for hit in hits]
+ pairs = [(query, pair_text(hit)) for hit in hits]
```

`pair_text` prefixes `"{product_name} | {section_path}"` onto the chunk. Ingestion already
does this for the bi-encoder (`_prefix()` in `structure_aware.py`); the cross-encoder — the
one model that reads query and chunk *together* — was the only stage still scoring bare
`payload["text"]`. A table chunk is pipe-markdown with no prose, so without its heading there
was nothing to match a question against.

## Per-question record: v0 → v2

| id | before | rank | after | rank | outcome |
|---|---|---|---|---|---|
| q01 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q02 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q03 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q04 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q05 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q06 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q07 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q08 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q09 | RETRIEVED_OK | 2 | RETRIEVED_OK | 1 | pass, rank improved |
| q10 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q11 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q12 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q13 | RETRIEVAL_FAIL | 4 | RETRIEVED_OK | 1 | **FIXED** |
| q14 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | **pass, but now gated** |
| q15 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q16 | OVER_ANSWERED | - | OVER_ANSWERED | - | still broken |
| q17 | ABSTAIN_OK | - | ABSTAIN_OK | - | unchanged |
| q18 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q19 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q20 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q21 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q22 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q23 | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q24 | OVER_ANSWERED | - | OVER_ANSWERED | - | still broken |
| q25 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q26 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q27 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q28 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q29 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |
| q30 ·EXACT-TOKEN | RETRIEVED_OK | 1 | RETRIEVED_OK | 1 | unchanged |

**Fixed: q13** — the only R failure, rank 4 → rank 1.
**Improved: q09** — rank 2 → rank 1.
**Untouched: q16, q24** — both Not-In-Corpus, both still answered. No retrieval change can
fix these; they need the abstention gate.

**New cost — q14 is now a false abstention.** Its correct clause
(`SECTION 7 - EXCLUSIONS > 7.3 Driving Under the Influence`) sits at **rank 1**, but scores
**0.0094** against `score_threshold = 0.01`. The gate would refuse to answer a question whose
correct clause the retriever put first. `hit@3` and `hit@1` both score this row 1.0; only the
`false_abstentions` column catches it.

## Cost: p50 latency

| | v0 | v2 | delta |
|---|---|---|---|
| **p50 total per query** | **2114 ms** | **2299 ms** | **+186 ms (+8.8%)** |
| p50 rerank stage | 1866 ms | 2060 ms | +194 ms |

The cost lands entirely in the rerank stage, which is what the change touches: the breadcrumb
adds ~10-15 tokens to every query-chunk pair the cross-encoder scores.

Mean and p95 are not reported. Each run pays a one-time model load on its first question
(~32 s against ~2.4 s), which corrupts the mean and would make v2 look like a *speed-up*.
The median is robust to it.

## Shipping decision

**Ship v2. Do not ship v1. Calibrate `score_threshold` in the same release.**

* **v1 is a clear reject**: +2162 ms p50 (+102%) for 0.0000 improvement on every metric. This
  is the "not worth the latency" case, and the tally called it in advance.
* **v2 is a clear ship**: +186 ms on a 2114 ms pipeline (+8.8%) for hit-rate@3 0.9630 → 1.0000
  and hit-rate@1 0.9259 → 1.0000.
* **The threshold must move with it.** v2 pushes q14's rank-1 score to 0.0094, just under the
  0.01 gate. Shipping v2 alone converts a working answer into an abstention.

The threshold cannot yet be calibrated, and the harness says so: in **all three** variants the
answerable and unanswerable score bands overlap.

```
v0/v1   answerable min 0.1039 (q30)   unanswerable max 0.9959 (q24)   overlap
v2      answerable min 0.0094 (q14)   unanswerable max 0.0374 (q16)   overlap
```

No single value of `score_threshold` separates "can answer" from "cannot answer" on this
corpus. That is the honest state of the abstention gate, and it owns 2 of the 3 remaining
failures.

## Bonus: MMR over the candidate list

`mmr()` added to `app/retrieval/fusion.py`, applied over the reranked list, lambda swept once.
Diversity measured two ways: mean pairwise cosine among the top 3 (lower = more diverse), and
distinct `(doc_id, section_path)` pairs in the top 3.

| lambda | hit@3 | mean pairwise cos (top-3) | distinct sections in top-3 |
|---|---|---|---|
| **1.0** (pure relevance) | 1.0000 | 0.6269 | 2.89 |
| 0.9 | 1.0000 | 0.5919 | 2.93 |
| 0.8 | 1.0000 | 0.5895 | 2.93 |
| 0.7 | 1.0000 | 0.5832 | 2.93 |
| 0.5 | 1.0000 | 0.5738 | 2.93 |
| 0.3 | 1.0000 | 0.5631 | 2.93 |

**Would not ship — but the reason changed.** On the contaminated corpus MMR *cost* a question:
at lambda 0.9 it pushed q14's drunk-driving exclusion from rank 3 to rank 5 to make room for
three diverse but irrelevant chunks. With the stray document gone that cost disappears —
hit@3 stays 1.0000 at every lambda.

What remains is that MMR buys nothing measurable here: distinct sections move 2.89 → 2.93 out
of 3, because the reranked top-3 is already almost entirely distinct. It adds a parameter to
tune and a failure mode (in insurance the near-duplicates it demotes are often the same clause
across form editions, and only one edition is in force on the date of loss) in exchange for
0.04 of a section. `mmr()` is committed but **not wired into the default retrieval path**.

## What I tried and threw away

Before removing the document, I tried fixing the NCB merge bug in code: add a `product`
attribute to each `<document>` block so the model can tell two products apart, plus prompt
rules forbidding cross-product statements. It fixed NCB. Then I measured the rest of the
system:

| "is dental treatment covered?" | answered |
|---|---|
| before the change | **16/16** |
| full change | 3/8 |
| prompt rule narrowed | 4/8 |
| context attribute only | 5/8 |

**16/16 vs 12/24, p ≈ 0.0005.** With product labels present the model more often emitted the
bare "dental is excluded" sentence without its accident carve-out, and the verifier correctly
rejected that — so the system abstained on a question it had been answering. Neither component
reproduced the regression in isolation (the verifier grounded the canonical answer 8/8 under
both contexts; generation kept the carve-out 6/6 under both prompts); it only appeared
end-to-end.

The change was reverted in full. Removing the contaminating document fixed the same bug with
no code, no latency, and no regression — dental stayed at 8/8 afterwards.

The lesson is worth keeping: **product-labelling the context is still the right defence for a
genuine multi-product corpus**, but it cannot ship until the groundedness judge stops
rejecting answers on sentence-level phrasing.

## Caveats

- 30 questions, 3 unanswerable. The abstention figure rests on those 3 — directional, not
  calibrated. The golden README targets 250 including `adversarial_injection` and
  `claim_status`; neither category exists yet.
- Questions are authored against a synthetic corpus, not real adjuster tickets.
- hit-rate@3 **and** hit-rate@1 are now both saturated at 1.0000 and cannot measure the next
  change. Use MRR, abstention accuracy and false-abstention count from here.
