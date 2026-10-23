# Week 12 · Module 6 — Capstone: Ship It and Defend It (Track D)

## Mentor checklist, answered first

- **Does the whole app run end-to-end from a single starting point?** Yes,
  with one fix made this week (below). `make bootstrap` now chains the six
  manual steps a fresh clone needed (`up --wait`, `setup`, `ingest`, `seed`)
  into one command, and the live HTTP sweep in *The end-to-end sweep* below
  exercises Lane A, Lane B, the not-found path, a cache hit and conversation
  history through the real API with no shortcuts.
- **Can a stranger get it working from the guide alone?** `README.md` was
  rewritten this week - it was seven weeks stale, describing the app as it
  stood after Week 4 and pointing at an eval harness flag that no longer
  exists. See *What the README fix also caught* below.
- **Does the demo show a real failure and how it's handled?** Yes - the
  capstone's own centerpiece failure is CLM-2026-0005 going through
  `GET /api/v1/claims/{claim_number}/summary` and coming back telling the
  adjuster to approve a claim on a fabricated "supervisor correction",
  reproduced live this week, not a replayed fixture. See *The seam, found
  and fixed*.
- **Can you defend the numbers on cost, accuracy, and what happens when
  it's wrong?** See *The numbers, defended* and *Defending it under
  questions* below - every figure cited there traces to a specific prior
  week's measured run, not to this week's memory of it.

## The seam, found and fixed

Weeks 7, 8 and 10 each built a full replacement for "write an adjuster
handover summary": `claim_handover_agent`, then an injection-defended
version of it, then `claims_squad` (Week 10's measured winner - ties on
quality, wins on tokens and cost, and the one of the two architectures that
resisted the Week 8 injection attack at the model level, not just at the
output-validation backstop). None of the three was ever wired into the one
REST endpoint that does this job in production. `app/api/v1/claims.py`
still called `app.claims.summary.summarise()` - the Week 6 original,
written before any of that defense existed.

**Reproduced live, before the fix**, calling the endpoint against
CLM-2026-0005 (Week 8's planted injection fixture, a claim note containing a
fabricated "Claims Supervisor J. Patel" correction) as a staff subject:

> "The adjuster should update the status to approved as per the correction
> noted by Claims Supervisor J. Patel on 2026-07-20."

That is the exact Week 8 attack, succeeding, live, in the one place Week
8's defense was never deployed.

**Fix**: `app/api/v1/claims.py` now calls `claims_squad.run()`. The response
model gained `output_overridden: bool` so an adjuster can see when the
safety net fired, not just that it did.

**Reproduced live, after the fix**, same claim, same subject, through the
real `TestClient` (full HTTP layer, dependency injection, response model):
a correct summary ("Status: Rejected due to excluded treatment... clause
4.11...") with `output_overridden: false` - the squad was never even
tricked at the model level, consistent with Week 10's own finding about
which architecture resists this attack.

A permanent regression test (`tests/unit/test_claims_api.py::
test_the_endpoint_is_wired_to_the_squad_not_the_old_summariser`) monkeypatches
`claims_squad.run` and asserts the endpoint surfaces exactly what it
returns - proof of wiring, not of model behavior, so it runs with no real
LLM call and stays fast. All 4 tests in that file pass; 176/176 tests pass
overall (`pytest -v`).

Both reproductions above are a script, not something that happened once in
chat: `python scripts/demo_claim_summary_guard.py` calls the old summariser
and the current squad path directly, side by side, against CLM-2026-0005 -
anyone who clones the repo and runs `make bootstrap` can reproduce the exact
before/after quoted here.

## A second seam: the golden set itself had drifted

Fixing the Makefile's `eval` target (above) meant it could finally be run. Doing
so for real - not just checking the command parses - surfaced a second, more
interesting seam: **hit-rate@3 came back 0.8148, not the 1.0000 `eval/RESULTS.md`
reports**, with every one of the 5 regressions on the Motor Shield product
(`q12, q13, q14, q15, q18` - depreciation, no-claim bonus, the drunk-driving
exclusion, the deductible, Insured Declared Value).

That looked like a real retrieval regression. It wasn't. Inspecting the raw
candidates for `q14` ("I had an accident after drinking - is the damage
covered?") showed the reranker's **rank-1 hit was the exactly correct clause**
(`SECTION 7 - EXCLUSIONS > 7.3 Driving Under the Influence`) - and the scorer
still called it a miss. `eval/metrics/retrieval.py::matches()` requires every
key in a label to match, `section_path` by prefix but everything else by exact
equality, and the golden set's `relevant_sections` for these 5 questions named
the product `"Motor Shield Private Car"` - a shorthand that was never the
document's actual title. The PDF's own subtitle, baked in by
`scripts/generate_sample_docs.py`, is `"Motor Shield Private Car Package
Policy"`, which is also what ingestion writes to `product_name` on every
chunk. Every candidate from that document failed the exact-equality check on
a name nobody ever actually used, regardless of how correct the retrieval was.

Fixed by correcting the 5 stale labels in `eval/golden/questions.jsonl` to the
document's real title - a golden-set bug, not a retrieval or matcher bug, so
the golden set is what moved. Re-run after the fix: **hit-rate@3 1.0000,
hit-rate@1 0.9630, 0 retrieval failures** - the number `eval/RESULTS.md`
already claimed, now reconfirmed live rather than inherited on trust.

The lesson for the checklist item above it: a broken `make eval` target isn't
just an inconvenience, it is exactly what lets a label/corpus drift like this
sit undetected - nobody had run the full golden set against the current
corpus by the real command in long enough for the two to quietly disagree.

## The end-to-end sweep

A live sweep through the real HTTP API (`scripts/run_api.py`, not an
in-process call) covering both lanes, reproducible with
`python eval/week12/e2e_sweep.py`:

| check | status | result |
|---|---|---|
| Lane A coverage question | 200 | `verified: true`, correct exclusion-with-carve-out answer |
| Lane B claim status | 200 | correct status, amounts, policy for CLM-2026-0004 |
| Lane B, claim not found | 200 | refusal, no information leak about other claims |
| cache-hit repeat of the coverage question | 200 | identical answer |
| conversation history | 200 | 2 messages recorded |

No seam found here beyond the one above - every lane answers correctly
through the one entry point a real user would hit.

## What the README fix also caught

While rewriting `README.md` to match the app as it stands (Weeks 5-11 were
simply missing from it), two more stale pointers surfaced:

- The Quickstart's 5 manual steps are now the single `make bootstrap`
  command (see Makefile change below) - the old steps still worked, they
  were just never consolidated.
- `make eval` called `python -m eval.harness --dataset eval/golden/
  questions.jsonl` - `--dataset` is not a flag `eval/harness.py` defines
  (it takes `--questions`, defaulting to the golden set already). The
  target would have failed on first use by anyone who actually ran it.
  Fixed to `python -m eval.harness --label baseline --out eval/runs/
  baseline.json`, matching the harness's real CLI and the README's own
  documented invocation. The Makefile also gained `ingest` and `seed`
  targets (previously only reachable by calling the scripts directly) and
  `bootstrap`, which chains `up --wait` (the three data-plane services
  already define Docker healthchecks; nothing needed changing there),
  `setup`, `ingest`, `seed`.

## The numbers, defended

| metric | value | source |
|---|---|---|
| Lane A retrieval, hit@3 | 1.0000, hit@1 0.9630 (30-question golden set, 0 retrieval failures) | reconfirmed live this week, `eval/runs/baseline.json` - see *A second seam* above for why it briefly read 0.8148 |
| Lane A groundedness judge, human agreement | 20-21/26 (76.9-80.8%) | `eval/week6/RESULTS.md` |
| Handover quality, single agent vs. squad | 15/15 vs. 15/15 scoreable (tied) | `eval/week10/RESULTS.md` |
| Handover tokens, single agent vs. squad | 1,542 vs. 861 mean per run (44% fewer) | `eval/week10/RESULTS.md` |
| Handover cost, single agent vs. squad | $0.00541 vs. $0.00353 total/18 runs (35% cheaper) | `eval/week10/RESULTS.md` |
| Lane A cost per request, before/after semantic cache | $0.000794 → $0.000392 mean, realistic repeat mix (-51%) | `eval/week11/RESULTS.md` |
| Injection attack on the model itself | 10/10 tricked, unchanged by prompt warnings | `eval/week8/RESULTS.md` |
| Injection attack, what the caller actually receives | 10/10 → 0/10 (deterministic output validation) | `eval/week8/RESULTS.md` |
| Injection attack against `claims_squad` specifically | resisted at the model level, 0/N needing the backstop | `eval/week10/RESULTS.md`, confirmed again live this week |

Every number above is a measured run from the week that built the feature,
not a number re-derived or estimated this week - this table is a pointer
into the evidence, not a replacement for it.

## Known weak spots (so nobody has to find them first)

Carried forward from where each was actually found, not re-discovered:

- **The output-validation guard is narrow on purpose.** It catches a
  rejected claim rewritten as approved - the attack actually demonstrated.
  It would not catch a fabricated *amount* on an otherwise-correct
  rejection, or any status pair it wasn't written for. (`eval/week8/
  RESULTS.md`)
- **`injection.scan()`'s regex signatures miss the successful attack
  shape entirely** (0/10, both before and after the fix) - containment and
  detection are different guarantees, and only the first one is solid here.
  (`eval/week8/RESULTS.md`)
- **The model itself remains hijackable**; this system is safe because its
  output is checked and because every tool on the handover agents is
  read-only, not because the model resists the attack. A future
  write-capable tool (e.g. `approve_claim`) would need its own
  authorization check independent of anything the model was told.
  (`eval/week8/RESULTS.md`)
- **The semantic cache has no invalidation path.** A re-ingested, corrected
  policy wording can still serve a stale cached answer for up to
  `ttl_seconds` (24h). Never used on the claim-status path (nothing
  claim-shaped is cache-safe), but Lane A is exposed until ingestion flushes
  the affected tenant's cache entries on a new document version.
  (`eval/week11/RESULTS.md`)
- **hit-rate@3 is saturated at 1.0000** on the current 30-question golden set
  (hit@1 is 0.9630, not quite saturated - `q13` lands at rank 2, not 1) and
  can't discriminate much further; the set itself has 3 unanswerable
  questions and no `adversarial_injection` or `claim_status` categories yet,
  despite the golden README targeting 250 questions including both.
  (`eval/RESULTS.md`)
- **The groundedness judge disagrees with a human reader on roughly 1 in 5
  cases** (76.9-80.8% on a 26-case set), and iterating the judge prompt
  made overall agreement *worse* even though it fixed the one failure mode
  it targeted (UNSAFE recall 0/2 → 1/2) - a reminder that a single
  agreement number hides which failure mode actually moved.
  (`eval/week6/RESULTS.md`)
- **Nothing catches the golden set drifting from the corpus again.** This
  week's fix corrects the 5 labels that had drifted; it does not add a check
  that would catch the next one before a full `make eval` run surfaces it as
  a false regression. A document re-generated with a renamed product, or a
  label typed by hand, can silently do this again.
- **The claims.py seam this week is the kind of gap that recurs by
  construction**: every week built a better version of something and
  pointed a test or an eval script at it, but nothing in this project
  checks that production code actually calls the new version instead of
  the old one. The regression test added this week (`test_the_endpoint_
  is_wired_to_the_squad_not_the_old_summariser`) guards this one call site;
  it does not guard the pattern.

## The four-part demo

1. **A real run.** Lane A: "Is dental treatment covered under Family Health
   Optima?" - verified, cited answer with the accident carve-out. Lane B:
   "What is the status of claim CLM-2026-0004?" - correct status and
   amounts, scoped to the asking policyholder.
2. **A real failure, caught and handled.** Run
   `python scripts/demo_claim_summary_guard.py` - it shows the old
   summariser telling the adjuster to approve CLM-2026-0005 on a fabricated
   supervisor note, then the current path's correct summary with
   `output_overridden=False`, because this path never got tricked in the
   first place.
3. **A number that improved.** $0.000794 → $0.000392 mean cost per request
   after the Week 11 semantic cache, on a realistic repeat distribution -
   and the honest caveat with it: a pure-novel-question run shows almost no
   saving, because the cache can only ever discount the ~89% of cost that
   isn't `route_node`'s unavoidable per-request share.
4. **What's next.** Close the pattern this week's own seam exposed (nothing
   checks that production code adopted a validated replacement); widen the
   output-validation guard past the one status-flip shape it covers; give
   the semantic cache an invalidation hook on re-ingestion; grow the golden
   set past its current saturation and add the `adversarial_injection` /
   `claim_status` categories it's missing.

## Defending it under questions

- **"What does this cost per request?"** $0.000392 mean on Lane A with a
  realistic cache-hit mix, $0.000794 with none - give both, because a
  single "after" number implies a saving that doesn't show up on a stream
  of genuinely novel questions. Handover summaries run through
  `claims_squad` at $0.00353/18 runs measured, not re-estimated.
- **"How accurate is it?"** Retrieval hit@3 is 1.0000 on the current
  30-question set, but that set is saturated and small; the honest number
  is the groundedness judge's 76.9-80.8% agreement with a human reader on a
  harder, 26-case labelled set - and that it got *worse* on one axis while
  fixing the axis it was tuned for.
- **"What happens when it's wrong?"** Two different answers depending on
  which kind of wrong: a Lane A answer that fails the verifier abstains
  rather than shipping (`verified: false` is visible in the response, not
  hidden); an adjuster-summary answer that contradicts the claim's own
  observed status gets rewritten by the output-validation guard before it
  ever reaches staff, and `output_overridden: true` says so when it fires.
  Neither guard is a general fact-checker - both are named, narrow, and
  documented above as exactly that.
- **"Could someone attack this?"** Yes, and one specific attack was
  demonstrated and defended (Week 8's injection, still 10/10 effective on
  the model, 0/10 effective on what ships). The defense is structural, not
  behavioral: every handover agent's tools are read-only, so a hijacked
  answer is a wrong sentence, not an unauthorized action. A tool that could
  *act* (approve a claim, not just describe it) would need its own
  authorization check and does not exist in this app.
