# Week 11 · Module 6 — Production: Observability, Cost & the Failure→Test Loop (Track D)

## Mentor checklist, answered first

- **Find a specific past answer from the logs quickly?** Yes - *The support
  drill*, below: a query derived from a vague complaint alone returns 8
  candidates from the real `messages` table, narrowed to exactly 1, confirmed
  only afterward.
- **Logs show time and cost per step, not one total?** Both now do -
  `messages.trace.timings_ms` (already existed, Week 5) and
  `messages.trace.tokens_by_step` / `cost_usd_by_step` (new this week), same
  per-node shape, same reason.
- **Measured cost per request and improved it?** Yes - `eval/week11/
  cost_before.json` / `cost_after_*.json`: $0.000794 → $0.000392 mean per
  request on a realistic repeat distribution (two numbers given, not one -
  see *Cutting it* for why a single "after" figure would mislead).
- **Turned a real failure into a permanent test?** Yes -
  `tests/unit/test_coverage_exclusion_regression.py`, proven to actually fail
  against the planted answer before being trusted, the same discipline every
  regression test this project has shipped has needed.

## What was actually being logged before this week

`messages.input_tokens` / `output_tokens` have existed as columns since the
schema was first written. Before this week, **every row ever written had
them NULL** - confirmed against the live table, not assumed: 36/36 real
messages, 0 with a token count.

Root cause: three of the four LLM calls a Lane A turn makes
(`route_node`, `generate_node`, `verify_node`) go through
`LLMProvider.structured()`, and `structured()` discarded `response.usage`
entirely - by design, not by bug, but a design nobody had revisited since.
The fourth call (`condense_node`, via `.complete()`) *did* get usage back
and threw it away anyway. `timings_ms` had been tracked per node since Week
5; cost never was, on any of the four calls that actually cost money.

**Fixed at the source, not patched at the edges.** `structured()` now
returns `tuple[T, Completion]` (`app/llm/base.py`, both providers) - the same
move Week 7 already made once, for the same reason: don't hide a real
number behind a method that discards it, fix the method. Six call sites
needed updating (`route.py`, `generate.py`, `verify.py`,
`ingestion/metadata.py`, `expansion.py` ×2); three of them don't need the
usage and now say so explicitly (`parsed, _usage = await ...`).

Query expansion (`app/retrieval/expansion.py`) turned out to matter more
than expected: `QUERY_EXPANSION_ENABLED=true` in this deployment, so up to 4
more LLM calls (paraphrase, HyDE, step-back, decompose) run concurrently
inside what the trace had only ever labelled one lump "retrieve" step. Their
cost is now threaded through `ExpandedQuery.tokens` → `RetrievalOutcome.
tokens_by_step` → `retrieve_node`'s own `tokens_by_step["retrieve_expand"]` -
three files, not one, because the cost genuinely originates three layers
below where the graph records it.

**One real row, after the fix** (`messages.trace`, abbreviated):

```json
"tokens_by_step": {
  "route":           {"input_tokens": 399,  "output_tokens": 44},
  "retrieve_expand":  {"input_tokens": 176,  "output_tokens": 18},
  "generate":        {"input_tokens": 1922, "output_tokens": 144},
  "verify":          {"input_tokens": 2028, "output_tokens": 17}
},
"cost_usd_by_step": {
  "route": 0.0000863, "retrieve_expand": 0.0000372,
  "generate": 0.0003747, "verify": 0.0003144
}
```
`messages.input_tokens`/`output_tokens` are now the sum of this (4,525 /
223 on that row) - populated for the first time, not just added as columns.

`audit.record_answer()` is also wired in now (`app/security/audit.py`,
called from `chat.py::_persist`) - it existed, fully correct, and was never
called; chat answers were invisible in the audit trail that claim views
already appear in.

## A bug found by actually exercising the logging path, not by reading it

Running `eval/week11/cost_report.py` with a test subject whose `user_id`
didn't yet exist in `users` triggered a real `ForeignKeyViolation` on the
audit write inside `claim_lookup_node` - expected, and `audit.record()`
logged it correctly. What wasn't expected: the *next* operation on that same
session (the node's own `commit()`) then raised `PendingRollbackError`, a
crash that looks completely unrelated to auditing, from code that never
touches `audit.py`. `record()`'s docstring says "never raises" - true for
the caller that invoked it directly, false for whatever ran next on the same
session, because a failed `flush()` leaves SQLAlchemy in a dirty state that
only an explicit `rollback()` clears, and `record()`'s `except` block never
issued one.

Fixed with one line (`await session.rollback()`), and
`tests/unit/test_failed_audit_write_does_not_poison_the_session` proves it:
confirmed the test fails without the fix (git-stashed it and re-ran - real
`PendingRollbackError`, not a hypothetical one) before trusting it as a
regression guard.

## The support drill: finding a planted bad answer from a vague complaint

Track D: *"Find the coverage answer that ignored an exclusion."*
`scripts/plant_drill_fixture.py` plants one answer, backdated ~23 days, into
the real `messages` table: asked *"Is dental treatment covered under my
Family Health Optima policy?"*, answered *"Yes, dental treatment is covered
under your Family Health Optima policy"* - no mention of clause 4.11, the
exclusion that actually applies (dental is excluded unless
accident-related; confirmed live, multiple times this session, that the
real pipeline currently states this correctly). This is a plant, as the
brief asks for, not a claim the bug exists today.

**The complaint** (the only thing a real support engineer would have):
*"A customer said last month we told them dental was covered, and then
their claim got rejected."*

**The query, derived from that sentence alone** - before looking at any
specific row:

```sql
SELECT * FROM messages
WHERE role = 'assistant' AND content ILIKE '%dental%'
ORDER BY created_at DESC;
```

**8 candidates** - real ones, not a planted set of one:

| date | confidence | content (truncated) |
|---|---|---|
| 2026-10-05 | 1.0 | "Dental treatment... excluded from..." |
| 2026-09-21 ×2 | 1.0 | claim-status answers (CLM-2026-0004/0005) |
| 2026-09-21 ×3 | 1.0 | "Dental treatment... excluded from..." |
| 2026-09-21 | 1.0 | claim-status answer |
| **2026-09-12** | **0.85** | **"Yes, dental treatment is covered..."** |

**Narrowed** using what the complaint itself says - *told them it WAS
covered* - not anything known in advance about which row is the plant:

```sql
... AND content ILIKE '%covered%'
    AND content NOT ILIKE '%exclu%'
    AND content NOT ILIKE '%clause%'
```

**1 row.** Checked `trace.planted` only afterward, to confirm - it was
`true`. The row's own shape corroborates the complaint independently: it is
the oldest of the 8 (matches "last month"), and the only one with
`confidence < 1.0` (0.85 - a real model would be less certain about a claim
with nothing supporting it, which is itself a signal worth building an
alert on, named below under *what's not covered*).

## Turned into a permanent test

`tests/unit/test_coverage_exclusion_regression.py`:
`dental_coverage_answer_states_the_exclusion()` - a plain regex, no LLM
(same reasoning as `eval/assertions.py`): an answer that affirms dental
coverage must also name the exclusion or its accident exception.

**Proven against the planted text before being trusted**, in the test
itself, not as a one-off check I ran and discarded:

```python
def test_check_actually_fails_on_the_planted_answer():
    ok, detail = dental_coverage_answer_states_the_exclusion(PLANTED_BAD_ANSWER)
    assert ok is False
```

...and proven not to false-positive on the real answers this session's live
API calls actually produced, so the guard is live without being noisy.

## Cost per request, measured and cut

**Baseline** (`cost_before.json`, cache not yet built, 15 runs across 5
representative questions, 3 trials each):

| | |
|---|---|
| mean / p99 cost per request | **$0.000794** / $0.001093 |
| share of cost by step | generate 43%, verify 39%, route 11%, retrieve_expand 7% |

Generate + verify alone are 82% of the bill - both carry the full retrieved
context in their prompt, which is why they dominate token count even though
neither makes more than one call.

### Cutting it

`app/cache/semantic.py` was a docstring-only stub before this week - same
shape as `app/claims/tools.py` before Week 7 and `scan_document` before
Week 8. Built on pieces that already existed and were never connected: the
embedder retrieval already loads, Valkey (`docker-compose`'s `valkey`
service, already running for Celery), and `CacheSettings.threshold`/
`ttl_seconds` (already configured in `app/config.py`, read by no code until
now).

Wired into the graph around the expensive middle of Lane A only
(`cache_check` after `route`, `cache_store` after `verify` -
`app/graph/builder.py`): a hit skips expansion, retrieval, generation and
verification entirely. **Never on the claim-status path** - a cached entry
carries no customer identity (policy wording reads the same for every
policyholder on a product), so nothing claim-shaped is safe to cache and
nothing claim-shaped is ever written there; enforced by which graph path
reaches `cache_store`, not by a filter inside the cache module itself.

**Two "after" numbers, not one - reporting only one would mislead, in
whichever direction it happened to be run:**

| | mean cost/req | vs. baseline |
|---|---|---|
| **Pure cold** (every question genuinely novel, 0 hits possible - `cold_cache_check`) | $0.000739 | ~unchanged - confirms the miss path costs nothing extra in $ (the one added embedding call is CPU, not a paid token) |
| **Realistic repeat distribution** (`cost_after_repeat_distribution.json` - same 5 questions × 3 trials, so 10/15 runs are genuine hits) | **$0.000392** | **-51%** |

The theoretical ceiling is lower than 51%: `cache_check` sits *after*
`route_node` (routing has to run to know a question is even cacheable), so
route's ~11% cost share is paid on every request, hit or miss - the cache
can only ever save the other 89%.

**What the 0.97 similarity threshold (pre-configured, never used before
this week) actually catches, measured, not assumed** - same embedder,
same question, four variants:

| variant | cosine score | clears 0.97? |
|---|---|---|
| "is dental treatment covered?" (case only) | 0.9958 | yes |
| "Does my policy cover dental treatment?" | 0.9099 | no |
| "Is dental treatment covered under my policy?" | 0.9072 | no |
| "What is the waiting period for pre-existing diseases?" | 0.5099 | no, correctly |

The cache catches near-duplicate phrasing (typos, casing, whitespace, the
same question asked the same way twice - which real support traffic does,
a lot), not loose paraphrasing - bge-m3 puts genuine paraphrases at
~0.90-0.91 for short insurance questions, below the configured bar. The
threshold is conservative by design (a false hit serves wrong information,
worse than a miss that just costs a few cents), and the score separation
between "related" (0.90) and "same topic, different question" (0.51) is
clean enough that 0.97 is doing the conservative thing correctly rather
than being miscalibrated - lowering it would trade safety for a higher hit
rate, a real trade-off, not a free win, and not one this week makes
unilaterally.

## The 10× plan - what breaks first, and the plan for it

Ranked by how directly each is evidenced this session, not by guess:

1. **The embedder and reranker are explicitly single-threaded, in code,
   today** (`threading.Lock()` in both `app/embeddings/bge_m3.py` and
   `app/retrieval/rerankers/bge_reranker.py`, with a comment explaining why:
   `FlagEmbedding`'s `encode` is not documented thread-safe). Every
   concurrent Lane A request queues behind the same lock for both passes.
   This is the first thing 10× traffic saturates, and it saturates
   regardless of how many FastAPI workers or replicas sit in front of it,
   because each replica has its own lock around its own copy of the same
   models. **Plan**: batch concurrent requests into one `encode()` call
   (the embedder already batches *within* one query's expansion variants -
   extending that across concurrent requests is the same technique, not a
   new one), or move embedding/reranking to a dedicated GPU-backed service
   so CPU lock contention stops being the shared resource.
2. **Cold-start model loading is real and currently always lazy** - observed
   this session, repeatedly: ~5-13s for the embedder, ~5-9s for the
   reranker, every time a fresh process first serves a request (`@property`
   loads on first access in both modules). At 10× traffic with
   autoscaling, every new replica pays this tax at the worst possible
   moment - under the load spike that triggered the scale-out. **Plan**:
   load both at process startup (a FastAPI lifespan hook), not on first
   request - a config change in spirit, a few lines in practice.
3. **`SecuritySettings.rate_limit_per_minute` is configured and enforced
   nowhere** - confirmed by grep, zero call sites outside `config.py`
   itself. Nothing currently stands between a traffic spike and the
   lock-serialized embedder above. **Plan**: the dependency is one line to
   add (`slowapi` or equivalent) once this is prioritized; the harder part
   is picking a number that protects the embedder without throttling
   legitimate 10× traffic, which needs the capacity numbers from (1)
   measured, not guessed.
4. **The semantic cache's own lookup is a linear scan**, `_MAX_SCAN=200`
   entries per tenant, deliberately - no vector index exists in Valkey
   (that's a RediSearch module this deployment doesn't run). Fine at this
   corpus's scale; a known, named ceiling (consistent with how this project
   marks deliberate simplifications), not a silent one. **Plan if it
   matters**: reuse Qdrant - it already holds vectors and already serves
   this app, rather than standing up a second vector index.
5. **DB pool** (`db_pool_size=10`, `max_overflow=20`, 30 total) is
   unexamined under real load - adequate for this session's traffic, not
   validated against 10×. **Plan**: load-test and size from the number, not
   from the current default.
6. **No client-side concurrency control in front of the OpenAI calls** - the
   existing `tenacity` retry (3 attempts, exponential backoff) handles a
   transient 429 gracefully for one request; it does nothing to stop many
   concurrent requests from *causing* a wave of 429s together at 10×
   volume. **Plan**: a semaphore or token-bucket in `app/llm/` ahead of
   the retry, sized to this account's actual OpenAI rate limit.

## Fine-tuning - why not this week

Listed as the brief's own "last resort," and this project's history backs
that ordering up: every week so far that found a real quality gap fixed it
with a prompt, a retrieval filter, or a deterministic check (Week 4's
corpus-contamination fix, Week 6's assertion fixes, Week 7's
`catalogue.resolve()` fix) - cheaper to build, cheaper to verify, and
reversible in a way a fine-tuned checkpoint is not. Nothing measured this
week or any prior week points at a failure mode only weight updates could
fix; fine-tuning stays the documented last resort, not the next thing
reached for.

## What's not covered

- **The cache has no invalidation path.** If a policy wording is corrected
  via re-ingestion, a cached answer from the old wording can still be
  served for up to `ttl_seconds` (24h, configured, not exercised this
  week). A real fix needs `app/ingestion/pipeline.py` to flush the relevant
  tenant's cache entries on a new document version - not built, named.
- **The 0.85-confidence signal on the planted row** (the lowest of all 8
  candidates, and the only genuinely wrong one) is real and not yet acted
  on: no alert exists today that would surface "a Lane A answer shipped
  with low confidence" on its own, before a complaint ever arrives. That is
  the natural next step in the data flywheel this week's brief names, not
  built this week.
- **No LangSmith/Phoenix/OpenTelemetry integration.** Every log in this
  report is structured logging plus the `messages.trace` JSONB column
  (Week 5), queried directly - sufficient for everything the mentor
  checklist asks this week, and a real gap against a production deployment
  that needs distributed tracing across more than one process.
