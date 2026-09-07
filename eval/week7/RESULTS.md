# Week 7 · Module 4 — Agent Loops, Track D (Insurance claims)

One command: `python -m eval.week7.race` · 5 claims · agent vs. fixed sequence,
same task, same tools, same data. Numbers below are read directly off the
`race_results.json` this run wrote - re-running will move them slightly
(model latency varies run to run) but not the shape of the result.

| | |
|---|---|
| **Task raced** | Investigate one claim and write its handover: status, notes, and (if rejected) the clause behind it |
| **Total wall time** | fixed **460 ms** vs agent **19,201 ms** (5 claims) |
| **Average latency** | fixed **92 ms** vs agent **3,840 ms** — **~42x** slower |
| **Tokens spent** | fixed **0** vs agent **7,309** (6,505 in / 804 out) - fixed spends none, there is no model in that path |
| **LLM calls** | fixed **0** vs agent **13** (2-3 per claim) |
| **Reliability** (2 transferable Week-6 assertions x 4 applicable claims) | fixed **8/8** vs agent **8/8** — **not a fair fight, see below** |

**On this task the fixed sequence wins on speed and cost and the two paths tie
on the assertions that transfer - but that tie does not mean "equally good."**
See *Reliability, honestly* and *Which one would I ship* below for why.

## The feature had to be built first

Nothing here existed. `app/claims/tools.py` was a docstring - `"""LangGraph tool
definitions bound to the authenticated subject."""` - and no function. No agent
loop, no fixed-sequence counterpart, no race harness. Built from scratch:

- `app/claims/tools.py` - three tools (`get_claim_status`, `get_claim_notes`,
  `search_policy_clause`), each scoped by a server-supplied `AuthSubject`.
- `app/agents/claim_handover_agent.py` - the hand-built ReAct loop.
- `app/claims/handover.py` - the fixed-sequence counterpart, same task.
- `eval/week7/race.py` - runs both against the same claims, writes
  `race_results.json`.

## A real bug, found by actually calling dead code

`RetrievalPipeline.retrieve_by_clause()` already existed in
`app/retrieval/hybrid.py` - a targeted, unembedded filter lookup for exactly
this hand-off. It was never called from anywhere in the codebase. Wiring it up
as `search_policy_clause` was the first time it ever ran, and it returned the
**wrong clause wording** on the first try:

```
hits: 2
--- 'Dental treatment... excluded... in all circumstances, whether or not
     necessitated by an accident...'                    <- effective 2024-04-01 to 2026-03-31
--- 'Dental treatment... This exclusion shall not apply where such
     treatment is necessitated by an accident...'        <- effective 2026-04-01 to 2027-03-31
```

Two non-superseded wordings of clause 4.11 exist, a claim apart, because the
policy was reworded 2026-04-01. `retrieve_by_clause` built its own filter from
scratch - tenant + section path + `is_superseded=False` - and never applied
`rf.product_name` or `rf.date_of_loss` at all, so either version could come
back depending on scroll order. This is the identical failure class
`app/graph/nodes/retrieve.py` already documents as ARCHITECTURE 5.3 for the
main retrieval path (a dental question once answered from expired wording for
the same reason). It just hadn't reached this second, unused method.

**Fix**: `retrieve_by_clause` now builds its filter with `filters.build(rf)` -
the already-correct, already-tested builder every other retrieval path uses -
instead of a second hand-rolled one, then adds the section-path match on top.
One call site changed, in `app/retrieval/hybrid.py`.

**Checked past the one clause that surfaced it**, since this is shared code
every other caller now inherits too - a fix verified on n=1 is how Week 4's
first NCB fix passed before it regressed dental. Re-ran with the date filter
against every clause ref the seeded corpus and the Week-6 cases actually use:

| clause | product | date of loss | hits | kind |
|---|---|---|---|---|
| 4.11 | Family Health Optima | 2026-05-27 | 1 | child |
| 4.12 | Family Health Optima | 2026-05-27 | 1 | child |
| 5.1 | Family Health Optima | 2026-05-27 | 1 | child |
| 5.2 | Family Health Optima | 2026-05-27 | 1 | child |
| 7.2 | Motor Shield Private Car | 2026-05-27 | 1 | child |
| 4.11 | Family Health Optima | 2025-01-01 (pre-reword) | 1 | child, correctly the *older* wording |

Every case returns exactly one hit on the wording actually in force. The
`filters.build()` `kinds` filter (CHILD/TABLE/TABLE_SUMMARY, absent from the
old hand-rolled version) did not exclude anything here - every clause in this
corpus sits on a `child` chunk - but that is a fact about this corpus, not a
guarantee, and is why this got checked rather than assumed.

## The agent

`app/agents/claim_handover_agent.py` - think, act, observe, repeat, until
`finish` or a budget trips.

**Not built on `structured()`.** The router and verifier already use
`LLMProvider.structured()` for constrained decoding, but
`app/llm/openai_provider.py` discards `response.usage` entirely in that method
- no token count comes back. Since token cost is exactly what this week
measures, the loop instead prompts for a JSON object and parses it defensively
(brace-extraction, the same technique `openai_provider._json_fallback` already
uses for non-native providers) and calls `complete()`, which does return usage.

**Stop conditions**, every one exercised by
`tests/unit/test_claim_handover_agent.py` with a queued fake LLM (no API
calls, deterministic, 8 tests):

| condition | limit | test |
|---|---|---|
| step budget | 6 decide/act cycles | `test_never_finishing_stops_at_max_steps` |
| time budget | 20 seconds | `test_time_budget_stops_before_a_second_call` |
| token budget | 4,000 tokens | `test_token_budget_stops_before_a_second_call` |
| repeated action | same tool + same args twice | `test_repeated_action_stops_the_loop` |
| malformed output | any JSON parse failure | `test_malformed_output_stops_without_crashing` |

Every stop path still returns an answer built from whatever was actually
observed before the budget tripped, rather than nothing.

**Memory**, and where it doubles as a correctness guard. Short-term only: the
running transcript of past actions and observations, replayed into every
prompt. Once `get_claim_status` returns a claim, its `product_name` and
`date_of_loss` go into an in-loop memory dict. Any later
`search_policy_clause` call has those two fields overwritten from memory
before it runs, regardless of what the model put in `action_input` - and if
the model tries to call it *before* `get_claim_status` (memory still empty),
the tool refuses outright rather than running unfiltered:

```
"Look up the claim with get_claim_status first - search_policy_clause needs its date of loss."
```

covered by `test_clause_search_refuses_without_a_prior_claim_lookup`. Refusing
here is the same choice `AuthSubject.require_policy_holder` makes elsewhere in
this codebase - raise rather than silently run an unscoped or unfiltered
query - applied to `date_of_loss` because that is what picks which wording
version the bug above was about. Long-term/cross-session memory (mem0, a
vector store) was not built - out of scope for a single investigate-and-report
task, and the brief lists it under advanced topics, not the core deliverable.

## The fixed sequence

`app/claims/handover.py` - the same branch `app/graph/nodes/claim_lookup.py`
already hardcodes for the customer chat lane (rejected + has a clause ref ->
fetch it), packaged as a function that returns the same shape the agent
returns, and it always calls all three tools that apply - status, notes, and
(if rejected) the clause. Zero LLM calls anywhere in it: every branch is
already known, so there is no decision to spend a model call on.

## The race

5 claims: the 4 real seeded ones (`scripts/seed_claims.py` - settled,
under_review, info_required, rejected-with-clause) plus one that does not
exist. Full per-claim numbers in `race_results.json`.

| claim | status | fixed ms | agent ms | agent tokens | agent LLM calls | agent tools called |
|---|---|---|---|---|---|---|
| CLM-2026-0001 | settled | 14 | 4,721 | 1,050 | 2 | `get_claim_status` |
| CLM-2026-0002 | under_review | 16 | 4,514 | 1,684 | 3 | `get_claim_status`, `get_claim_notes` |
| CLM-2026-0003 | info_required | 11 | 4,006 | 1,696 | 3 | `get_claim_status`, `get_claim_notes` |
| CLM-2026-0004 | rejected (clause 4.11) | 417 | 3,695 | 1,856 | 3 | `get_claim_status`, `search_policy_clause` |
| CLM-9999-9999 | not found | 2 | 2,265 | 1,023 | 2 | `get_claim_status` |

Every run finished cleanly - none of the 5 live claims tripped a stop
condition; that only happens in the adversarial unit tests above, by design,
since a working model on an easy task should not be looping.

**Cost.** At gpt-4o-mini pricing the 5-claim run is about **$0.0015** total -
not the number that matters here. What matters is *zero vs. non-zero*: the
fixed path has no per-call cost to multiply by claim volume, and the agent's
does not disappear at scale, it multiplies with it.

## Reliability, honestly

`claim_number_echoed` and `denial_cites_clause` (from `eval/assertions.py`, run
against `app.claims.summary.from_claim_view`) both score 8/8 - the 4 applicable
claims (not-found has no record to check against) times 2 checks each. But
this tie does not discriminate the way the headline table makes it look like
it does: `claim_number_echoed` **cannot fail** on the fixed path, because the
fixed path builds the claim number from the database row, not from anything
that could drift. It is the same shape of problem as Week 6's always-SAFE
baseline dominating a 24:2 label set - a check that one arm structurally
cannot fail is not evidence the two arms performed equally.

The table above shows why. On **CLM-2026-0001 and CLM-2026-0004** the agent
decided `get_claim_notes` was not worth calling and finished without it. On
CLM-2026-0004 that costs nothing observable (the clause text carries the
rejection). On **CLM-2026-0001 it does**: the only note on that claim is
*"Amount settled directly with the network hospital"* - the operative fact for
a settled claim, since it is what actually confirms the money moved - and the
agent's handover never mentions it, while the fixed sequence's does. Neither
assertion catches this, because both assertions check for invented or
contradicted facts, not omitted ones. This is not "flexibility the fixed
sequence lacks" - for an adjuster handover it is **under-answering on 1 of 4
real claims**, and it happened on two separate live runs of the race, not once.

## Which one would I ship, and why

**The fixed sequence, for this task, without reservation** - now more strongly
than the raw 8/8-vs-8/8 suggests, once the tie is read correctly. The branch
space is fully known and small: found-or-not, rejected-with-clause-or-not.
There is nothing here for a model to discover that the code does not already
know how to check, and the agent's one behavioral difference from the fixed
path - skipping a tool call it judged unnecessary - produced a worse answer on
a settled claim, not a better one. Given that, the agent costs ~42x the
latency and real (if small) token spend, for output that is no more reliable
on the checks that transfer and demonstrably thinner on at least one real
claim.

That is not an argument against building the agent - it is the argument this
week is built to make. I would reach for an agent version of this feature the
day the *set of steps itself* becomes the unknown: for example, an adjuster
research assistant that has to decide, per claim, how many documents to pull
and from where, when that cannot be enumerated as an `if`. Lane A's retrieval
already lives at that edge (variable number of query expansions, a broadened
retry, companion fetches) - it is coded as a fixed pipeline today, and that is
a judgment call the same way this one is, not a law. A claim handover is not:
every claim in this system takes exactly one of five known shapes, and the code
already knew that before the agent was asked to find out for itself.
