# Week 5 · Task Set D — failure taxonomy

20 traces, seeded random sample (`seed 20260831`, frame = 162 turns before 2026-08-30).
Full sample list, the 20 observation sentences and the replay evidence: [`notes.md`](notes.md).

Severity: **claim-affecting** = could wrongly deny or wrongly pay · **adjuster-friction** =
merely annoys the person reading it.

| # | Failure mode | Count | % of 20 | Severity | Example trace_id |
|---|---|---|---|---|---|
| 1 | **Names a section heading as a "Clause" the customer cannot look up** — writes "Clause 3 - Benefits and Sub-limits" or appends the clause reference as a loose trailing line, when Section 3 has no clause number | 4 | 20% | adjuster-friction | `976d0b17-8e33-4f89-b30a-e5fa270911c6` |
| 2 | **Refuses a question that an identical repeat answers** — same question, same 8 chunks retrieved, answered on one turn and "couldn't verify" on the next | 2 | 10% | adjuster-friction | `f5f8eab6-1440-41fb-9084-f6d1ee2563ab` |
| 3 | **Says "I couldn't verify" when the documents simply do not mention the topic** — the customer is offered an agent instead of being told the policy is silent | 2 | 10% | adjuster-friction | `04bca064-f358-480b-aede-f4ea16e52768` |
| 4 | **Takes far longer than a turn should** — 25 s and 116 s against a 7.5 s median, with no indication to the caller | 2 | 10% | adjuster-friction | `39f39b25-1b85-425c-8a4c-506c988f9260` |
| 5 | **Writes a clause number in the prose that is not the clause it cited** — answer text says "(Clause 7)", citations say 6.2 and 6.4 | 1 | 5% | **claim-affecting** | `a74f2cd1-a6ab-453c-b1b2-b447e9599b22` |
| 6 | **Answers a question that names no product from whichever product ranked first** — "tell me about the coverage and benefits" answered with one specific policy's Sum Insured of INR 10,00,000 as though it were the customer's | 1 | 5% | **claim-affecting** | `35fe7556-f346-4f4f-8ac2-73042009b1ad` |

**12 of 20 traces (60%) carry at least one of these. 8 of 20 (40%) were clean** — including
all three motor-exclusion answers, both claim-status lookups, and the prompt-injection
attempt, which was refused correctly in about a second.

## What the counts do and do not support

The two **claim-affecting** modes are the two rarest, at one trace each. One occurrence in
20 is not a frequency — it is an existence proof. Both are listed because a mode that can
put the wrong clause number in front of an adjuster matters at n=1 in a way that a
formatting wart does not, but neither count should be quoted as a rate.

The most common mode is also the least dangerous, which is the opposite of a fix order:
frequency alone would send next week's work at citation formatting.

## Dated prediction — 2026-08-31

Attacking **mode 2, "refuses a question that an identical repeat answers"**, because it is
the only mode whose cause is already located: `verify_node` sends the answer to an LLM
groundedness judge, and that judge rejects sentences it accepts on a rerun. Measured this
week on the dental question, 16/16 answered before a context change and 12/24 after — the
judge, not the retriever, decides.

**The change:** grade groundedness per *answer* rather than per *sentence*, so an answer
whose first sentence states an exclusion and whose second states the carve-out is judged as
one claim rather than two.

**The prediction, falsifiable:** re-running the same 20 traces after that change drops mode 2
from **10% (2 of 20) to 0%**, and the sample-wide abstention count falls from **4 of 20 to
2 of 20** — the two aromatherapy turns should still abstain, because the corpus genuinely
does not mention it. If abstentions fall below 2, the change has broken the guard rather
than fixed the judge, and I will have been wrong in the more dangerous direction.
