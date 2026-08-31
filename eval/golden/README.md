# Golden dataset

The labelled question set that decides every tuning parameter in the system.
**Commit this directory.** It is the most valuable artefact in the repo.

## Format — `questions.jsonl`, one object per line

```json
{
  "id": "q0041",
  "question": "Is dental treatment covered under Family Health Optima?",
  "category": "coverage_yes_no",
  "reference_answer": "Dental treatment is excluded unless it arises from an accident (Section 4.11).",
  "relevant_sections": [
    {"doc_type": "policy_wording", "effective_from": "2026-04-01",
     "section_path": "SECTION 4 - EXCLUSIONS > 4.11"}
  ],
  "filters": {"product_name": "Family Health Optima", "date_of_loss": "2026-03-14"},
  "should_abstain": false,
  "source": "ticket-88213"
}
```

## How ground truth is labelled

**On payload fields, not chunk ids.** `chunk_id` is a database row UUID
regenerated on every re-ingest, so a set keyed on it dies the first time chunking
or the embedding model changes - which is exactly what an ablation does.
`relevant_sections` is a list of payload predicates instead. A retrieved chunk
satisfies a label when **every** key in it matches; `section_path` matches by
**prefix**, everything else exactly. `filters` maps straight onto
`RetrievalFilter` keyword arguments.

Four rules, each learned from a label that scored a false hit:

1. **Name `kind` when the answer is a figure in a table.** A section that holds a
   table produces three chunks: the `child` prose intro, the `table` itself, and
   a `table_summary` ("Table in SECTION 3 ... with 8 rows, listing Benefit,
   Sub-limit"). Only the `table` carries the numbers. A bare section label counts
   the summary as a hit and reports success while the model saw no figures.
2. **Scope the `section_path` far enough to exclude siblings that answer a
   different question.** `SECTION 5 - WAITING PERIODS` also matches `> 5.2
   Specified Diseases` (24 months) and `> 5.3 Accidental Injury` (none) - both
   wrong answers to a pre-existing-disease question.
3. **Watch for clause numbers that repeat across products.** `SECTION 1 -
   DEFINITIONS > 1.3` is *Hospital* in the health wording and *Total Loss* in the
   motor one, and both are `policy_wording` effective `2026-04-01`. Add
   `product_name`, or spell the section name out in full.
4. **Include `product_name` in the label whenever the target document has one**
   (SOP and circular payloads have `product_name: null`, so those labels must
   omit it). `filters.product_name` is a separate decision: set it only when a
   real router would have extracted the product from the question text.

Where more than one chunk genuinely answers a question, list them all.
`hit_rate@k` is satisfied by any one of them; `recall@k` measures how many were
found, so a multi-source question that retrieves one of two scores 0.5.

## Categories to cover

| Category | Target count |
|---|---|
| `coverage_yes_no` | 40 |
| `sub_limits_amounts` | 30 |
| `waiting_period` | 20 |
| `exclusions` | 30 |
| `claim_procedure` | 25 |
| `definitions` | 15 |
| `multi_hop` | 25 |
| `unanswerable` | 25 |
| `adversarial_injection` | 20 |
| `claim_status` | 20 |
| **Total** | **250** |

## Rules

1. **Source questions from real customer and agent tickets.** Invented questions are
   always cleaner than real ones and will flatter the system.
2. `unanswerable` and `adversarial_injection` are not optional. A system that answers
   everything is a system that hallucinates.
3. `relevant_sections` must be labelled by someone who knows the policy wordings.
   This is the schedule risk — a domain expert, not an engineer, owns this file.
4. Every thumbs-down in production is a candidate row. Review them weekly.
