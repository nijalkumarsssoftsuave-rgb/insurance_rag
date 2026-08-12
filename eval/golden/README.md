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
  "relevant_chunk_ids": ["d17:s4.11:c0", "d17:s4.11:c1"],
  "filters": {"product_name": "Family Health Optima", "date_of_loss": "2026-03-14"},
  "should_abstain": false,
  "source": "ticket-88213"
}
```

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
3. `relevant_chunk_ids` must be labelled by someone who knows the policy wordings.
   This is the schedule risk — a domain expert, not an engineer, owns this file.
4. Every thumbs-down in production is a candidate row. Review them weekly.
