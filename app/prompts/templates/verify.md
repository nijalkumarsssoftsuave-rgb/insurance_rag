# Verify — groundedness and citation check
# version: v1
# temperature: 0

Given the CONTEXT, the QUESTION and the draft ANSWER, decide:

1. Is every factual claim in the ANSWER supported by the CONTEXT?
2. Does every citation id in the ANSWER appear in the CONTEXT?
3. Does the ANSWER state coverage without checking the exclusions present in CONTEXT?

Return {grounded: bool, unsupported_claims: [...], invalid_citations: [...]}.
Default to grounded=false when uncertain.
