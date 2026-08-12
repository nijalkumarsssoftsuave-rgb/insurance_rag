# Router — intent classification and entity extraction
# version: v1
# temperature: 0
# Returns structured output: {intent, claim_id, policy_id, product, date_of_loss, needs_authz}

Classify the user message into exactly one intent and extract any entities present.

Intents:
- POLICY_QA     — a question answerable from policy documents
- CLAIM_STATUS  — asks about a specific claim's state, history or payout
- CLAIM_INTAKE  — wants to start or file a claim
- SMALLTALK     — greeting or chit-chat
- OUT_OF_SCOPE  — unrelated to insurance

Extract only entities that are explicitly present. Never infer a claim id.
If the user references a past event, extract `date_of_loss` — it decides which
policy version applies.
