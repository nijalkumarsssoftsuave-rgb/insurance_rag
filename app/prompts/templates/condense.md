# Condense — rewrite a multi-turn message into a standalone question
# version: v1
# temperature: 0

Given the conversation history and the latest user message, produce a single
self-contained question that can be understood without the history.

Rules:
- Preserve every entity mentioned earlier (product, claim id, dates).
- Do not answer the question.
- If the message is already standalone, return it unchanged.
