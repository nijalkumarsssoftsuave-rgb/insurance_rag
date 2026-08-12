# Contextualize — situating prefix for a chunk before embedding
# version: v1
# temperature: 0
# Applied at ingest time. Prompt-cache the document; only the chunk varies.

Given the whole document and one chunk from it, write 1–2 sentences situating the
chunk within the document — what section it belongs to and what it governs.

Output only those sentences. No preamble.
