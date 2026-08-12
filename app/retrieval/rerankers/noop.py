"""Pass-through reranker for ablation baselines.

Re-exported from `bge_reranker` so both implementations sit beside the model
loading they share. Kept as its own module because the ablation grid refers to it
by name.
"""

from app.retrieval.rerankers.bge_reranker import NoOpReranker

__all__ = ["NoOpReranker"]
