"""Typed settings (pydantic-settings). Every tunable knob lives here.

Grouped into nested models with env prefixes, so the flat names in .env.example
map straight through: QDRANT_URL -> settings.qdrant.url.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_BASE = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class AppSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": ""}

    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"  # noqa: S104 - containerized, bound by compose
    api_port: int = 8000

    # POC only: link the demo subject to the holder written by
    # scripts/seed_claims.py so the claim lane can be demonstrated. Off by
    # default, which leaves the subject owning nothing and makes claim lookups
    # refuse - the honest behaviour for an unauthenticated caller.
    demo_policy_holder: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


class LLMSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": ""}

    llm_provider: Literal["openai", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    openai_api_key: SecretStr = SecretStr("")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"


class EmbeddingSettings(BaseSettings):
    """BAAI/bge-m3 - dense (1024d) plus learned sparse from one forward pass."""

    model_config = _BASE | {"env_prefix": "embedding_"}

    model: str = "BAAI/bge-m3"
    dim: int = 1024
    # 512, not bge-m3's 8192 default: chunks are ~400 tokens and attention cost
    # is superlinear in sequence length (ARCHITECTURE 4.2).
    max_length: int = 512
    batch_size: int = 16
    device: Literal["cpu", "cuda"] = "cpu"
    use_fp16: bool = False


class RerankerSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "reranker_"}

    enabled: bool = True
    # Measured on a 6-physical-core CPU, 60 candidates:
    #   bge-reranker-v2-m3  14.9 s   <- unusable interactively without a GPU
    #   bge-reranker-base    3.9 s
    # So the CPU default is `-base`. Switch to v2-m3 when RERANKER_DEVICE=cuda,
    # or when multilingual reranking is required (ARCHITECTURE 4.3).
    model: str = "BAAI/bge-reranker-base"
    device: Literal["cpu", "cuda"] = "cpu"
    top_n: int = 8
    # How many candidates reach the cross-encoder. Latency is linear in this, so
    # it is the main dial for the retrieval budget: 60 -> 3.9 s, 24 -> ~1.5 s.
    candidates: int = 24
    # tau - below this the bot abstains rather than answering (ARCHITECTURE 8.6g).
    #
    # Deliberately low. Cross-encoder scores are not calibrated across query
    # types: a correctly-ranked dental clause scored 0.013 while a room-rent
    # match scored 0.914, so a threshold high enough to be meaningful for one
    # question falsely abstains on another. This value is a "clearly irrelevant"
    # floor only; groundedness is enforced by `verify.py`, not by this number.
    # CALIBRATE ON THE GOLDEN SET before trusting it (ARCHITECTURE 11.3).
    score_threshold: float = 0.01


class QdrantSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "qdrant_"}

    url: str = "http://localhost:6333"
    grpc_port: int = 6334
    prefer_grpc: bool = True
    api_key: SecretStr | None = None
    collection: str = "policy_chunks_v1"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    search_ef: int = 128
    quantization: Literal["none", "scalar"] = "scalar"
    timeout: int = 60

    # Named vector keys. Kept here so the writer and the reader cannot drift apart.
    dense_vector_name: str = "dense"
    lexical_vector_name: str = "lexical"
    bm25_vector_name: str = "bm25"


class StorageSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": ""}

    database_url: str = "postgresql+psycopg://insurance:insurance@localhost:5432/insurance_rag"
    redis_url: str = "redis://localhost:6379/0"
    object_store: Literal["local", "s3"] = "local"
    object_store_path: Path = PROJECT_ROOT / "data" / "raw"

    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Alembic, Celery workers and scripts use the sync engine."""
        return self.database_url


class IngestionSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "ingest_"}

    # Docling by default, for every PDF.
    #
    # An earlier design routed to a much faster parser (pypdfium2) and escalated
    # to Docling only when a table was detected. Three successive heuristics for
    # "does this document contain a table" all failed on real documents - numeric
    # density flagged ordinary clauses, absolute line length misfired across page
    # geometries, and consecutive-run detection missed grids whose rows wrap.
    #
    # Policy wordings nearly always contain benefit grids, and a flattened grid
    # gives customers the wrong sub-limit with full confidence. Paying ~35 extra
    # seconds per document to always parse correctly is the better trade than a
    # heuristic that is wrong in the direction of silent data corruption.
    #
    # Set true only for a corpus you know is prose-only.
    prefer_fast_parser: bool = False
    ocr_enabled: bool = False


class ChunkingSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "chunk_"}

    child_tokens: int = 400
    child_overlap: int = 80
    parent_max_tokens: int = 2000
    strategy: Literal["fixed", "recursive", "structure_aware"] = "structure_aware"
    # Bumped whenever the chunker changes shape, so two strategies can coexist
    # in one collection and be compared (ARCHITECTURE 7.2).
    strategy_version: str = "v1"


class RetrievalSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": ""}

    retrieval_dense_top_k: int = 50
    retrieval_sparse_top_k: int = 50
    retrieval_bm25_enabled: bool = False
    rrf_k: int = 60
    context_token_budget: int = 6000
    query_expansion_enabled: bool = True
    # Each variant is another local bge-m3 forward pass AND another prefetch
    # branch feeding the cross-encoder, so variant count multiplies the two
    # slowest stages. Measured on CPU: 4 variants cost 4.3 s to encode and
    # pushed reranking to 5.7 s. One paraphrase keeps most of the recall benefit
    # at a quarter of the cost; raise it if you move to a GPU.
    query_expansion_variants: int = 1
    # Off by default on CPU: a hypothetical-answer probe is a whole extra LLM
    # call plus an extra encode, for a recall gain the eval harness has not yet
    # measured on this corpus.
    hyde_enabled: bool = False
    contextual_prefix_enabled: bool = False


class CacheSettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "semantic_cache_"}

    enabled: bool = True
    threshold: float = 0.97
    ttl_seconds: int = 86400


class SecuritySettings(BaseSettings):
    model_config = _BASE | {"env_prefix": ""}

    jwt_secret: SecretStr = SecretStr("change-me-to-a-long-random-string")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    pii_masking_enabled: bool = True
    injection_guard_enabled: bool = True
    rate_limit_per_minute: int = 30


class ObservabilitySettings(BaseSettings):
    model_config = _BASE | {"env_prefix": "langfuse_"}

    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: SecretStr | None = None
    secret_key: SecretStr | None = None


class Settings(BaseSettings):
    model_config = _BASE

    app: AppSettings = Field(default_factory=AppSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


def ensure_model_cache() -> Path:
    """Point HuggingFace at ``data/models`` before any model library imports.

    ``HF_HOME`` in .env is read by pydantic-settings but never reaches
    ``os.environ``, and transformers only consults the real environment. Without
    this, weights land in the user profile - so a developer downloads 4.5 GB into
    ``data/models`` and the worker then downloads them again somewhere else.

    Call from every model loader. Uses ``setdefault``, so an explicitly exported
    HF_HOME still wins.
    """
    import os

    cache = Path(os.environ.get("HF_HOME") or PROJECT_ROOT / "data" / "models")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    return cache
