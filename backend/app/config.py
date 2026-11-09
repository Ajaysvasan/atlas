import os
from pathlib import Path

GIB = 1 << 30

import numpy as np

from logging_setup import (
    configure_logging,
    get_logger,
    log_context,
    log_timing,
    new_correlation_id,
)


class Config:
    """Configuration class for the backend application."""

    APP_NAME = "Final Year Project Backend"
    DATASET_PATH = Path(__file__).resolve().parent / "dataset"
    DEBUG = False
    LOG_FILE = "log/app.log"
    ABS_PATH = Path(__file__).resolve().parent
    DATA_DIR = os.path.join(ABS_PATH, "data")
    DB_PATH = os.path.join(DATA_DIR, "hierarchical_db")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    # None: the draft model also answers. A different GGUF here is loaded once,
    # beside the draft model, and serves the main role only.
    MAIN_MODEL_PATH = None
    DRAFT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    DRAFT_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
    DRAFT_MODEL_PATH = os.path.join(ABS_PATH, "models", "draft_model")
    # One resident instance serves every prompt; its KV cache is sized to this,
    # not to the model's 128k maximum (~5 GB at 131072, ~1.2 GB here).
    SHARED_MODEL_CONTEXT_WINDOW = 32_768
    DRAFT_MODEL_CONTEXT_WINDOW = SHARED_MODEL_CONTEXT_WINDOW
    MODEL_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))
    INDEX_PATH = os.path.join(DATA_DIR, "disk_ann_index")
    DISTANCE_METRIC = "l2"
    VECTOR_DTYPE = np.float32
    EMBEDDING_DIMENSIONS = 128
    MAX_VECTORS = 1000000
    COMPLEXITY = 100
    GRAPH_DEGREE = 120
    NUM_THREADS = 4
    K_NEIGHBORS = 9

    # Document chunk vectors live in pgvector's `vectors` table under this
    # project id. Real project ids are uuid4 hex, so none can be this.
    GLOBAL_VECTOR_SCOPE = "global"

    # The vector index never plans to hold more than this in RAM, however large
    # the corpus grows; past it the built index moves to disk.
    VECTOR_INDEX_RAM_MB = 512
    # Vectors ingested since the last index build, held in memory until the next.
    RECENT_VECTOR_CAPACITY = 20_000
    # Ingestion rebuilds once this many vectors wait outside the built index:
    # half the capacity, so one failed build still leaves room.
    INDEX_REBUILD_AT = 10_000
    # Memory no index allocation may plan away from the rest of the system.
    MEMORY_HEADROOM_MB = 256

    # How many conversation turns the summariser feeds the draft model in one
    # window. Measured in turns, not tokens: the window is addressed by
    # sequence_number, and _WINDOW_OVERLAP_CHUNKS extends it further back.
    MAIN_MODEL_CONTEXT_WINDOW_TURNS = 100

    # Placeholder until the main model is chosen.
    MAIN_MODEL_CONTEXT_WINDOW_TOKENS = 8192

    SNAPSHOT_EVERY_N_TURNS = 20

    SYSTEM_PROMPTS_FILE = ABS_PATH / "system_prompts.yml"

    # Estimated tokens: past the soft limit a query is condensed, past the hard one refused.
    QUERY_SOFT_LIMIT_TOKENS = 2_000
    QUERY_HARD_LIMIT_TOKENS = 24_000
    # Holds the hard limit plus prompts; the full 128k would need ~5 GB of KV cache.
    QUERY_DECOMPOSITION_CONTEXT_WINDOW = 32_768
    # One sub-query per worker planner.
    MAX_SUB_QUERIES = 4

    CONVERSATION = Path(
        os.path.join(
            DATA_DIR,
            "memory",
            "topic_pool",
            "project_pool",
            "conversation_pool",
            "conversation_vectors",
            "Conversation",
        )
    )
    PROJECT = Path(os.path.join(DATA_DIR, "memory", "topic_pool", "project_pool"))
    MEMORY_DB = Path(DATA_DIR) / "memory" / "memory_layer" / "memory_layer.db"
    VECTOR_DIMENSIONS = 128

    # Saved llama.cpp inference state, one checkpoint per conversation. A cache:
    # deleting the directory loses only the time to recompute it.
    INFERENCE_CACHE_DIR = Path(DATA_DIR) / "inference_cache"
    INFERENCE_CACHE_MAX_BYTES = 2 * GIB

    # Where a running `start`/`serve`/`-q` holds its single-instance lock and
    # publishes its state for `main.py status` in another process.
    RUN_DIR = Path(DATA_DIR) / "run"

    THINKING_MAX_WORKERS = 2
    THINKING_MAX_JOB_RETRIES = 1
    THINKING_MAX_RUNTIME_RETRIES = 1
    THINKING_MAX_PLAN_REVISIONS = 1
    THINKING_MAX_PLANNED_JOBS = 3
    THINKING_DEADLINE_SECONDS = 900.0

    GENERATION_MAX_TOKENS = 1_200
    GENERATION_CONTEXT_TOKENS = 12_000
    # Turns of this conversation replayed verbatim into the answer prompt.
    GENERATION_HISTORY_TURNS = 6

    # Everything ingestion writes for the document corpus may occupy at most
    # this many bytes (binary GiB); None disables the quota, not disk checks.
    # MAX_INGESTION_STORAGE_BYTES in the environment overrides it.
    MAX_INGESTION_STORAGE_BYTES = 1 * GIB
    # Disk that must stay free whatever the quota says.
    INGESTION_DISK_RESERVE_BYTES = 512 * 1024 * 1024
    CORPUS_DIR = Path(DATA_DIR) / "corpus"

    # KSV acquisition. What acquired material may occupy in the corpus in all
    # (inside MAX_INGESTION_STORAGE_BYTES, never beside it), and what a single
    # acquisition may download: search pages and documents. None is unlimited;
    # MAX_ACQUISITION_STORAGE_BYTES and MAX_ACQUISITION_DOWNLOAD_BYTES in the
    # environment override them.
    MAX_ACQUISITION_STORAGE_BYTES = 256 * 1024 * 1024
    MAX_ACQUISITION_DOWNLOAD_BYTES = 16 * 1024 * 1024

    # Vector ids are stored in signed 64-bit columns (SQLite INTEGER, Postgres
    # bigint). Hash-derived ids are unsigned and overflow both for roughly half
    # of all inputs, so they are masked into the non-negative signed range.
    VECTOR_ID_MASK = (1 << 63) - 1


def parse_storage_limit(value) -> int | None:
    """A byte count, or None for unlimited; anything else is refused."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("", "null", "none", "unlimited"):
            return None
        try:
            value = int(text)
        except ValueError as error:
            raise ValueError(f"storage limit {value!r} is not a whole number of bytes") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"storage limit must be an int or None, got {value!r}")
    if value <= 0:
        raise ValueError(f"storage limit must be positive, got {value}")
    return value


for _limit in (
    "MAX_INGESTION_STORAGE_BYTES",
    "MAX_ACQUISITION_STORAGE_BYTES",
    "MAX_ACQUISITION_DOWNLOAD_BYTES",
):
    setattr(Config, _limit, parse_storage_limit(os.environ.get(_limit, getattr(Config, _limit))))

config = Config()

__all__ = [
    "Config",
    "config",
    "configure_logging",
    "get_logger",
    "log_context",
    "log_timing",
    "new_correlation_id",
    "parse_storage_limit",
]
