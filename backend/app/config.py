import os
from pathlib import Path

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
    DATASET_PATH = Path("dataset").resolve().parent / "dataset"
    DEBUG = False
    LOG_FILE = "log/app.log"
    ABS_PATH = Path(__file__).resolve().parent
    DATA_DIR = os.path.join(ABS_PATH, "data")
    DB_PATH = os.path.join(DATA_DIR, "hierarchical_db")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = None
    DRAFT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    DRAFT_MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
    DRAFT_MODEL_PATH = os.path.join(ABS_PATH, "models", "draft_model")
    DRAFT_MODEL_CONTEXT_WINDOW = 131072  # 128K tokens
    INDEX_PATH = os.path.join(DATA_DIR, "disk_ann_index")
    DISTANCE_METRIC = "l2"
    VECTOR_DTYPE = np.float32
    EMBEDDING_DIMENSIONS = 128
    MAX_VECTORS = 1000000
    COMPLEXITY = 100
    GRAPH_DEGREE = 120
    NUM_THREADS = 4
    K_NEIGHBORS = 9

    # How many conversation turns the summariser feeds the draft model in one
    # window. Measured in turns, not tokens: the window is addressed by
    # sequence_number, and _WINDOW_OVERLAP_CHUNKS extends it further back.
    MAIN_MODEL_CONTEXT_WINDOW_TURNS = 100

    # Unsummarised turns that must accumulate before ConversationPoolManager
    # takes a snapshot on its own.
    SNAPSHOT_EVERY_N_TURNS = 20

    PROJECT = Path("")

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
    VECTOR_DIMENSIONS = 128

    # Vector ids are stored in signed 64-bit columns (SQLite INTEGER, Postgres
    # bigint). Hash-derived ids are unsigned and overflow both for roughly half
    # of all inputs, so they are masked into the non-negative signed range.
    VECTOR_ID_MASK = (1 << 63) - 1


config = Config()

__all__ = [
    "Config",
    "config",
    "configure_logging",
    "get_logger",
    "log_context",
    "log_timing",
    "new_correlation_id",
]
