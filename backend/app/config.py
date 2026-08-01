import logging
import os
from pathlib import Path

import numpy as np


class Config:
    """
    Configuration class for the backend application.
    Modify or extend these attributes as needed for your final year project.
    """

    APP_NAME = "Final Year Project Backend"
    DATASET_PATH = Path("dataset").resolve().parent / "dataset"
    DEBUG = False
    LOG_FILE = "log/app.log"
    ABS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ABS_PATH, "data")
    DB_PATH = os.path.join(DATA_DIR, "hierarchical_db")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = None
    INDEX_PATH = os.path.join(DATA_DIR, "disk_ann_index")
    DISTANCE_METRIC = "l2"
    VECTOR_DTYPE = np.float32
    EMBEDDING_DIMENSIONS = 128
    MAX_VECTORS = 1000000
    COMPLEXITY = 100
    GRAPH_DEGREE = 120
    NUM_THREADS = 4
    K_NEIGHBORS = 9
    CONVERSATION_SNAPSHOT_DB = Path(
        os.path.join(
            DATA_DIR,
            "memory",
            "topic_pool",
            "project_pool",
            "conversation_pool",
            "conversation_snapshot.db",
        )
    )

    SUMMARY_VECTOR_DB_PATH = Path(
        os.path.join(
            DATA_DIR,
            "memory",
            "topic_pool",
            "project_pool",
            "conversation_pool",
            "conversation_vectors",
            "summary_vectors",
        )
    )
    CUMMULATIVE_VECTOR_DB_PATH = Path(
        os.path.join(
            DATA_DIR,
            "memory",
            "topic_pool",
            "project_pool",
            "conversation_pool",
            "conversation_vectors",
            "cummulative_vector",
        )
    )
    VECTOR_DIMENSIONS = 128


config = Config()


def get_logger(name: str) -> logging.Logger:
    """
    Retrieves or initializes a logger with file and stream handlers.
    Ensures that the log directory exists and logs are formatted meaningfully.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Ensure log directory exists
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = (
            os.path.join(base_dir, Config.LOG_FILE)
            if not os.path.isabs(Config.LOG_FILE)
            else Config.LOG_FILE
        )
        log_dir = os.path.dirname(log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
        )

        # File Handler (writing to the log directory)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if Config.DEBUG else logging.INFO)
        logger.addHandler(file_handler)

        # Stream Handler (console output)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        logger.setLevel(logging.DEBUG if Config.DEBUG else logging.INFO)
        logger.propagate = False

    return logger
