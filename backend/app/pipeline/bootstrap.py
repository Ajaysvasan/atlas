"""Builds the pipeline from the real components, each loaded once per process."""

from pathlib import Path
from typing import Dict

from config import Config, get_logger

logger = get_logger(__name__)


class ProvenanceIndex:
    """Source URL by passage parent id, reread when the manifest or the chunk store changes."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self.__seen = None
        self.__map: Dict[str, str] = {}

    def __call__(self) -> Dict[str, str]:
        from data_layer.ingestion.corpus_ingestion import MANIFEST, source_urls

        manifest = Path(Config.CORPUS_DIR if self.directory is None else self.directory) / MANIFEST
        try:
            stamp = (manifest.stat().st_mtime, Path(Config.DB_PATH).stat().st_mtime)
        except FileNotFoundError:
            return {}
        if stamp != self.__seen:
            self.__map = source_urls(manifest.parent)
            self.__seen = stamp
        return self.__map


class Components:
    """Everything the pipeline and the status report share, so status inspects what actually runs."""

    def __init__(self, *, acquisition: bool = True) -> None:
        from data_layer.ingestion.corpus_ingestion import ingest_files
        from generation_layer.context import load_generation_prompts
        from generation_layer.generator import Generator
        from inference.checkpoints import CheckpointStore
        from inference.runtime import ReasoningModel, shared_runtime
        from inference.sessions import KVSessionManager
        from knowledge_sufficiency.ksv_manager import KSVManager
        from memory import conversation_registry
        from memory.memory_database import MemoryDatabase
        from memory.memory_manager import shared_models
        from observability.telemetry import Telemetry
        from pipeline.acquisition import Acquirer
        from pipeline.atlas import RetrieverHandle
        from pipeline.naming import Namer
        from thinking_layer.orchestrator import ThinkingOrchestrator

        self.runtime = shared_runtime()
        self.database = MemoryDatabase.shared()
        self.models = shared_models()
        self.ksv = KSVManager()
        # In PostgreSQL beside the exchange log; rows an earlier version kept in
        # SQLite are moved there on first start and the SQLite table kept as a backup.
        self.registry = conversation_registry.open_conversation_registry()
        migrated = conversation_registry.migrate_from_sqlite(self.registry, Config.MEMORY_DB)
        if migrated is not None:
            logger.info("Conversation registry migrated to PostgreSQL: %s", migrated)
        # Projects created before routing was scoped to their owner get one (bug 10.1).
        from memory.topic_pool.project_pool.project_data_repo.project_meta_data import (
            backfill_project_owners,
        )

        filled, mixed = backfill_project_owners(self.database)
        if filled or mixed:
            logger.info("Project owners backfilled: %d; left ownerless (mixed users): %d", filled, len(mixed))
        self.checkpoints = CheckpointStore(database=self.database)
        removed = self.checkpoints.sweep()
        if removed:
            logger.info("Removed %d stale checkpoint file(s) or row(s) at startup", removed)
        self.sessions = KVSessionManager(self.runtime.main, self.checkpoints)
        self.thinking = ThinkingOrchestrator(ReasoningModel(self.runtime))
        self.generator = Generator(self.sessions)
        self.generation_prompts = load_generation_prompts()
        self.namer = Namer(self.runtime.draft)
        self.telemetry = Telemetry()
        self.provenance = ProvenanceIndex()
        self.retriever = RetrieverHandle(self.__build_retriever)
        self.acquirer = (
            # build_index=False: acquired vectors join the recent index; a full
            # rebuild per acquisition would cost the whole corpus every time.
            Acquirer(self.ksv, self.models.preparation.encode,
                     ingest=lambda paths, **hooks: ingest_files(paths, build_index=False, **hooks))
            if acquisition else None
        )
        self.decomposer = None

    def __build_retriever(self):
        from retrieval_layer.retriever import Retriever

        return Retriever(
            self.models.settings,
            preparation=self.models.preparation,
            reranker=self.models.reranker,
        )

    def warm_up(self) -> list:
        """Load what the first query would otherwise wait for; returns what could not be loaded."""
        problems = []
        self.runtime.draft.ensure_loaded()
        if self.runtime.separate_main:
            self.runtime.main.ensure_loaded()
        self.models.embedder  # noqa: B018 - loading is the point
        try:
            # One search opens the vector index and loads the reranker.
            self.retriever.get().retrieve("database")
        except Exception as error:
            logger.warning("Global retrieval is not available yet: %s", error)
            problems.append(f"global retrieval: {error}")
        return problems

    def pipeline(self):
        """The pipeline; opens the PostgreSQL exchange log, so it fails here if that is down."""
        from pipeline.atlas import AtlasPipeline
        from query_decomposition.decompose import Decompose

        if self.decomposer is None:
            self.decomposer = Decompose(database=self.database)
        return AtlasPipeline(
            decomposer=self.decomposer,
            memory_models=self.models,
            ksv=self.ksv,
            retriever=self.retriever,
            thinking=self.thinking,
            generator=self.generator,
            generation_prompts=self.generation_prompts,
            registry=self.registry,
            namer=self.namer,
            database=self.database,
            acquirer=self.acquirer,
            telemetry=self.telemetry,
            provenance=self.provenance,
            model_calls=self.model_calls,
        )

    def model_calls(self) -> Dict[str, int]:
        """Calls by purpose across every resident model, for per-request deltas."""
        calls = dict(self.runtime.draft.stats().calls)
        if self.runtime.separate_main:
            for purpose, count in self.runtime.main.stats().calls.items():
                calls[f"main:{purpose}"] = count
        return calls

    def close(self) -> None:
        self.retriever.close()
        self.registry.close()
        if self.decomposer is not None:
            self.decomposer.close()
        self.runtime.close()
