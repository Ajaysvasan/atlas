"""Bounded checks of each component. None generates text with a model, and none needs the service's process.

What only the running service knows (what is loaded, its counters) comes in as
`facts` — `service_state.service_facts()` in that process, or what it published
for a monitor in another one; None when Atlas is not running.
"""

import os
import sqlite3
import time
from enum import Enum
from pathlib import Path
from typing import Callable, List, NamedTuple

from config import Config, get_logger

logger = get_logger(__name__)

PG_TIMEOUT_SECONDS = 3


class Health(Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_INITIALIZED = "NOT_INITIALIZED"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNKNOWN = "UNKNOWN"


class ComponentHealth(NamedTuple):
    name: str
    status: Health
    detail: str
    milliseconds: float


def _check(name: str, probe: Callable[[], tuple]) -> ComponentHealth:
    started = time.monotonic()
    try:
        status, detail = probe()
    except Exception as error:
        status, detail = Health.UNAVAILABLE, f"{type(error).__name__}: {_safe(error)}"
    return ComponentHealth(name, status, detail, (time.monotonic() - started) * 1000.0)


def _safe(error: BaseException) -> str:
    """An error message with anything that looks like a credential removed."""
    text = str(error).splitlines()[0] if str(error) else ""
    password = os.getenv("DB_PASSWORD")
    if password:
        text = text.replace(password, "***")
    return text[:200]


def _postgres():
    import psycopg
    from dotenv import load_dotenv

    load_dotenv()
    names = ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        return None, f"missing {', '.join(missing)} in .env"
    return psycopg.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"), connect_timeout=PG_TIMEOUT_SECONDS,
    ), None


def _sqlite_tables(path: Path) -> set:
    if not Path(path).exists():
        return set()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        return {row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    finally:
        connection.close()


def _sqlite_count(path: Path, table: str) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        return connection.execute(f'select count(*) from "{table}"').fetchone()[0]
    finally:
        connection.close()


NOT_RUNNING = "Atlas is not running"


def check_all(components) -> List[ComponentHealth]:
    """Every component's health from inside the running process; `components` is bootstrap.Components."""
    from observability.service_state import service_facts

    return check(service_facts(components))


def _sqlite_sum(path: Path, table: str, column: str) -> tuple:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    try:
        return connection.execute(f'select count(*), coalesce(sum("{column}"), 0) from "{table}"').fetchone()
    finally:
        connection.close()


def check(facts: dict | None, absent: str = NOT_RUNNING) -> List[ComponentHealth]:
    """Every component's health. Reads only; with `facts` None, service-held parts report NOT_INITIALIZED, saying `absent`."""
    results: List[ComponentHealth] = []
    pg_state = {}
    running = facts is not None

    def add(name, probe):
        results.append(_check(name, probe))

    def configuration():
        from config import parse_storage_limit

        parse_storage_limit(Config.MAX_INGESTION_STORAGE_BYTES)
        data = Path(Config.DATA_DIR)
        if not os.access(data if data.exists() else data.parent, os.W_OK):
            return Health.DEGRADED, f"{data} is not writable"
        limit = Config.MAX_INGESTION_STORAGE_BYTES
        return Health.HEALTHY, f"loaded; ingestion quota {'unlimited' if limit is None else f'{limit:,} bytes'}"

    def prompts():
        from prompt_registry import load_section
        from query_decomposition.prompts import load_prompts

        load_prompts()
        load_section("thinking_layer", ("planner", "worker", "stitch", "job_critic", "plan_critic"))
        load_section("generation_layer", ("answer",))
        load_section("naming", ("conversation", "project"))
        return Health.HEALTHY, f"all sections present in {Path(Config.SYSTEM_PROMPTS_FILE).name}"

    def model(path, published):
        def probe():
            import importlib.util

            # find_spec, not import: a monitor need not load llama.cpp's library to know it is there.
            if importlib.util.find_spec("llama_cpp") is None:
                return Health.NOT_CONFIGURED, "llama-cpp-python is not installed"
            name = Path(path).name
            if not Path(path).exists():
                return Health.NOT_CONFIGURED, f"model file missing: {name}"
            if not running:
                return Health.NOT_INITIALIZED, f"{name} present; loaded by `start`/`serve` ({absent})"
            if published.get("load_error"):
                return Health.UNAVAILABLE, f"failed to load: {published['load_error'][:120]}"
            if not published.get("loaded"):
                return Health.NOT_INITIALIZED, f"{name} present; loads on first use"
            return Health.HEALTHY, (
                f"{name} resident, n_ctx={published.get('n_ctx')}, "
                f"{published.get('calls', 0)} call(s), {published.get('failures', 0)} failure(s)"
            )
        return probe

    def draft_model():
        from inference.shared_model import ModelSpec

        published = facts["draft"] if running else {}
        return model(published.get("path") or ModelSpec.draft().path, published)()

    def main_model():
        published = facts.get("main") if running else None
        if published is None and (running or not Config.MAIN_MODEL_PATH):
            return Health.HEALTHY, "served by the draft model (MAIN_MODEL_PATH not set)"
        published = published or {}
        return model(published.get("path") or Config.MAIN_MODEL_PATH, published)()

    def postgres():
        connection, problem = _postgres()
        if connection is None:
            pg_state["error"] = problem
            return Health.NOT_CONFIGURED, problem
        try:
            with connection.cursor() as cursor:
                cursor.execute("select to_regclass('vectors'), to_regclass('conversation_table'), "
                               "to_regclass('conversations')")
                vectors_table, exchanges_table, conversations_table = cursor.fetchone()
                global_rows = total_rows = 0
                if vectors_table:
                    cursor.execute("select count(*) filter (where project_id = %s), count(*) from vectors",
                                   (Config.GLOBAL_VECTOR_SCOPE,))
                    global_rows, total_rows = cursor.fetchone()
                exchanges = 0
                if exchanges_table:
                    cursor.execute("select count(*) from conversation_table")
                    exchanges = cursor.fetchone()[0]
                conversations = None
                if conversations_table:
                    cursor.execute("select count(*) from conversations")
                    conversations = cursor.fetchone()[0]
            pg_state.update(ok=True, global_rows=global_rows, total_rows=total_rows,
                            exchanges=exchanges, exchanges_table=bool(exchanges_table),
                            conversations=conversations)
            return Health.HEALTHY, f"reachable; {total_rows} vector row(s), {exchanges} exchange(s)"
        finally:
            connection.close()

    def from_pg(name_ok, detail):
        def probe():
            if "ok" not in pg_state:
                return Health.UNAVAILABLE, f"PostgreSQL: {pg_state.get('error') or 'unreachable'}"
            return Health.HEALTHY, detail()
        return probe

    def memory_db():
        path = Path(Config.MEMORY_DB)
        tables = _sqlite_tables(path)
        if not tables:
            return Health.NOT_INITIALIZED, "no memory database yet; created on first write"
        needed = {"memory_mapping_table", "full_conversation"}
        missing = needed - tables
        if missing:
            return Health.DEGRADED, f"missing table(s): {', '.join(sorted(missing))}"
        mapped = _sqlite_count(path, "memory_mapping_table")
        return Health.HEALTHY, f"{len(tables)} table(s); {mapped} mapped conversation(s)"

    def table_probe(table, what):
        def probe():
            path = Path(Config.MEMORY_DB)
            tables = _sqlite_tables(path)
            if table not in tables:
                return Health.NOT_INITIALIZED, f"no {what} yet"
            return Health.HEALTHY, f"{_sqlite_count(path, table)} {what}"
        return probe

    def local_retrieval():
        if "ok" not in pg_state:
            return Health.DEGRADED, "turn vectors unreachable; keyword recall only"
        return Health.HEALTHY, "turn keyword and vector search available"

    def chunk_store():
        path = Path(Config.DB_PATH)
        if not path.exists():
            return Health.HEALTHY, "empty corpus (no chunk store yet)"
        tables = _sqlite_tables(path)
        counts = {t: _sqlite_count(path, t) for t in ("Documents", "Chunks", "RecursiveChunks", "vector_meta_data")
                  if t in tables}
        return Health.HEALTHY, ", ".join(f"{k} {v}" for k, v in counts.items()) or "no tables yet"

    def vector_index():
        from data_layer.vector_db_manager.index_generations import IndexGenerations

        generations = IndexGenerations(Config.INDEX_PATH)
        current = generations.current()
        built = running and bool(facts["retriever"]["built"])
        if current is None:
            labels = _sqlite_count(Path(Config.DB_PATH), "vector_meta_data") if "vector_meta_data" in _sqlite_tables(Path(Config.DB_PATH)) else 0
            if labels:
                return Health.DEGRADED, f"no built generation; {labels} vector(s) searched from the recent index only"
            return Health.HEALTHY, "no generation built (empty corpus)"
        state = "open" if built else ("opens on first query" if running else f"opened by `start` ({absent})")
        return Health.HEALTHY, f"generation {current} through label {generations.through()}; {state}"

    def global_retrieval():
        if not running:
            return Health.NOT_INITIALIZED, f"built by `start`/`serve` ({absent})"
        retriever = facts["retriever"]
        if retriever.get("error"):
            return Health.UNAVAILABLE, f"could not start: {retriever['error'][:120]}"
        if not retriever.get("built"):
            return Health.NOT_INITIALIZED, "builds on first query (loads reranker and index)"
        generation = retriever.get("generation")
        return Health.HEALTHY, f"ready; searching generation {generation or 'none (recent vectors only)'}"

    def ksv():
        from knowledge_sufficiency.thresholds import PARTIAL_FLOOR, SUFFICIENT_FLOOR

        return Health.HEALTHY, f"floors: sufficient {SUFFICIENT_FLOOR}, partial {PARTIAL_FLOOR}"

    def acquisition():
        from data_layer.ingestion.corpus_ingestion import acquired_bytes
        from knowledge_sufficiency.sources import DEFAULT_SOURCES

        if running and not facts.get("acquisition"):
            return Health.NOT_CONFIGURED, "disabled (--no-acquire)"
        searchable = sum(1 for s in DEFAULT_SOURCES if s.searchable)
        mapped = sum(1 for s in DEFAULT_SOURCES if s.has_sitemap)
        limit = Config.MAX_ACQUISITION_STORAGE_BYTES
        used = acquired_bytes(Path(Config.CORPUS_DIR))
        budget = Config.MAX_ACQUISITION_DOWNLOAD_BYTES
        detail = (f"{len(DEFAULT_SOURCES)} trusted source(s), {searchable} searchable, "
                  f"{mapped} with a sitemap; acquired "
                  f"{used:,} of {'unlimited' if limit is None else f'{limit:,}'} bytes; download budget "
                  f"{'unlimited' if budget is None else f'{budget:,}'} bytes per acquisition; network not probed")
        if limit is not None and used >= limit:
            return Health.DEGRADED, "acquisition limit reached; " + detail
        if not running:
            return Health.NOT_INITIALIZED, f"{absent}; " + detail
        return Health.HEALTHY, detail

    thinking_self_test = {}

    def thinking():
        from thinking_layer.selftest import run_self_test

        thinking_self_test.update(run_self_test())
        failed = [k for k, ok in thinking_self_test.items() if not ok]
        if failed:
            return Health.UNAVAILABLE, f"self-test failed: {', '.join(failed)}"
        return Health.HEALTHY, "orchestration self-test passed (stub model)"

    def thinking_part(part):
        def probe():
            if not thinking_self_test:
                return Health.UNKNOWN, "self-test did not run"
            ok = thinking_self_test.get(part)
            return (Health.HEALTHY, "exercised by the self-test") if ok else (Health.UNAVAILABLE, "self-test failed")
        return probe

    def generation():
        return Health.HEALTHY, f"prompt loaded; max {Config.GENERATION_MAX_TOKENS} tokens per answer"

    def kv():
        directory = Path(Config.INFERENCE_CACHE_DIR)
        if directory.exists() and not os.access(directory, os.W_OK):
            return Health.DEGRADED, f"{directory} not writable; answers recompute every turn"
        path = Path(Config.MEMORY_DB)
        count, usage = (_sqlite_sum(path, "inference_checkpoints", "size_bytes")
                        if "inference_checkpoints" in _sqlite_tables(path) else (0, 0))
        limit = Config.INFERENCE_CACHE_MAX_BYTES
        detail = f"{count} checkpoint(s), {usage / 1e6:.1f} of " + (
            "unlimited" if limit is None else f"{limit / 1e6:.0f} MB")
        if not running:
            return Health.HEALTHY, f"{detail}; session counters while Atlas runs"
        sessions = facts["kv"]
        return Health.HEALTHY, (
            f"{detail}; restores {sessions['restores']}, misses {sessions['misses']}, "
            f"invalidated {sessions['invalidations']}, save failures {sessions['save_failures']}"
        )

    def ingestion():
        from data_layer.ingestion.quota import IngestionQuota, postgres_corpus_bytes

        quota = IngestionQuota(postgres_usage=postgres_corpus_bytes if "ok" in pg_state else None)
        usage = quota.usage()
        free = quota.free_disk_bytes()
        limit = quota.limit_bytes
        pg = "unknown" if usage.postgres_bytes is None else f"{usage.postgres_bytes:,}"
        detail = (f"corpus {usage.total:,} bytes (PostgreSQL share {pg}) of "
                  f"{'unlimited' if limit is None else f'{limit:,}'}; disk free {free / 1e9:.1f} GB")
        if limit is not None and usage.total >= limit:
            return Health.DEGRADED, "quota full; " + detail
        if free < quota.disk_reserve_bytes:
            return Health.DEGRADED, "disk below reserve; " + detail
        return Health.HEALTHY, detail

    add("Configuration", configuration)
    add("Prompt registry", prompts)
    add("Draft model", draft_model)
    add("Main model", main_model)
    add("PostgreSQL", postgres)
    add("Query Decomposition", from_pg("exchanges", lambda: (
        "exchange log ready" if pg_state.get("exchanges_table") else "exchange log created on first query")))
    add("Memory database (SQLite)", memory_db)
    def memory_manager():
        status, detail = memory_db()
        if status is not Health.HEALTHY and status is not Health.NOT_INITIALIZED:
            return status, detail
        if not running:
            embedder = f"embedder loaded by `start` ({absent})"
        elif facts.get("embedder_resident"):
            embedder = "embedder resident"
        else:
            embedder = "embedder loads on first query"
        return Health.HEALTHY, f"routing, recall and storage ready; {embedder}"

    add("Memory Manager", memory_manager)
    add("Conversation persistence", table_probe("full_conversation", "stored turn(s)"))
    def registry():
        if "ok" not in pg_state:
            return Health.UNAVAILABLE, f"PostgreSQL: {pg_state.get('error') or 'unreachable'}"
        count = pg_state.get("conversations")
        if count is None:
            return Health.NOT_INITIALIZED, "PostgreSQL `conversations` is created when Atlas starts"
        return Health.HEALTHY, f"PostgreSQL `conversations`: {count} conversation(s)"

    add("Conversation registry", registry)
    add("Memory Mapping", table_probe("memory_mapping_table", "mapped conversation(s)"))
    add("Snapshot subsystem", table_probe("cumulative_vector_meta_data", "snapshot(s)"))
    add("Memory-Local Retrieval", local_retrieval)
    add("Vector storage (pgvector)", from_pg("vectors", lambda: f"{pg_state['global_rows']} corpus vector(s)"))
    add("Chunk storage", chunk_store)
    add("Vector index (DiskANN)", vector_index)
    add("Global Retrieval", global_retrieval)
    add("KSV", ksv)
    add("Knowledge Acquisition", acquisition)
    add("Thinking Layer", thinking)
    for part in ("Planner", "Scheduler", "Job Pool", "Worker Pool", "Accumulator", "Critic"):
        add(part, thinking_part(part))
    add("Generation Layer", generation)
    add("KV Session Manager", kv)
    add("Ingestion subsystem", ingestion)
    return results
