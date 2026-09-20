"""One real round trip: a file in, a vector into PostgreSQL, the vector back out.

No mocks and no fakes. Everything the test suite stubs — psycopg, pgvector, the
embedding model — is real here, which is the only way to find out whether the
stores actually work. Run it from the app directory:

    python scripts/smoke.py

It writes under a throwaway project id and deletes what it wrote before exiting.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, configure_logging  # noqa: E402
from data_layer.ingestion.Chunker.chunker import Chunker  # noqa: E402
from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager  # noqa: E402
from data_layer.ingestion.normalizer.normalizer import NormalizationProfiles  # noqa: E402
from data_layer.ingestion.TextFileProcessor.file_loader import FileLoader  # noqa: E402
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor  # noqa: E402
from data_layer.vector_db_manager.repository.vectorRepository import (  # noqa: E402
    VectorRepository,
)

PROJECT_ID = "__smoke_test__"
DOCUMENT = """# Retrieval Layer

The store exposes search and nothing else. Fusion, reranking and dedup live
above the line, in code that does not know which store it is using.
"""


def preflight() -> list[str]:
    """What has to be true before any of this can work, reported all at once.

    A traceback from psycopg names one missing thing at a time and buries it in
    a stack. These are setup steps, not bugs, and the fix for each is a command.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv(str(Path(__file__).resolve().parent.parent / ".env"))
    problems = []

    try:
        import pgvector.psycopg  # noqa: F401
    except ImportError:
        problems.append("pgvector is not installed:  pip install pgvector")

    import psycopg

    settings = {name: os.getenv(name) for name in
                ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        problems.append(f"missing in .env: {', '.join(missing)}  (see .env.example)")
        return problems

    dsn = dict(user=settings["DB_USER"], password=settings["DB_PASSWORD"],
               host=settings["DB_HOST"], port=settings["DB_PORT"])
    try:
        with psycopg.connect(dbname="postgres", connect_timeout=5, **dsn) as conn:
            if not conn.execute(
                "select 1 from pg_available_extensions where name = 'vector'"
            ).fetchone():
                problems.append(
                    "the pgvector extension is not installed on the PostgreSQL "
                    "server.\n      Fedora:  sudo dnf install pgvector"
                )
            if not conn.execute(
                "select 1 from pg_database where datname = %s", (settings["DB_NAME"],)
            ).fetchone():
                problems.append(
                    f"database {settings['DB_NAME']!r} does not exist:  "
                    f"createdb {settings['DB_NAME']}"
                )
    except psycopg.OperationalError as error:
        problems.append(f"cannot reach PostgreSQL at "
                        f"{settings['DB_HOST']}:{settings['DB_PORT']} - {error}")
    return problems


def main() -> int:
    configure_logging(console_level="WARNING")

    problems = preflight()
    if problems:
        print("Not ready to run:")
        for problem in problems:
            print(f"   - {problem}")
        return 2

    workspace = Path(tempfile.mkdtemp())
    (workspace / "design.md").write_text(DOCUMENT)

    print("1. loading and extracting")
    files = FileLoader().load_files(workspace)
    texts = TextExtractor().extract_all(files)
    print(f"   {len(texts)} file(s)")

    print("2. normalising and chunking")
    documents = NormalizationProfiles.rag_ingestion().normalize_all(texts)
    hierarchical, recursive = Chunker(
        db_path=str(workspace / "chunks.db")
    ).chunk_per_document(documents)
    chunks = hierarchical + recursive
    print(f"   {len(chunks)} chunk(s); first: {chunks[0].chunk[:60]!r}")

    print("3. embedding (loads the real model)")
    embedded = EmbeddingManager().embed(chunks)
    first = embedded[0]
    print(f"   {len(embedded)} vector(s), {first.vector.shape[0]} dims, {first.vector.dtype}")

    print("4. writing to PostgreSQL")
    repository = VectorRepository(PROJECT_ID)
    try:
        repository.batch_insert([e.vector_id for e in embedded], [e.vector for e in embedded])
        print(f"   wrote {len(embedded)} vector(s) under project {PROJECT_ID!r}")

        print("5. reading back")
        returned = repository.search(first.vector_id)
        print(f"   got {type(returned).__name__} {returned.shape} {returned.dtype}")

        matches = bool((returned == first.vector).all())
        print(f"   identical to what went in: {matches}")
        return 0 if matches else 1
    finally:
        repository.batch_delete([e.vector_id for e in embedded])
        repository.close()
        print("   cleaned up")


if __name__ == "__main__":
    raise SystemExit(main())
