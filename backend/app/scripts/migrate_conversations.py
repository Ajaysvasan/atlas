"""Move the conversation registry from SQLite (memory_layer.db) to PostgreSQL and verify every row.

    uv run python scripts/migrate_conversations.py

Atlas also does this on start; this script does it on its own and prints the
evidence. Safe to run repeatedly: once migrated there is nothing left to move.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config, configure_logging  # noqa: E402


def main() -> int:
    configure_logging(log_file=Config.LOG_FILE)
    from memory.conversation_registry import (
        SQLITE_TABLE,
        migrate_from_sqlite,
        open_conversation_registry,
    )

    path = Path(Config.MEMORY_DB)
    before = []
    if path.exists():
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            if db.execute("select 1 from sqlite_master where name = ?", (SQLITE_TABLE,)).fetchone():
                before = db.execute(f"select conversation_id, user_id, conversation_name, created_at "
                                    f"from {SQLITE_TABLE}").fetchall()
    print(f"SQLite {path.name}: {len(before)} conversation row(s) to migrate")

    registry = open_conversation_registry()
    try:
        report = migrate_from_sqlite(registry, path)
        if report is None:
            print("Nothing to migrate (already done, or never kept in SQLite).")
        else:
            print(f"Copied {report.copied}, already present {report.already_present}, renamed {list(report.renamed)}")
        problems = 0
        for conversation_id, user_id, name, created in before:
            stored = registry.get(conversation_id, user_id)
            same = (stored is not None and stored.conversation_name == name
                    and datetime.fromisoformat(stored.created_at) == datetime.fromisoformat(created))
            renamed = report is not None and any(r[0] == conversation_id for r in report.renamed)
            problems += not (same or renamed)
            print(f"  {'ok ' if same else 'RENAMED' if renamed else 'MISSING'} {conversation_id} "
                  f"{user_id} {name!r} {created}")
        print(f"PostgreSQL conversations: {registry.count()} row(s)")
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            tables = sorted(r[0] for r in db.execute(
                "select name from sqlite_master where type='table' and name like 'conversations%'"))
        print(f"SQLite tables now: {tables}")
        return 1 if problems else 0
    finally:
        registry.close()


if __name__ == "__main__":
    sys.exit(main())
