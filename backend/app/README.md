# `backend/app`

A RAG backend: ingests documents into a vector index, and keeps conversation
memory organised as Topic → Project → Conversation → Snapshot.

## Layout

```
main.py              entry point: parses flags, configures logging, starts the CLI
config.py            every constant, and the re-exports for logging
logging_setup.py     logging for the whole application
cli/                 the interactive loop
data_layer/          documents: files -> text -> chunks -> vectors
memory/              conversations: turns -> summaries -> snapshots
download_models/     one-off scripts to fetch the draft model
test/                the suite
docs/                API reference
```

Each module directory has a **README.md** describing what it does, how data
flows through it, and why it was built that way. `docs/` is the API reference;
the READMEs are the architecture. Code comments are only for logic that reads
wrong without one.

## Running it

```bash
python main.py                    # interactive CLI
python main.py --verbose          # debug logging, console and file
python main.py --log-json         # one JSON object per line
PYTHONPATH=. pytest test/ -q      # the suite
```

## Why `config.py` re-exports logging instead of implementing it

`logging_setup` must not import `config`, or any module importing `config` in
order to log would close an import cycle. The dependency runs one way only —
`config -> logging_setup` — and `config` re-exports `get_logger` so the
`from config import get_logger` that every module already uses keeps working.

## What works and what does not

| | |
| :--- | :--- |
| Ingestion, chunking, embedding | Built and tested |
| Conversation memory, summaries, snapshots | Built and tested |
| Project registry and query routing | Built and tested |
| Retrieval, the query pipeline, the thinking layer | Not started |
| The CLI query loop | A placeholder (**bugs 2.1, 2.2**) |

Nothing has yet run end to end against a real PostgreSQL — `bugs.md` 5.1
explains why that matters more than the passing test count suggests.
