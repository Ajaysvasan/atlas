# Atlas pipeline (`backend/app`)

The Python half of Atlas, where all the intelligence lives. It ingests documents into a
pgvector + DiskANN index and keeps conversation memory organised as Topic → Project →
Conversation → Snapshot. It answers a query end to end on one local model:
decomposition → memory → retrieval and sufficiency checks → acquisition → thinking →
generation → persistence. The model is Qwen2.5-3B-Instruct (GGUF, run by llama.cpp).

It runs two ways:

- **as a CLI**, for development and single-user use;
- **as a gRPC server** (`main.py serve`), behind the Java REST API in [`../atlas`](../atlas/README.md),
  which handles accounts, JWTs and uploads. The contract is
  [`../proto/atlas/v1/atlas_service.proto`](../proto/atlas/v1/atlas_service.proto).

[← Project README](../../README.md) · [Java API](../atlas/README.md) ·
[Web frontend](../../frontend/web_frontend/README.md) · [Backend reference](../BACKEND.md)

## Setup

Requires [uv](https://docs.astral.sh/uv/) and PostgreSQL with pgvector. uv installs
**Python 3.11**, which is a hard requirement: `diskannpy` publishes no wheels for newer
versions.

```bash
cp .env.example .env                       # DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
createdb Vectors && psql -d Vectors -c 'CREATE EXTENSION IF NOT EXISTS vector;'
uv sync --all-extras                       # .venv from uv.lock
uv pip install llama-cpp-python            # CPU; NVIDIA GPU: CMAKE_ARGS=-DGGML_CUDA=on uv pip install llama-cpp-python
uv run --no-sync python download_models/download_draft_model.py   # ~2 GB into models/draft_model/
uv run --no-sync python scripts/smoke.py   # optional: one real round trip through PostgreSQL
```

`.env` holds your database password and is ignored by git. After the first install, always
use **`uv run --no-sync`**: a plain `uv sync` or `uv run` removes `llama-cpp-python`,
because it isn't in the lock file (bug 9.9). If the model download stalls, set
`HF_HUB_DISABLE_XET=1`.

Optional: `uv run --no-sync python scripts/build_corpus.py` ingests 58 documentation pages
from postgresql.org, sqlite.org and cassandra.apache.org, within the storage quota. Check
them with `scripts/validate_corpus.py`.

## Running it

```bash
uv run --no-sync python main.py start              # interactive CLI (the default with no command)
uv run --no-sync python main.py --user NAME start  # whose conversations (default $ATLAS_USER or your login)
uv run --no-sync python main.py -q "question"      # one question in a new conversation, then exit
uv run --no-sync python main.py --no-acquire start # never fetch from trusted sources
uv run --no-sync python main.py serve              # gRPC server for the Java API
```

- **`serve`:** needs `ATLAS_GRPC_SERVICE_TOKEN` (16+ characters), with the same value the
  Java API was started with. Keep it in one secrets file that both terminals `source`; see
  the [project README](../../README.md#5-shared-secrets-once).
  - It listens on `127.0.0.1:50051` once the model, embedder, reranker and index have
    loaded.
  - It refuses to bind anywhere but loopback without TLS.
- **One at a time:** only one of `start`, `serve` and `-q` runs per data directory. Each
  holds `data/run/atlas.lock`.

From another terminal, these monitors read the running service's state and write nothing:

```bash
uv run --no-sync python main.py status              # what is loaded, counters, every component's health
uv run --no-sync python main.py status --time       # stage timings of the last query
uv run --no-sync python main.py status --retries    # planner and critic retries of the last query
uv run --no-sync python main.py --user NAME conversations
```

At the `Atlas >` prompt everything is a question, except these commands: `switch`,
`switch <n|id-prefix>`, `new`, `status`, `help` and `exit`.

`--user`, `--no-acquire`, `-q` and the logging flags (`--verbose`, `--log-json`,
`--log-file PATH`) go **before** the command: `main.py --user NAME start` works, but
`main.py start --user NAME` is rejected. The `LOG_*` variables override the logging flags.
The CLI is described in [`cli/README.md`](cli/README.md).

## Layout

```text
main.py                 entry point: flags, logging, then the CLI, `serve` or a monitor
config.py               every constant; re-exports the logging helpers
logging_setup.py        logging for the whole application
pipeline/               AtlasPipeline.ask (the stages in order), Components bootstrap, uploads, acquisition
query_decomposition/    mode, topics and sub-queries; the exchange log (conversation_table)
memory/                 topics, projects, conversations, snapshots; the conversation registry
retrieval_layer/        hybrid search, fusion, re-ranking, diversity, hydration, context assembly
knowledge_sufficiency/  KSV (keep only passages that support an answer) and trusted-source discovery
thinking_layer/         planner, job pool and scheduler, workers, critics
generation_layer/       budgeted context assembly and the final answer
inference/              one shared llama.cpp model per process; KV-cache checkpoints
data_layer/             ingestion: files → text → chunks → vectors → DiskANN generations
storage/                PostgreSQL connections, SQLite setup, timestamps
grpc_server/            the gRPC adapter: service token, one-at-a-time gate, uploads, settings
observability/          request telemetry, health checks, the published service state
cli/                    the interactive loop, commands, rendering, monitors
scripts/                corpus build, smoke tests, gRPC code generation, end-to-end driver
download_models/        model download and llama.cpp install helpers
test/                   the suite, one directory per layer
docs/                   API reference
```

Directories with a **README.md** (`cli`, `data_layer`, `memory`, `knowledge_sufficiency`,
`storage`, `download_models`) explain what the module does, how data flows through it, and
why it was built that way. `docs/` is the API reference. Code comments are only for logic
that reads wrong without one. [`CLAUDE.md`](CLAUDE.md) holds the most complete architecture
notes for every layer.

### Where data lives

| Store | Path | Holds |
|---|---|---|
| PostgreSQL + pgvector | DB from `.env` | Every vector; `conversations` (the registry: id, owner, name); `conversation_table` (the exchange log) |
| DiskANN index | `data/disk_ann_index/` | Built index generations plus an in-memory index of recent ingests |
| SQLite (chunks) | `data/hierarchical_db/` | Documents, sections, chunks, vector labels, document owners |
| SQLite (memory) | `data/memory/memory_layer/memory_layer.db` | Topics, projects, snapshots, conversation turns, KV-checkpoint records |
| Corpus | `data/corpus/` | Downloaded and acquired sources, users' uploads, `manifest.jsonl` |
| KV checkpoints | `data/inference_cache/` | Saved model state per conversation. Safe to delete; it is recomputed. |
| Service state | `data/run/` | `atlas.lock` and `atlas.json`, read by `status` |
| Model | `models/draft_model/` | `qwen2.5-3b-instruct-q4_k_m.gguf` |

## Tests

```bash
uv run --no-sync pytest test/ -m "not real_model and not live"   # fast (~30 s); no database or model needed
uv run --no-sync pytest test/grpc_testing/ -v                    # gRPC server over loopback
uv run --no-sync pytest test/ -m real_model                      # the real model; ~5.5 min on a CPU
uv run --no-sync pytest test/live_testing/ -v                    # real PostgreSQL, throwaway ids
uv run --no-sync python scripts/generate_grpc.py --check         # are the gRPC bindings current?
```

- **Expected result:** the full run has **2348 passed**: 2334 without the model, plus 14
  `real_model` tests.
- **Isolation:** the root `conftest.py` keeps the default suite off your real data. It
  redirects every data path, swaps PostgreSQL for in-memory stand-ins, and refuses database
  connections in any test not marked `live`.
- **Self-skipping:** the `live` and `real_model` suites skip themselves when PostgreSQL or
  the model isn't available. `-rs` shows why.
- **Full stack:** `scripts/backend_e2e.py` drives the running Java API end to end. It writes
  throwaway `e2e_*` users.

See [`test/HOW_TO.md`](test/HOW_TO.md) for more.

## Why `config.py` re-exports logging instead of implementing it

`logging_setup` must not import `config`. Otherwise any module that imports `config` in
order to log would create an import cycle. The dependency runs one way only,
`config -> logging_setup`, and `config` re-exports `get_logger`. That keeps working the
`from config import get_logger` that every module already uses.

## Status

The whole pipeline is built and tested: ingestion, memory, retrieval, KSV, acquisition,
thinking and generation. The CLI and `-q` run it end to end, and `serve` exposes it to the
Java API with per-user isolation of conversations, projects and uploads.

Open issues (details in [`bugs.md`](bugs.md) and [`production_impact_report.md`](production_impact_report.md)):

- `uv sync` removes `llama-cpp-python`, and `install_llama_cpp.py` can't reinstall it
  (bug 9.9).
- There is no deletion of documents or conversations yet.
- The storage quota is one pool shared by every user.
- Over gRPC, a query that has started runs to completion even if its caller has gone away.
- Python and MDN docs can only be acquired by explicit URL.
