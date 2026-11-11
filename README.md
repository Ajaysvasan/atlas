# Atlas

**A local, project-aware memory and retrieval assistant.**

Chat assistants forget earlier discussion once a conversation outgrows the context window,
and most retrieval-augmented (RAG) systems only look up documents. They don't remember the
conversations themselves. Many also depend on cloud vector databases and hosted models.

Atlas answers questions from your own documents and from a long-term memory of your
conversations, and everything runs on your machine:

- **Hierarchical memory.** Conversations are organised as Topic → Project → Conversation →
  Snapshot. Each query is routed to the right project, and older turns are summarised rather
  than dropped.
- **Retrieval.** Documents are chunked, embedded and stored in PostgreSQL (pgvector) and a
  DiskANN index. Passages are re-ranked and checked for sufficiency, and missing knowledge
  can be fetched from a list of trusted sources.
- **Reasoning.** A thinking layer (plan, jobs, critique) runs before generation, all on one
  local model, Qwen2.5-3B-Instruct (GGUF, run by llama.cpp).
- **Multi-user web app.** Accounts, private uploads and per-user conversation history are
  served through a REST API and a React frontend.

## Architecture

```text
Browser ──REST/JSON──▶ Java Spring Boot API ──gRPC──▶ Python pipeline ──▶ PostgreSQL + pgvector,
(React, :5173)         (backend/atlas, :8080)         (backend/app, :50051)   SQLite, DiskANN, model
```

| Part | Role |
|---|---|
| **Web frontend** | Sign up, sign in, ask questions, browse conversation history, upload documents. It talks only to the Java API. |
| **Java API** | The public REST API and the security boundary: accounts, Argon2id passwords, JWTs, request validation, uploads, and HTTP error mapping. |
| **Python pipeline** | All the intelligence: decomposition, memory, retrieval, thinking, generation and ingestion. It is also usable on its own through a CLI. |
| **Contract** | [`backend/proto/atlas/v1/atlas_service.proto`](backend/proto/atlas/v1/atlas_service.proto). The two backends share a service token, so Python only trusts calls from the Java API. |

## Documentation

| Read this | For |
|---|---|
| [`frontend/web_frontend/README.md`](frontend/web_frontend/README.md) | The React + Tailwind web app: setup, scripts, structure, design |
| [`backend/atlas/README.md`](backend/atlas/README.md) | The Java Spring Boot API: setup, packages, tests |
| [`backend/app/README.md`](backend/app/README.md) | The Python pipeline. Each module directory has its own README. |
| [`backend/BACKEND.md`](backend/BACKEND.md) | The two-service backend in depth: every setting, the full REST API, the security model |

## Repository layout

```text
frontend/
  web_frontend/        React 19 + Tailwind CSS 4 single-page app (Vite)
backend/
  atlas/               Java 21 / Spring Boot 4 REST API
  app/                 Python 3.11 pipeline, CLI and gRPC server
  proto/               gRPC contract shared by both backends
  BACKEND.md           backend reference
```

## Setting up

### Prerequisites

| Tool | Version | Used by |
|---|---|---|
| PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension | any version pgvector supports | both backends |
| [uv](https://docs.astral.sh/uv/) | recent | Python pipeline. It installs Python 3.11 itself. **3.11 is required**, because `diskannpy` has no wheels for newer versions. |
| Java JDK | 21 | Java API (Maven comes with the `./mvnw` wrapper) |
| Node.js | 20.19+ or 22.12+ | web frontend |
| Docker | any | only for the Java tests (Testcontainers) |

The model takes about 2 GB of disk. On the CPU a query takes minutes; on an NVIDIA GPU it
takes seconds. See [Using an NVIDIA GPU](#using-an-nvidia-gpu-optional).

### 1. Database

```bash
createdb Vectors
psql -d Vectors -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

Both backends use this one database. The Python pipeline creates its own tables, and the
Java API creates `user_table` with Flyway on first start.

### 2. Python pipeline

```bash
cd backend/app
cp .env.example .env                       # then set DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
uv sync --all-extras                       # builds .venv from uv.lock
uv run --no-sync python download_models/install_llama_cpp.py   # llama-cpp-python for this machine's GPU or CPU
uv run --no-sync python download_models/download_draft_model.py   # Qwen2.5-3B-Instruct, Q4_K_M
uv run --no-sync python scripts/smoke.py   # optional: one real round trip through PostgreSQL
```

`.env` holds your database password and is ignored by git. Never commit it.

After the first install, always use `uv run --no-sync`. A plain `uv sync` or `uv run`
re-syncs the environment and **removes `llama-cpp-python`**, which isn't in the lock file.

Optional: `uv run --no-sync python scripts/build_corpus.py` ingests a starter corpus of
documentation pages from postgresql.org, sqlite.org and cassandra.apache.org.

#### Using an NVIDIA GPU (optional)

The language model (llama.cpp), the embedder and the reranker run on the GPU when one is
usable. Everything else (the vector index, the databases, parsing and chunking) stays on the
CPU. Nothing needs configuring, but the GPU has to be usable first:

1. **The NVIDIA driver must be loaded.** `nvidia-smi` should list the card. On Linux with
   Secure Boot, the driver module from akmods is signed with a local key the firmware doesn't
   trust yet, so it's rejected at boot and `nvidia-smi` fails. Enroll the key once, then reboot
   and choose *Enroll MOK* on the blue screen:
   ```bash
   sudo mokutil --import /etc/pki/akmods/certs/public_key.der
   ```
2. **The CUDA toolkit (`nvcc`)** must be on `PATH` to compile llama.cpp for the GPU. RTX 50xx
   (Blackwell) cards need CUDA 12.8 or later.
3. **Rebuild llama-cpp-python for CUDA.** The installer detects the GPU and its architecture,
   and keeps the installed version:
   ```bash
   uv run --no-sync python download_models/install_llama_cpp.py
   ```

Then `main.py status` shows, for example, `resident on the GPU (every layer)` for the model
and `on the GPU (NVIDIA GeForce RTX ...)` for the embedder and reranker. If a GPU load fails
(for example, out of GPU memory), that model falls back to the CPU and `status` says why.

| Setting | Default | Meaning |
|---|---|---|
| `ATLAS_GPU_LAYERS` | `-1` | Model layers on the GPU: `-1` all of them, `0` none (CPU only), or a number for a GPU with little memory |
| `ATLAS_TORCH_DEVICE` | `auto` | Embedder and reranker: `auto` (GPU when present), `cuda` or `cpu` |

With the defaults, Atlas uses about 4 to 4.5 GB of GPU memory at its 32k-token context.

### 3. Java API

Nothing to install beyond JDK 21. The first build downloads Maven and the dependencies:

```bash
cd backend/atlas
./mvnw -q -DskipTests package
```

By default it reads the database settings from `backend/app/.env`.

### 4. Web frontend

```bash
cd frontend/web_frontend
npm install
cp .env.example .env                       # VITE_API_BASE_URL: where the Java API listens
```

### 5. Shared secrets (once)

Both backends need the **same** `ATLAS_GRPC_SERVICE_TOKEN`, and the Java API needs a JWT
signing secret. Generate them **once** and keep them in a file outside the repository.
Running `openssl rand` again in each terminal creates different tokens, and the Java API
will then fail to reach the pipeline.

```bash
mkdir -p ~/.config/atlas
cat > ~/.config/atlas/secrets.env <<EOF
export ATLAS_JWT_SECRET="$(openssl rand -base64 48)"
export ATLAS_GRPC_SERVICE_TOKEN="$(openssl rand -hex 32)"
EOF
chmod 600 ~/.config/atlas/secrets.env
```

## Running

Start the services in this order, each in its own terminal:

```bash
# 1. Python pipeline: wait until it is listening on 127.0.0.1:50051 (loading the model takes a while)
source ~/.config/atlas/secrets.env
cd backend/app && uv run --no-sync python main.py serve

# 2. Java API
source ~/.config/atlas/secrets.env
cd backend/atlas && ./mvnw spring-boot:run

# 3. Web frontend
cd frontend/web_frontend && npm run dev
```

Open <http://localhost:5173>, create an account, upload documents on the Documents page, and
ask questions in the Workspace.

The Python pipeline also works without the web stack, through its CLI:

```bash
cd backend/app
uv run --no-sync python main.py start            # interactive
uv run --no-sync python main.py -q "question"    # one question, then exit
uv run --no-sync python main.py status           # from another terminal: what the running Atlas is doing
```

Only one of `start`, `serve` and `-q` can run at a time per data directory.

## Tests

```bash
cd backend/app && uv run --no-sync pytest test/ -m "not real_model and not live"   # fast, needs no database or model
cd backend/atlas && ./mvnw test                                                     # needs Docker; never touches your database
cd frontend/web_frontend && npm test && npm run typecheck
```

There are also opt-in suites:

- `backend/app/test/live_testing`: real PostgreSQL.
- `-m real_model`: the real model.
- `backend/app/scripts/backend_e2e.py`: drives the full running stack.

These suites write throwaway data. See each component's README for details.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| The frontend shows the API as unreachable | The Java API isn't running, or `VITE_API_BASE_URL` points to the wrong port. Restart `npm run dev` after editing `.env`. |
| Queries fail with `BACKEND_UNAVAILABLE` | The pipeline isn't running yet, or the two backends have different `ATLAS_GRPC_SERVICE_TOKEN` values. `source` the same secrets file in both terminals and restart both. |
| The Java API can't start: port 8080 in use | Another service holds port 8080. Run it with `ATLAS_HTTP_PORT=8081 ./mvnw spring-boot:run` and set `VITE_API_BASE_URL=http://localhost:8081`. |
| Browser CORS errors | The frontend isn't on `http://localhost:5173` or `:3000`. Add its address to `ATLAS_ALLOWED_ORIGINS`. |
| `No module named 'llama_cpp'` | `uv sync` removed it. Reinstall it with `uv run --no-sync python download_models/install_llama_cpp.py`, and use `uv run --no-sync` from then on. |
| `status` says `no GPU offload` | llama-cpp-python is a CPU build, or the GPU isn't visible. Check `nvidia-smi`, then rerun the installer. See [Using an NVIDIA GPU](#using-an-nvidia-gpu-optional). |
| Another Atlas is running | `start`, `serve` and `-q` lock the data directory. Stop the other one (`main.py status` shows it). |
