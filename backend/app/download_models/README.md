# Model downloads (`download_models/`)

## What this module does

One-off setup scripts that fetch and build what the summariser needs. Neither is
imported by the application.

| Script | Does |
| :--- | :--- |
| `download_draft_model.py` | Fetches the Qwen2.5-3B-Instruct GGUF into `Config.DRAFT_MODEL_PATH` |
| `install_llama_cpp.py` | Installs `llama-cpp-python` with the right backend for the machine |

## Why `install_llama_cpp.py` detects hardware

`llama-cpp-python` compiles against one acceleration backend, chosen at install
time: CUDA, ROCm, Metal, or plain CPU. Getting it wrong means either a build
failure or a model that silently runs on the CPU at a fraction of the speed. The
script probes for `nvidia-smi`, `rocm-smi` and Apple Silicon so nobody has to
remember the right environment variable.

## Why the download script is idempotent

It checks for the file before fetching. The GGUF is ~2 GB, and re-running setup
is a normal thing to do.

## Known gap

**Bug 6.1** — `download_draft_model.py` passes `local_dir_use_symlinks`, which
the pinned `huggingface-hub==1.21.0` removed. The call raises `TypeError` before
downloading anything, so the draft model cannot be fetched by the documented
command, which in turn blocks `ConversationSummary`.
