# CLI (`cli/`)

## What this module does

The interactive development CLI over the full pipeline, plus two monitors that report on a
running Atlas without loading anything. `main.py` chooses between them:

```bash
uv run --no-sync python main.py start                    # interactive (also the default with no command)
uv run --no-sync python main.py --user NAME start        # as NAME (default: $ATLAS_USER, then your login name)
uv run --no-sync python main.py -q "question"            # one question in a new conversation, then exit
uv run --no-sync python main.py status [--time | --retries]
uv run --no-sync python main.py --user NAME conversations
```

Every option goes **before** the command: `--user`, `--no-acquire`, `-q`, `--verbose`,
`--log-json` and `--log-file`. They belong to the top-level parser, so
`main.py start --user NAME` is rejected with "unrecognized arguments".

| File | Contents |
|---|---|
| `cli_interface.py` | `cli_interface()`: builds `Components`, warms up, runs `CliSession`. `CliSession`: the read-dispatch loop, queries, `switch`, `new`, `status`. |
| `commands.py` | `parse()`: decides whether a typed line is a command or a query. `HELP`. |
| `monitor.py` | `main.py status` and `main.py conversations`: read and exit, never load a model |
| `render.py` | All text output: conversation lists, health, timings and retries reports, durations, timestamps |

## Continuing an earlier conversation

Conversations live in PostgreSQL (the `conversations` registry), keyed by user. Start the CLI
as the user who owns them, then use `switch` at the prompt:

```text
Atlas > switch          # numbered list of your 50 newest conversations: name, id, created
Select conversation (number or id, Enter to cancel): 2
[PostgreSQL MVCC] Atlas > switch 2      # or directly by list number
[PostgreSQL MVCC] Atlas > switch 3f9a   # or by id; a unique prefix of 4+ hex characters is enough
```

- **After switching:** the prompt shows the conversation's name, and each query continues
  it, with its memory (summaries and recent turns) and its saved KV checkpoint when one
  matches.
- **No transcript:** `switch` doesn't print earlier turns. The web workspace and
  `GET /conversations/{id}/exchanges` show them.
- **`new`:** the next query starts a fresh conversation.
- **`-q`:** always opens a new conversation.
- **Web accounts:** conversations started in the web app belong to the account's user id,
  a UUID (`user_table.user_id`), so run the CLI with `--user <that uuid>` to reach them.
- **Listing without loading anything:** `main.py --user NAME conversations` lists them, most
  recently active first, with exchange counts. It works even while `serve` is running.

## Commands vs queries

At the `Atlas >` prompt, anything is a query except a line that is exactly a command:

- **Commands:** `switch`, `switch <n|id>`, `new`, `status`, `status --time`,
  `status --retries`, `help`, `exit` (or `quit`).
- **Exact match only:** `switch to PostgreSQL 16?` is a question, not a command.
- **Leading `/`:** forces a command (`/status`). An unknown or malformed `/command` is an
  error rather than a query.

`parse()` makes this one decision, so it's the only place to change it.

## One service at a time

`start`, `serve` and `-q` load the model and index, so each takes an exclusive lock on
`data/run/atlas.lock` and is refused while another holds it. Stop `serve` before starting
the CLI.

The running service publishes `data/run/atlas.json` after warm-up and after every request:
what is loaded, counters, and recent request records, never prompt text. `status` reads that
file, and `conversations` reads PostgreSQL read-only. Neither writes anything or imports
llama.cpp or torch. Both exit 0 if Atlas is running and 3 if not, the codes service managers
use. When Atlas is stopped, `--time` and `--retries` show the last run's final query,
labelled as such.

## Why Ctrl+C doesn't leave

A model call can't be interrupted midway, and a request's writes must stay consistent. So
Ctrl+C during a query prints "Interrupted." and returns to the prompt instead of exiting. At
the prompt it only reminds you how to leave. `exit` or Ctrl+D ends the session.

## Correlating logs

`AtlasPipeline.ask` wraps each request in `log_context(request_id=..., conversation_id=...)`.
That way every line any module logs while answering carries the same ids, even while the
thinking layer's workers interleave. The CLI gets a fresh request id per query, and over gRPC
it is the Java request id.
