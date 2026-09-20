# CLI (`cli/`)

## What this module does

The interactive loop: read a query, answer it, repeat.

## State

`cli_interface()` reads input and prints it back. The line where the query would
be answered is `# some stuff` — there is no retrieval, no model, no response
(**bugs 2.1 and 2.2**). `main.py --query` is the same placeholder in
non-interactive form.

This is the thinnest part of the project and the only part a user sees.

## Why the query id is bound here

The CLI is the outermost point that knows a query has begun, so it is the only
place that can mark every log line the query produces. Binding it deeper would
miss whatever ran before that point; binding it shallower would not know when
one query ends and the next starts.

## What it already does right

Each query is wrapped in `log_context(query_id=...)`, so every line any module
logs while answering it carries the same id. That is what will let one request
be pulled out of a log that the planner's concurrent nodes are all writing to.

`KeyboardInterrupt` is caught separately from other exceptions: ctrl-c is how
the loop is meant to end, not a failure.

## When the query loop is built

It should assemble its prompt from `Turn` objects — role and text — rather than
from the summariser's transcript text, which is that module's own format and not
an interface.
