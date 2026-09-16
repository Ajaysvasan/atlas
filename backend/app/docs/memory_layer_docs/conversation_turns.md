# Reading a Conversation (`Turn`)

## Overview & Purpose

A conversation is stored one turn per row: the text in `summary_chunks`, and
the speaker, position and timestamp in `full_conversation`. Every turn has
always been written with its `role`. Until 1.1, nothing read the role back.

The original readers (`fetch_all`, `get_ranged_chunks`, `get_n_chunks`,
`get_sequence_after`) return bare chunk text as one-element tuples. Read
through them, a conversation is an ordered blob in which user and assistant
turns cannot be told apart. That is unusable for the one thing this layer
exists to feed: assembling a prompt. The summariser was the first casualty —
it fed the draft model `" ".join(...)` over bare text, so the model received
`How does DiskANN work? It builds a Vamana graph.` with no way to know who
asked and who answered.

The readers below return `Turn`, which keeps the speaker attached.

---

## `Turn`

Defined in `fullconversation_repository.py`. A `NamedTuple`, so it compares and
unpacks like a tuple and is addressable by field name.

| Field | Type | Source |
| :--- | :--- | :--- |
| `sequence_number` | `int` | `full_conversation.sequence_number` — the conversation order |
| `role` | `str` | `full_conversation.role` — `user`, `assistant` or `system` |
| `text` | `str` | `summary_chunks.chunk` |
| `created_at` | `str` | `full_conversation.created_at` — ISO-8601 UTC |
| `chunk_id` | `str` | the id that links a turn to snapshot metadata |

`role` and `text` are named after the write API, `append_turn(role, text)`, so a
turn round-trips under the same names. `role` reads back normalised — a turn
appended as `"  ASSISTANT "` reads back as `"assistant"`.

---

## Readers

Every reader orders by `sequence_number`. `created_at` is never a sort key: it
is caller-supplied through the low-level `add()`, which is what Bugs 4.30 and
4.31 were.

| `ConversationPoolManager` | `FullConversation` | `FullConversationRepository` | Returns |
| :--- | :--- | :--- | :--- |
| `history()` | `get_all_turns()` | `get_all_turns()` | every turn |
| `recent(n)` | `get_last_n_turns(n)` | `get_last_n_turns(n)` | the newest `n`, oldest first; `[]` for `n <= 0` |
| `context(start, end)` | `get_turns(start, end)` | `get_turns(start, end)` | `start <= sequence_number <= end` |
| `since(sequence)` | `get_turns_since(sequence)` | `get_turns_after(sequence)` | `sequence_number > sequence` |

`recent(n)` rejects a non-positive `n` explicitly: it would otherwise reach
SQLite as `LIMIT -1`, which means "no limit" and returns the whole conversation.

### A breaking change on the manager

`history()`, `recent()`, `context()` and `since()` on `ConversationPoolManager`
used to return the bare-text tuples and now return `Turn`. Code that did
`row[0]` to get the text now gets `sequence_number` — use `turn.text`. They had
no callers in the source tree when this changed, and the manager is what the
query layer will assemble prompts from, so it returns the shape that can be
assembled.

The text-only readers on `FullConversation` and `FullConversationRepository`
are unchanged.

---

## How the summariser uses them

### The transcript

`render_transcript(turns)` in `conversation_summary.py` writes one line per turn,
prefixed with the speaker:

```
User: How does DiskANN work?
Assistant: It builds a Vamana graph and searches it greedily.
User: Should retrieval span both vector stores?
Assistant: Only if memory is treated as a retrieval source.
```

`get_current_conversation(n)` returns this transcript for the current window.
The system prompt tells the model each line begins with its speaker and asks it
to attribute requests, decisions and answers to whoever made them.

This is the summariser's own format. The query layer will want chat messages,
not a transcript, and should build those from `Turn` directly rather than parse
this text.

### Batching never cuts a turn from its speaker

When a window is too large for one pass, `__batch_turns` packs the transcript
into batches of at most `max_chars`. The rules:

- **Batches break between turns.** A character split lands mid-turn, and the
  rest of that turn then opens the next batch with no label, so the model
  attributes it to nobody — or to whoever spoke before it.
- **Overlap is whole trailing turns**, up to `overlap_chars`, never a slice of
  text. The running summary is what carries context from one batch to the next;
  the overlap only needs to keep a short reply beside the turn it answers.
  Overlap is also capped so it always leaves room for the next line, so it can
  never push a batch over `max_chars`.
- **Only a turn too long for a batch by itself is cut.** It goes through the
  character splitter, with its own overlap, and every piece keeps its label.
- **An empty window is one empty batch**, the same thing the character
  splitter returned for an empty string.

### One window for the prompt and the coverage

`take_snapshot` reads the window twice — once as `Turn` for the prompt, once as
chunk rows for the snapshot record — and computes its start once for both. The
start depends on the summarised watermark, so computing it twice would let a
snapshot landing between the reads give the prompt and the recorded coverage
two different windows.

---

## Tests

| Where | What it pins |
| :--- | :--- |
| `test_full_converation.py::TestTurns` | Role round-trips for every role; interleaved dialogue reconstructs exactly; repeated text keeps its own speaker; inclusive and strict bounds; `n <= 0`; ordering ignores `created_at` with rows inserted out of order |
| `test_full_converation.py::TestBucketTurns` | Normalised role reads back; every bucket reader carries roles; a rejected batch leaves no turn |
| `test_conversation_pool_manager.py` | Every manager reader returns who said what, against real SQLite; a snapshot fired by `record_turn` sends the model the labelled transcript |
| `test_conversation_summary.py::TestBatchTurns` | Over 300 random shapes each: every batch fits, every line opens with a speaker, turns arrive whole, once, in order, and overlap is whole turns within its limit. An oversized turn's pieces leave no gap in its text |
| `test_conversation_summary.py::TestRealBatchingReachesTheModel` | With real batching, not a patched batcher: every prompt the model receives holds whole labelled turns, within budget, and no turn is lost across batches |
| `test_conversation_summary.py::TestTakeSnapshot` | The model is given the covered turns with roles; prompt and coverage read the same window |

Each was checked against a deliberate regression — the role column replaced by
a constant, labels dropped, batching reverted to a character split, overlap
allowed to overflow, the window start computed twice, the manager reverted to
bare text — and each regression fails at least one of them.

Two regressions passed every other test and are caught only by the tests added
for them: every batch after the first replaced by the first (the multi-batch
tests patch the batcher with canned strings, so they could not see it), and a
piece dropped from the middle of an oversized turn (every remaining piece is
still a substring of the text).
