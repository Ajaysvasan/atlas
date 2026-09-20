# Conversation summariser (`conversation_summary_pipeline/`)

## What this module does

Reads a window of conversation, asks the local draft model to summarise it, and
persists the result as a snapshot.

Model: Qwen2.5-3B-Instruct GGUF, 128K context, loaded through `llama_cpp` only
while a summary is being produced.

## How a snapshot is produced

```
take_snapshot(sequence)
  1. window start   = min(look-back, watermark + 1)
  2. covered rows   <- get_context_rows(start, sequence)     for the record
  3. turns          <- get_turns(start, sequence)            for the prompt
  4. transcript     "User: ...\nAssistant: ..."
  5. batches        whole turns, never split mid-turn
  6. rolling        each batch's output is the next batch's "previous summary"
  7. persist        embed, then SnapShot.add
```

## Why the window starts at the watermark, not just a look-back

The look-back is anchored to the newest turn. Whenever more unsummarised turns
piled up than the window spans — a bulk import, a large
`SNAPSHOT_EVERY_N_TURNS`, a period where snapshotting failed — every turn older
than the window was skipped, while the watermark still advanced past them and
declared them summarised. They could never be recovered. Starting no later than
the first unsummarised turn closes that gap, and the window may then be larger
than one pass, which is what the batching is for.

## Why the start is computed once

It depends on the watermark, so computing it twice would let a snapshot landing
in between give the prompt and the recorded coverage two different windows.

## Why batching breaks between turns

Each batch is read by the model as a transcript. A character split lands
mid-turn, and the rest of that turn opens the next batch with **no speaker
label**, so whatever it says is attributed to nobody — or to whoever spoke
before it. Overlap carries whole trailing turns for the same reason. Only a turn
too long for one batch is cut, and every piece keeps its label.

The running summary is what actually carries context across batches; the overlap
only has to keep a short reply next to the turn it answers.

## Why the cumulative vector id binds the timestamp

The embedder derives ids from content, which is right for chunks. It is wrong
here: `cumulative_vector_id` is a primary key and two snapshots can legitimately
produce identical summary text — a conversation whose gist stops changing, at
temperature 0.1. That collided and lost the snapshot.

## Why the model is loaded and unloaded every time

The weights are multiple gigabytes. They are loaded in a thread in parallel with
computing the batches, and released in a `finally` — including when batching
itself raises, which previously left the model resident with its VRAM unfreed.

## Known gap

An empty completion is not persisted: storing it would advance the watermark and
mark those turns summarised by nothing.
