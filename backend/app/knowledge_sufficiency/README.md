# Knowledge Sufficiency Verification (`knowledge_sufficiency/`)

## What this module does

Answers one question: **can this query be answered with the knowledge supplied?**
It returns one of three bands — FULL, PARTIAL or NONE — and the candidates worth
keeping. It does not retrieve anything, call a model, reach the network, or
decide what happens next.

## Why it is a sibling of `memory/` and not inside it

Two callers need it, and they need different things from it.

The **memory layer** asks whether this conversation can answer the query. The
**global retrieval layer** asks whether the system can at all, and if not goes
and gets the knowledge. Same question, different remedy — so the check belongs
to neither layer and sits beside both.

## Why it takes candidates rather than fetching them

The memory layer's candidates come from pgvector; the global retrieval layer's
come from DiskANN. If this subsystem fetched, it would have to know about both
stores and would become the place their differences accumulate. Taking vectors
keeps it a pure function over arrays: no mocks in its tests, and the two callers
stay free to fetch however suits them.

## Why the band is decided by the best score

A single strong match means the answer is present. Twenty mediocre ones do not
add up to it — they mean twenty documents are vaguely on topic. Summing or
averaging would let a crowd of weak matches outvote the absence of a real one,
which is the failure mode worth avoiding when the consequence is answering from
thin knowledge.

PARTIAL is a band on that same best score, not a coverage calculation. Deciding
*which parts* of a query are answered would mean decomposing it first, which is
a different subsystem.

## Why `supporting` holds more than the winner

On a PARTIAL verdict the caller combines what it has with what retrieval brings
back, so it needs everything already above the partial floor — not just the one
that set the band.

## The thresholds are placeholders

```
SUFFICIENT_FLOOR = 0.60    best >= this          -> FULL
PARTIAL_FLOOR    = 0.35    best >= this, < above -> PARTIAL
                           otherwise             -> NONE
```

`PARTIAL_FLOOR` borrows `ProjectManager`'s measured `SIMILARITY_FLOOR`, which
answers a similar question over the same embedding model. **`SUFFICIENT_FLOOR`
is a guess.** Both should be measured against real conversation vectors before
anyone relies on them.

The cost of being wrong is asymmetric: too high and every query triggers a
network fetch it did not need; too low and the system answers confidently from
knowledge it does not have. The second is worse, which argues for erring high
until there is data.

A test pins the current values — not to freeze them, but so that recalibrating
fails it and prompts the docs to be updated in the same commit.

## Shared cosine

`similarity.py::cosine_scores` moved here from `ProjectManager`, which now
imports it. It had been one of three similarity implementations in the tree; a
subsystem whose whole job is scoring should not add a fourth, and the dependency
points the right way with it here.

## Not built yet

The global retrieval layer also needs the network-call side of this subsystem —
fetch, chunk, embed, re-check, repeat until sufficient. That is a second API on
the same subsystem and is deliberately absent: only the memory layer's
`sufficiency_verification` is implemented.

## Tests

`test/ksv_testing/test_sufficiency.py` — 36 tests, no mocks. Band edges are
tested by constructing vectors at an exact cosine to the query rather than
hoping a random one lands near a floor.
