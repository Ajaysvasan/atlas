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

## Datasets are a different kind

A documentation page is bounded by `MAX_BYTES`; a dataset is not. It can be
hundreds of gigabytes, and it carries a licence that may forbid the use being
made of it. So `datasets.py` screens on two things prose never needs:

- **Licence.** Only recognised permissive licences pass. An absent or
  unrecognised licence is **refused, not assumed open** — the cost of being
  wrong is redistributing someone's work without the right to.
- **Size.** Over the cap is refused, and so is an unknown size: it cannot be
  bounded in advance and no prose cap applies to a dataset download.

It returns **references, never payloads**. Deciding whether a dataset is worth
having is this subsystem's job; downloading and ingesting one is the data
layer's pipeline.

## What has been acquired

`acquisition_store.py` keeps one row per document knowledge was taken from:
`id`, `topic`, `query`, `url`, `created_at`. Owned by this subsystem rather than the memory
layer, because it answers a question about acquisition and not about any
conversation or project.

`AUTOINCREMENT`, not a bare `INTEGER PRIMARY KEY`. Both assign ids, but only one
refuses to reuse the id of a deleted row — and for a provenance record, an id
pointing at two different documents over the table's life is worse than a gap in
the numbers.

**Recording is tied to storage, not to fetching.** The table says what the
system took knowledge from, so a document that was fetched and then discarded as
off-subject, or kept but never stored because it did not answer the query, is
not recorded. Nothing was taken from it.

The stamp uses `storage.timestamps.utc_now`, not a format of this module's own.
That helper exists because `utc_now` was once written out four times over
(bug 4.58); a copy here would be the fifth, and two formats in one database do
not sort against each other.

The same URL may appear twice. It is a log of acquisitions, not an index of
documents held: the same page fetched for a different query is a different
acquisition, and collapsing them would lose which query it answered.

## Shared cosine

`similarity.py::cosine_scores` moved here from `ProjectManager`, which now
imports it. It had been one of three similarity implementations in the tree; a
subsystem whose whole job is scoring should not add a fourth, and the dependency
points the right way with it here.

## What may be collected at all

An acquisition is bounded by an `AcquisitionTarget` — a topic, optional
subtopics, and the query. The topic is what makes the bound possible: without
it, "fetch until the query is answered" has no notion of *unrelated*, and a page
about something else contributes several thousand chunks that answered nothing
and are then searched forever.

**Two floors, two jobs.** A chunk must first clear the topic floor — is this
material about the right subject at all — before its similarity to the query
counts toward sufficiency. They are separate because material can be squarely
on-topic without answering the question, which is worth keeping, while material
that answers a differently-worded question about another subject is not.

**Subtopics narrow, they do not broaden.** With subtopics supplied a chunk must
clear the topic floor *and* match at least one of them. A page fetched for
"write-ahead logging" carries plenty that is about databases generally, and
keeping it is how an index fills with things nobody asked for.

The gate runs **per round, before anything accumulates**, so off-subject
material never joins the running total and never reaches the store. On top of
it, `max_kept` caps how many chunks one acquisition may retain; when there is
more on-subject material than room, the ones closest to the *query* are kept,
because that is what was actually asked.

So the worst case is statable: `max_rounds` documents, each at most `MAX_BYTES`,
yielding at most `max_kept` chunks, every one of them on-subject.

## Which sources, and where the list lives

`sources.toml`, beside the code. Not a database: an allowlist that anything with
write access can change is a much weaker guarantee than one in git, where every
trust change is a reviewable commit.

**Descriptions are load-bearing config, not documentation.** Sources are chosen
by scoring the description against the topic, so it should say what the source
covers in the words someone would use to ask about it. "SQLite documentation"
ranks worse than naming write-ahead logging, pragmas and query planning.

Selection is semantic, the same mechanism `ProjectManager` uses to route a query
to a project — no tags to maintain, and it adapts to topics nobody anticipated.
Scored against the **topic and subtopics**, never the query: a source is chosen
for the subject it covers, while the question's wording is about the answer.

`SOURCE_FLOOR = 0.25` is **measured, unlike the other floors here.** Over six
topics against the shipped descriptions, 0.10 to 0.30 all selected the right
source every time, while the number searched fell from 3.0 to 1.5; 0.35 began
missing. 0.25 searches 1.8 sources of 8 with no misses — against 7 before
selection existed. Six topics is a small set, so re-measure when the list grows.

When nothing clears the floor the best single source is used anyway. That beats
searching everything, and beats searching nothing.

## Where candidates come from

`discovery.py` searches the trusted sources **themselves** — each carries its own
search URL, and the terms are built from topic, subtopics and query. No general
web search is involved, so every candidate comes from inside a source that is
already on the allowlist.

Links are extracted generically rather than with a parser per source: take every
link on the results page, then discard those that leave the host. That needs no
knowledge of each site's markup, and the host rule is what makes it safe — a
results page that links outward simply yields fewer candidates.

A caller that already knows where to look can pass `urls` and skip discovery.

## Acquiring knowledge that is not here yet

The second API, `acquire`, is the one the global retrieval layer uses: fetch,
chunk, embed, verify, and repeat while the answer is not there. The memory layer
never calls it — a shortfall there is answered by retrieval, not by going to the
network.

Knowledge **accumulates across rounds** and is verified against everything
gathered so far, not just the newest document. Two sources that each fall short
alone may answer the query together, and re-verifying only the latest would miss
that.

Only knowledge that answered the query is stored. Keeping every fetch would fill
the index with material that answered nothing, and that cost is paid on every
later search. `store_on_partial` exists for callers who would rather keep it,
and is off by default.

One bad source does not end the attempt. An untrusted URL or a failed fetch is
recorded in `failures` and the loop moves on, because the next source may hold
the answer.

## Why the source has to be trusted

Anything this fetches becomes knowledge the system answers from, so the question
is not "can we reach it" but "would we repeat it". `sources.py` holds an
allowlist of primary and standards sources; everything else is refused.

The matching rules are the substance:

| Rule | Why |
| :--- | :--- |
| https only | Plain http is rewritable in transit, so a trusted host over http is not trusted content |
| Host matched label by label | `endswith` would accept `docs.python.org.attacker.example` |
| Every redirect hop re-checked | A trusted host redirecting elsewhere is the ordinary way an allowlist leaks, so `follow_redirects` stays off and each hop goes back through the check |
| Resolved address must be outside this network | A listed name answering with `127.0.0.1` or `169.254.169.254` would turn a fetch into a request against this machine or a cloud metadata service |
| Size cap, timeout, content-type allowlist | A fetch should not be able to exhaust memory or hang the caller |

The address check is not airtight: a name can resolve differently between the
check and the request. It closes the straightforward case, and that limit is
stated here rather than implied.

The shipped allowlist is a **starting set to be curated**, not a default to grow
casually. A test asserts that no general-purpose site is on it — the point is
primary sources, not the open internet.

## Tests

`test/ksv_testing/test_sufficiency.py` — 36 tests, no mocks. Band edges are
tested by constructing vectors at an exact cosine to the query rather than
hoping a random one lands near a floor.
