# Knowledge Sufficiency Verification (`KSVManager`)

## Overview & Purpose

Decides whether a query can be answered from the knowledge handed to it, and
returns the candidates worth keeping. Architecture and rationale live in
`knowledge_sufficiency/README.md`. This page is the API.

---

## `KSVManager`

```python
KSVManager(sufficient_floor: float = 0.60, partial_floor: float = 0.35)
```

| Parameter | Meaning |
| :--- | :--- |
| `sufficient_floor` | Best cosine at or above this is FULL |
| `partial_floor` | Best cosine at or above this, below the above, is PARTIAL |

Raises `ValueError` if either floor is outside `[0, 1]`, or if
`sufficient_floor < partial_floor` — which would leave no range in which
anything could be PARTIAL.

### `sufficiency_verification`

```python
sufficiency_verification(query_vector, candidates) -> Verdict
```

| Parameter | Meaning |
| :--- | :--- |
| `query_vector` | The embedded query |
| `candidates` | An iterable of `(id, vector)`. Ids are whatever the caller's store uses — the memory layer passes ints |

Raises `ValueError` if a candidate's width does not match the query's.

---

## `Verdict`

```python
Verdict(sufficiency, best, supporting, scores)
```

| Field | Meaning |
| :--- | :--- |
| `sufficiency` | `Sufficiency.FULL` / `.PARTIAL` / `.NONE` |
| `best` | The top cosine. `-1.0` when there were no candidates |
| `supporting` | Ids at or above `partial_floor`, best first |
| `scores` | Every score, in the order the candidates came in |

| Property | True when |
| :--- | :--- |
| `is_sufficient` | FULL |
| `needs_retrieval` | PARTIAL or NONE |

A `NamedTuple`, so it unpacks.

---

## Using it from the memory layer

```python
verdict = KSVManager().sufficiency_verification(query_vector, candidates)

if verdict.is_sufficient:
    return chunks_for(verdict.supporting)

retrieved = retrieval_layer.fetch(query)
return combine(chunks_for(verdict.supporting), retrieved)
```

On NONE, `supporting` is empty, so the same two lines cover the no-knowledge
case without a third branch.

---

## Bands

| Best score | Verdict |
| :--- | :--- |
| `>= sufficient_floor` | FULL |
| `>= partial_floor`, `< sufficient_floor` | PARTIAL |
| below | NONE |
| no candidates | NONE, `best == -1.0` |
| every candidate a zero vector | NONE, `best == -1.0` |

Floors are **inclusive**. A score sitting exactly on one is inside that band.

`-1.0` rather than `0.0` for the empty case: zero is a real cosine — an
orthogonal vector — and would sit inside a band if a floor were ever set below
it.

---

## `acquire`

```python
acquire(target, embed, urls=None, store=None, fetch=None,
        sources=DEFAULT_SOURCES, max_rounds=3, store_on_partial=False,
        topic_floor=0.45, max_kept=200) -> Acquisition
```

Fetch, chunk, embed, verify, repeat. Used by the global retrieval layer; the
memory layer does not call it.

| Parameter | Meaning |
| :--- | :--- |
| `target` | An `AcquisitionTarget`: topic, optional subtopics, query, and their vectors |
| `urls` | Candidate documents. `None` searches the trusted sources instead |
| `topic_floor` | A chunk below this against the topic is discarded, never stored |
| `max_kept` | Most chunks one acquisition may retain. The closest to the query win |
| `embed` | `Sequence[str] -> Sequence[(id, vector)]` |
| `store` | Called with everything acquired, only if the query was answered |
| `fetch` | Defaults to the guarded fetcher. Injectable, which is how tests run the loop without a network |
| `max_rounds` | At most this many sources are tried. Must be >= 1 |
| `store_on_partial` | Keep knowledge that only partly answered. Off by default |

```python
Acquisition(verdict, acquired, stored, rounds, failures)
```

| Field | Meaning |
| :--- | :--- |
| `verdict` | A `Verdict` over everything gathered |
| `acquired` | `(id, vector)` pairs from every round |
| `stored` | Whether `store` was called |
| `rounds` | How many sources were tried |
| `failures` | `(url, reason)` for each source skipped |
| `discarded` | Chunks dropped as off-subject or over the cap |
| `recorded` | Ids written to the acquisition store, empty if nothing was stored |

---

## `AcquisitionStore`

```python
AcquisitionStore(db_path=None)      # data/ksv/acquired_knowledge.sql
```

| Method | Returns |
| :--- | :--- |
| `record(topic, query, url)` | `int` — the new id |
| `record_many(topic, query, urls)` | `List[int]`, one transaction |
| `for_topic(topic)` | `List[AcquiredRecord]`, oldest first |
| `seen(url)` | `bool` |
| `all_records()` | `List[AcquiredRecord]` |
| `close()` | `None` |

```sql
acquired_knowledge
    id          integer primary key autoincrement
    topic       text not null
    query       text not null
    url         text not null
    created_at  text not null

idx_acquired_topic on (topic)
idx_acquired_url   on (url)
```

`AcquiredRecord(id, topic, query, url, created_at)`. The stamp comes from
`storage.timestamps.utc_now` — ISO-8601 UTC with microseconds, sortable as text.
One batch shares one stamp: those rows came from a single acquisition, and
stamping them apart would imply an ordering the id already carries.

Pass it to `acquire(..., acquisition_store=store)` and the loop records every
document that contributed, once the knowledge has been stored.

---

## `AcquisitionTarget`

```python
build_target(topic, query, embed_one, subtopics=()) -> AcquisitionTarget
```

Raises `ValueError` without a topic — it is what bounds collection — or without
a query. Blank subtopics are dropped. `target.terms` is what gets typed into a
source's search box: topic, subtopics and query joined.

A chunk is kept when `cos(chunk, topic) >= topic_floor`, **and**, if subtopics
were given, `cos(chunk, subtopic) >= topic_floor` for at least one of them.
Subtopics narrow rather than broaden.

---

## Trusted sources

`check_trusted(url, sources, allow_private=False)` returns the host or raises
`UntrustedSource`. `is_trusted(url, sources)` is the boolean form.

Refused: anything not https, any host not on the allowlist or a subdomain of
one, and any host resolving to a loopback, private, link-local, reserved or
multicast address. Redirects are followed manually so every hop is re-checked.

`FetchFailed` covers a reachable but unusable source: an error status, a
non-text content type, too many redirects, or a transport failure.

---

## Tests

`test/ksv_testing/` — 81 tests, no mocks and no network. `test_sufficiency.py`
(36), `test_trusted_sources.py` (26), `test_acquisition.py` (19).
