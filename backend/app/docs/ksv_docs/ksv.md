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

## Tests

`test/ksv_testing/test_sufficiency.py` — 36, no mocks.
