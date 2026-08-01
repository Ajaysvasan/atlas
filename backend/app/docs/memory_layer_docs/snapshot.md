# Conversational Snapshots (`snapshot.py` & `snapShotNodes.py`)

## Overview & Purpose
The `snapshot.py` and `snapShotNodes.py` submodules manage conversational snapshot history in memory. They provide bidirectional cursor navigation (`left_cursor` and `right_cursor`) over chronological turn nodes (`SnapShotNode`) and implement cosine similarity search (`SnapShot.search`) to retrieve the most semantically relevant conversational snapshot given a query vector.

---

## Classes & Public APIs

### `class SnapShotNode` (`snapShotNodes.py`)
An immutable frozen dataclass (`@dataclass(frozen=True)`) representing a single timestamped snapshot turn inside a conversation history tree.

#### Fields & Parameters
| Field | Type | Description |
| :--- | :--- | :--- |
| `snapshot_id` | `str` | Unique string identifier of the snapshot. |
| `time_of_snap_shot` | `str` | Timestamp string indicating when the snapshot occurred. |
| `size_of_the_summary` | `int` | Byte size of the summary vector stored on disk. |
| `len_of_the_summary` | `int` | Character or token count of the summary text string. |
| `summary_vector_ids` | `List[uint32]` | List of vector index IDs (`DiskANN` tags) associated with this snapshot's summaries. |
| `conversation_id` | `str` | Foreign key referencing the parent conversation. |
| `cumulative_summary_vector_id` | `uint32` | Vector index ID of the rolling cumulative summary up to this snapshot. |

---

### `class SnapShot` (`snapshot.py`)
In-memory snapshot container that tracks chronological history list (`__snap_shot_list`) and exposes cursor movements (`advance`, `prev`) along with vector similarity searches (`search`).

#### Constructor: `__init__(self) -> None`
Initializes internal cursors to `-1` (`self.__left_cursor = -1`, `self.__right_cursor = -1`) and creates an empty list (`self.__snap_shot_list = []`).

---

#### Methods

##### `add(self, snapshot_id: str, time_of_snapshot: str, size_of_the_summary: int, len_of_the_summary: int, summary_vector_ids: List, conversation_id: str, cumulative_summary_vector_id: uint32, reset_right_pointer: bool = True, reset_left_pointer: bool = True) -> None`
Constructs a `SnapShotNode` using the provided parameters, appends it to `__snap_shot_list`, and optionally resets the left/right traversal cursors.

###### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `snapshot_id` | `str` | *Required* | Snapshot unique identifier. |
| `time_of_snapshot` | `str` | *Required* | Timestamp string (`ISO 8601`). |
| `size_of_the_summary` | `int` | *Required* | Summary byte size. |
| `len_of_the_summary` | `int` | *Required* | Summary character length. |
| `summary_vector_ids` | `List` | *Required* | List of vector ID tags. |
| `conversation_id` | `str` | *Required* | Parent conversation identifier. |
| `cumulative_summary_vector_id` | `uint32` | *Required* | Cumulative summary vector tag. |
| `reset_right_pointer` | `bool` | `True` | If `True`, resets `__right_cursor` to `len(__snap_shot_list) - 1`. |
| `reset_left_pointer` | `bool` | `True` | If `True`, resets `__left_cursor` to `0`. |

---

##### `advance(self) -> None`
Advances `self.__left_cursor` by `1` index position.
- **Raises:** `InvalidCursorException("left", self.__left_cursor + 1)` if `__left_cursor + 1 >= len(__snap_shot_list)`.

---

##### `prev(self) -> None`
Decrements `self.__right_cursor` by `1` index position.
- **Raises:** `InvalidCursorException("right", self.__right_cursor - 1)` if `__right_cursor - 1 < 0`.

---

##### `search(self, query: List) -> SnapShotNode`
Performs bidirectional cosine similarity evaluation between the `query` vector and snapshot cumulative vectors, returning the snapshot node (`SnapShotNode`) with the highest similarity score.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `query` | `List` | Numeric query vector (`List[float]`) to match against historical snapshots. |

###### Return Value
- **Type:** `SnapShotNode`
- **Description:** The historical snapshot node exhibiting the maximum cosine similarity to `query`.

###### How It Works
1. Calls internal helper `__find_best_snapshot(query)`.
2. Verifies `len(__snap_shot_list) > 0` (`raises NullPointerException("No snap shots found")` otherwise).
3. Iterates bidirectionally using `while self.__left_cursor <= self.__right_cursor:`, inspecting both `left_snap = self.__snap_shot_list[self.__left_cursor]` and `right_snap = self.__snap_shot_list[self.__right_cursor]`.
4. Computes PyTorch `cosine_similarity(...)` across snapshot cumulative vectors against `tensor(query)` and updates `best_snap_shot_idx` when higher similarity scores (`best_similarity`) are found.
5. Increments `self.__left_cursor += 1` and decrements `self.__right_cursor -= 1` until cursors cross.
6. In `finally:`, resets cursors (`__reset_right_pointer()`, `__reset_left_pointer()`) and returns `self.__snap_shot_list[best_snap_shot_idx]`.
