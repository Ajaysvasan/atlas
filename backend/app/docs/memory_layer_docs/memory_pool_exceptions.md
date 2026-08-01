# Memory Pool Exceptions Module (`memory_pool_exceptions.py`)

## Overview & Purpose
The `memory_pool_exceptions.py` module provides specialized custom exception classes used across the conversational memory and snapshot traversal layers (`SnapShot` and `ConversationVectorManager`). These exceptions ensure clear diagnostic feedback when cursors exceed valid list bounds, when operations target empty snapshot arrays, or when vector arrays have mismatched dimensionality.

---

## Classes & Public APIs

### `class InvalidCursorException(Exception)`
Raised when an in-memory snapshot traversal (`SnapShot.advance()` or `SnapShot.prev()`) attempts to move a cursor beyond the valid range of `self.__snap_shot_list`.

#### Constructor: `__init__(self, pointer_name: str, index_accessed: int) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `pointer_name` | `str` | Name of the cursor (`"left"` or `"right"`). |
| `index_accessed` | `int` | The out-of-bounds target index (`self.__left_cursor + 1` or `self.__right_cursor - 1`). |

#### String Representation (`__str__`)
Returns: `"The {self.pointer_name} went out of bound. Tried to access index {self.index_accessed}"`

---

### `class NullPointerException(Exception)`
Raised when operations such as `SnapShot.__reset_left_pointer()`, `SnapShot.__reset_right_pointer()`, or `SnapShot.search()` are executed against an empty `SnapShot` instance (`len(self.__snap_shot_list) == 0`).

#### Constructor: `__init__(self, error_message: str) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `error_message` | `str` | Detailed error context string (`"No snap shots found"`). |

#### String Representation (`__str__`)
Returns: `{self.error_message}`

---

### `class InvalidVectorDimension(Exception)`
Raised by `ConversationVectorManager.add_summary_vectors()` when a submitted summary vector array (`vectors.shape[1]`) does not match the expected `vector_dimension` (`Config.DIMENSIONS`).

#### Constructor: `__init__(self, invalid_dimension: int, expected_dimension: int) -> None`
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `invalid_dimension` | `int` | The actual column width (`shape[1]`) received from the caller. |
| `expected_dimension` | `int` | The expected vector dimension configured in `self.vector_dimension` (e.g. `128`). |

#### String Representation (`__str__`)
Returns: `"Got the dimension {self.dimension}. Expected the dimension {self.expected_dimension}"`
