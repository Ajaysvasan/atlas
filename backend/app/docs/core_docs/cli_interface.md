# Interactive CLI Interface Module (`cli/cli_interface.py`)

## Overview & Purpose
The `InteractiveCLI` class provides a REPL (Read-Eval-Print Loop) session interface that allows administrators and developers to interactively query the backend pipeline, submit textual inputs, and trigger ingestion workflows from the terminal.

---

## Classes & Public APIs

### `class InteractiveCLI`
Manages standard input prompts, query dispatching, and graceful interrupt handling during interactive debugging sessions.

#### Constructor: `__init__(self) -> None`
Initializes the CLI session instance and acquires a dedicated logger (`get_logger(__name__)`).

---

### Methods

#### `start_session(self) -> None`
Starts the continuous REPL loop, waiting for user queries and routing them to processing endpoints.

##### Parameters
*None.*

##### Return Value
- **Type:** `None`

##### How It Works
1. Logs session start (`logger.info("Starting interactive CLI session...")`) and prints a welcome header (`"--- Final Year Project Interactive CLI ---"`).
2. Enters an infinite `while True:` read loop using Python's `input("Query > ")`.
3. Checks if the user entered `"exit"` or `"quit"` (case-insensitive). If matched, prints a goodbye message, logs session termination, and breaks out of the loop.
4. If a non-empty query string is received, passes it to `process_query(query)`.
5. Catches `KeyboardInterrupt` (`Ctrl+C`) during input prompt waiting, allowing users to cleanly exit without traceback dumps.

---

#### `process_query(self, query: str) -> None`
Handles processing of a single interactive query string submitted during the REPL session.

##### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `query` | `str` | The text string submitted by the user at the command prompt. |

##### Return Value
- **Type:** `None`

##### How It Works
1. Logs the reception of the user query (`logger.debug(f"Received query: {query}")`).
2. Prints an acknowledgment confirming query receipt (`f"\n[Processing Query]: '{query}' ...\n"`).
3. Serves as the hook where downstream retrieval (`VectorDbManager.search_vector`), document generation, or conversational snapshot memory (`ConversationVectorMetaDataManager`) can be invoked to return results to the user.
