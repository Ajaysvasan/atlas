# Main Application Entrypoint (`main.py`)

## Overview & Purpose
The `main.py` script serves as the primary execution entrypoint for running the backend service in standalone mode. It configures startup parameters, initializes command-line argument parsers, adjusts diagnostic logging verbosity, and launches the interactive command-line interface session.

---

## Functions & Public APIs

### `parse_arguments() -> argparse.Namespace`
Constructs and parses command-line flags passed when launching the application via `python main.py`.

#### Parameters
*None (reads directly from `sys.argv` via `argparse.ArgumentParser`).*

#### Command-Line Arguments
| Flag | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--verbose` | `-v` | `bool` (flag) | `False` | Enables verbose diagnostic logging (`DEBUG` level output to file and console). |
| `--dataset` | `-d` | `str` | `Config.DATASET_PATH` | Specifies an optional custom path to the dataset directory for ingestion. |

#### Return Value
- **Type:** `argparse.Namespace`
- **Description:** An object containing the parsed boolean flag `verbose` and string/path attribute `dataset`.

---

### `main() -> None`
Orchestrates the application lifecycle by parsing arguments, setting global diagnostic modes, checking dataset directories, and spinning up the CLI interface.

#### Parameters
*None.*

#### Return Value
- **Type:** `None`

#### How It Works
1. Invokes `parse_arguments()` to retrieve user-supplied runtime configurations.
2. Updates `Config.DEBUG = args.verbose`. If `--verbose` is enabled, loops through attached handlers on the primary logger (`get_logger("backend_main")`) and elevates their thresholds to `logging.DEBUG`.
3. Verifies whether the specified dataset directory exists (`Path(args.dataset).exists()`). Logs informative status messages or warnings if the directory is missing.
4. Instantiates `InteractiveCLI` and calls `cli.start_session()` to block and handle interactive terminal queries.
5. Catches `KeyboardInterrupt` (`Ctrl+C`) and unexpected exceptions to perform clean shutdown and log stack traces (`logger.error(..., exc_info=True)`).
