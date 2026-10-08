import importlib
import sys
from pathlib import Path

import pytest


TEST_DIR = Path(__file__).resolve().parent
APP_DIR = TEST_DIR.parents[1]


@pytest.fixture(autouse=True)
def _restore_config_module(monkeypatch):
    """Put back the config module these tests replace.

    Left replaced, every later test sees a different `Config` class from the
    one already-imported modules hold, and a monkeypatch of one misses the
    other — the memory database redirect included.
    """
    original = sys.modules.get("config")
    if original is not None:
        monkeypatch.setitem(sys.modules, "config", original)


def import_config():
    """Import config.py from the application directory."""
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))

    sys.modules.pop("config", None)

    return importlib.import_module("config")


def test_dataset_path_is_independent_of_working_directory(monkeypatch):
    """
    Regression test for Bug 1.2.

    DATASET_PATH must be based on the directory containing config.py,
    not on the process's current working directory.
    """

    expected_path = APP_DIR / "dataset"

    # Import config while running from the application directory.
    monkeypatch.chdir(APP_DIR)

    config_module = import_config()
    path_from_app_directory = Path(config_module.Config.DATASET_PATH)

    # Remove config so it is imported again under a different cwd.
    sys.modules.pop("config", None)

    # Use the test directory as a different working directory.
    monkeypatch.chdir(TEST_DIR)

    config_module = import_config()
    path_from_test_directory = Path(config_module.Config.DATASET_PATH)

    # Both must resolve to the application dataset directory.
    assert path_from_app_directory == expected_path
    assert path_from_test_directory == expected_path

    # Changing cwd must not change DATASET_PATH.
    assert path_from_app_directory == path_from_test_directory


def test_dataset_path_does_not_follow_current_working_directory(
    monkeypatch,
):
    """
    Reproduces the original Bug 1.2.

    The old implementation would resolve DATASET_PATH as:

        <current_working_directory>/dataset

    The fixed implementation must resolve it relative to config.py.
    """

    # Use the test directory as an unrelated working directory.
    monkeypatch.chdir(TEST_DIR)

    config_module = import_config()

    dataset_path = Path(config_module.Config.DATASET_PATH)
    expected_path = APP_DIR / "dataset"
    cwd_based_path = TEST_DIR / "dataset"

    # DATASET_PATH must point to app/dataset.
    assert dataset_path == expected_path

    # It must NOT point to cwd/dataset.
    assert dataset_path != cwd_based_path