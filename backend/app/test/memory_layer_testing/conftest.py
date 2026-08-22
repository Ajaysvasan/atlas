"""
Shared pytest configuration for memory layer tests.

Sets up sys.path and mocks heavy external dependencies (psycopg, dotenv)
before any test module is imported, so every test file in this directory
gets a clean environment without duplication.
"""

import os
import sys
from unittest.mock import MagicMock

# --------------------------------------------------------------------------- #
# Mock external dependencies that are not available in the test environment
# --------------------------------------------------------------------------- #
for _mod in ["psycopg", "dotenv", "dotenv.main"]:
    sys.modules.setdefault(_mod, MagicMock())

# llama_cpp — not installed in the test environment; lazy-imported only inside
# ConversationSummary.__load_model, so a top-level stub is enough for import
# resolution. Tests that exercise __load_model patch it directly.
_llama_cpp_mock = MagicMock()
sys.modules.setdefault("llama_cpp", _llama_cpp_mock)

# torch IS installed (snapshot.py uses it for cosine_similarity); do NOT mock it
# here — mocking would replace the real tensor/cosine_similarity with MagicMocks
# that cannot be compared with `>`, breaking all search tests.

# --------------------------------------------------------------------------- #
# Ensure the app root is on sys.path so package-style imports work:
#   from memory.topic_pool... import ...
# --------------------------------------------------------------------------- #
_here = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.abspath(os.path.join(_here, "..", ".."))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# --------------------------------------------------------------------------- #
# Add memory/ to sys.path so snapshot.py's bare-module import resolves:
#   from memory_pool_exceptions import ...   (Bug 4.23 workaround)
# --------------------------------------------------------------------------- #
_memory_dir = os.path.join(_app_dir, "memory")
if _memory_dir not in sys.path:
    sys.path.insert(0, _memory_dir)
