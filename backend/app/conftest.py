"""Root configuration. Imported before any test module."""

# Several test modules install a MagicMock for psycopg, each guarded so it only
# fires when the real module is absent. Importing it here first makes those
# guards see it, so an interpreter that has psycopg tests against the real
# adapters instead of a mock.
try:
    import psycopg  # noqa: F401
except ImportError:
    pass
