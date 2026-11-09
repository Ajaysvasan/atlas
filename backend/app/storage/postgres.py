"""A connection to the PostgreSQL database named in .env, where the vectors and conversation tables live."""

import os

from config import get_logger
from data_layer.datalayer_exceptions.datalayer_exceptions import MissingDatabaseConfiguration

logger = get_logger(__name__)

SETTINGS = ("DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")


def connect(connect_timeout: int | None = None):
    """A new connection; raises MissingDatabaseConfiguration naming any unset key, never a value."""
    import psycopg
    from dotenv import load_dotenv

    load_dotenv()
    settings = {name: os.getenv(name) for name in SETTINGS}
    missing = [name for name, value in settings.items() if value is None]
    if missing:
        logger.error("PostgreSQL configuration incomplete; missing %s", ", ".join(missing))
        raise MissingDatabaseConfiguration(missing)
    options = {"connect_timeout": connect_timeout} if connect_timeout else {}
    return psycopg.connect(
        dbname=settings["DB_NAME"],
        user=settings["DB_USER"],
        password=settings["DB_PASSWORD"],
        host=settings["DB_HOST"],
        port=settings["DB_PORT"],
        **options,
    )
