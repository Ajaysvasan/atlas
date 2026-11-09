"""conversation_table in PostgreSQL: each query as it arrives, and the system's response once there is one."""

import threading
import uuid
from datetime import datetime, timezone
from typing import List, NamedTuple, Tuple

from config import get_logger
from query_decomposition.query_decomposition_exceptions import (
    ExchangeNotFound,
    ResponseAlreadyRecorded,
)

logger = get_logger(__name__)

_COLUMNS = (
    "exchange_id, conversation_id, user_id, query, query_created_at, "
    "system_response, system_response_time"
)


class Exchange(NamedTuple):
    exchange_id: str
    conversation_id: str
    user_id: str
    query: str
    query_created_at: datetime
    system_response: str | None
    system_response_time: datetime | None


def _connect():
    from storage.postgres import connect

    connection = connect()
    info = connection.info
    logger.info("Connected to the conversation table in %s@%s:%s", info.dbname, info.host, info.port)
    return connection


class ConversationTable:
    def __init__(self, connection=None) -> None:
        self.conn = _connect() if connection is None else connection
        # The gRPC server reads one table from several threads; a transaction must not interleave.
        self.__lock = threading.RLock()
        self.__create_table()

    def __create_table(self) -> None:
        with self.__lock, self.conn.cursor() as cursor:
            cursor.execute("""
                create table if not exists conversation_table(
                    exchange_id varchar primary key,
                    conversation_id varchar not null,
                    user_id varchar not null,
                    query text not null,
                    query_created_at timestamptz not null,
                    system_response text,
                    system_response_time timestamptz,
                    check ((system_response is null) = (system_response_time is null))
                )
            """)
            cursor.execute(
                "create index if not exists idx_conversation_table_conversation "
                "on conversation_table(conversation_id, user_id, query_created_at)"
            )
        self.conn.commit()

    def __rows(self, query: str, parameters) -> List[tuple]:
        with self.__lock:
            try:
                with self.conn.cursor() as cursor:
                    cursor.execute(query, parameters)
                    rows = cursor.fetchall()
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return rows

    def __one(self, query: str, parameters) -> Exchange | None:
        rows = self.__rows(query, parameters)
        return Exchange(*rows[0]) if rows else None

    def record_query(self, conversation_id: str, user_id: str, query: str) -> Exchange:
        """Store a query as it arrives; its response is recorded later."""
        exchange = self.__one(
            f"insert into conversation_table "
            f"(exchange_id, conversation_id, user_id, query, query_created_at) "
            f"values (%s, %s, %s, %s, %s) returning {_COLUMNS}",
            (
                uuid.uuid4().hex,
                conversation_id,
                user_id,
                query,
                datetime.now(timezone.utc),
            ),
        )
        logger.debug(
            "Recorded query %s of conversation %s", exchange.exchange_id, conversation_id
        )
        return exchange

    def record_response(self, exchange_id: str, system_response: str) -> Exchange:
        """Attach the system's response to a recorded query, once."""
        exchange = self.__one(
            f"update conversation_table "
            f"set system_response = %s, system_response_time = %s "
            f"where exchange_id = %s and system_response is null "
            f"returning {_COLUMNS}",
            (system_response, datetime.now(timezone.utc), exchange_id),
        )
        if exchange is not None:
            logger.debug("Recorded the response to exchange %s", exchange_id)
            return exchange
        if self.get(exchange_id) is None:
            raise ExchangeNotFound(exchange_id)
        raise ResponseAlreadyRecorded(exchange_id)

    def get(self, exchange_id: str) -> Exchange | None:
        return self.__one(
            f"select {_COLUMNS} from conversation_table where exchange_id = %s",
            (exchange_id,),
        )

    def history(self, conversation_id: str, user_id: str) -> List[Exchange]:
        """Every exchange of one conversation, oldest first."""
        rows = self.__rows(
            f"select {_COLUMNS} from conversation_table "
            f"where conversation_id = %s and user_id = %s "
            f"order by query_created_at, exchange_id",
            (conversation_id, user_id),
        )
        return [Exchange(*row) for row in rows]

    def page(self, conversation_id: str, user_id: str, limit: int,
             after: Tuple[datetime, str] | None = None) -> List[Exchange]:
        """Up to `limit` exchanges of one conversation, oldest first, after (query_created_at, exchange_id)."""
        position = (None, None) if after is None else after
        rows = self.__rows(
            f"select {_COLUMNS} from conversation_table "
            f"where conversation_id = %s and user_id = %s "
            f"and (%s::timestamptz is null or (query_created_at, exchange_id) > (%s::timestamptz, %s)) "
            f"order by query_created_at, exchange_id limit %s",
            (conversation_id, user_id, position[0], position[0], position[1], limit),
        )
        return [Exchange(*row) for row in rows]

    def close(self) -> None:
        with self.__lock:
            self.conn.close()


def open_conversation_table() -> ConversationTable:
    """The conversation table on the PostgreSQL named in .env."""
    return ConversationTable()
