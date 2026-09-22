from pathlib import Path
from typing import List
import uuid
from datetime import datetime , timezone

from memory.memory_pool_exceptions import TopicNotFound
from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import (
    Topic,
    TopicPoolMetaHandler,
)
from config import get_logger
from memory.timestamps import utc_now

logger = get_logger(__name__)



class TopicManager:
    topic: str
    query: str | None
    __repo: TopicPoolMetaHandler

    def __init__(
        self,
        topic: str,
        query: str | None = None,
        topic_pool_path: str | Path | None = None,
    ) -> None:
        if topic is None or topic == "":
            raise ValueError("topic cannot be empty")
        if query is not None and query == "":
            raise ValueError("query cannot be empty")
        self.topic = topic
        self.query = query

        self.__repo = TopicPoolMetaHandler(topic_pool_path)

    def __is_topic_exists(self) -> bool:
        return self.__repo.is_topic_exists(self.topic)

    def get_topic_id(self) -> str:
        topic_id = self.__repo.get_topic_id(self.topic)
        if topic_id is None:
            logger.info("No active topic named %r", self.topic)
            raise TopicNotFound(self.topic)
        return topic_id

    def create_new_topic(self) -> str:
        topic_id = uuid.uuid4().hex
        self.__repo.create_new_topic(
            topic_name=self.topic, topic_id=topic_id, created_at=utc_now()
        )
        logger.info("Created topic %r as %s", self.topic, topic_id)
        return topic_id

    def soft_delete(self) -> str:
        topic_id = self.__repo.soft_delete_by_name(self.topic)
        logger.info("Soft deleted topic %r (%s)", self.topic, topic_id)
        return topic_id

    def list_topics(self) -> List[Topic]:
        return self.__repo.get_all_topics()

    def close(self) -> None:
        self.__repo.close()

    def __enter__(self) -> "TopicManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
