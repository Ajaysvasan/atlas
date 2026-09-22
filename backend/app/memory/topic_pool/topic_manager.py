from pathlib import Path
import uuid
from datetime import datetime , timezone

from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import TopicPoolMetaHandler
from config import get_logger

logger = get_logger(__name__)


def utc_now()->str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")

class TopicManager:
    topic: str
    query: str
    __repo: TopicPoolMetaHandler

    def __init__(
        self, topic: str, query: str, topic_pool_path: str | Path | None = None
    ) -> None:
        if topic is None or topic == "":
            raise ValueError("topic cannot be empty")
        if query is None or query == "":
            raise ValueError("query cannot be empty")
        self.topic = topic
        self.query = query

        self.__repo = TopicPoolMetaHandler(topic_pool_path)

    def __is_topic_exists(self):
        return self.__repo.is_topic_exists(self.topic)

    def get_topic_id(self):
        if self.__is_topic_exists():
            return self.__repo.get_topic_id(self.topic)
        logger.info("No active topic named %r", self.topic)
        raise Exception("The topic doesn't exists")

    def create_new_topic(self):
        if self.__is_topic_exists():
            raise Exception("topic already exists")
        topic_id = uuid.uuid4().hex
        created_at = utc_now()
        self.__repo.create_new_topic(topic_name=self.topic, topic_id = topic_id , created_at=created_at)
        logger.info("Created topic %r as %s", self.topic, topic_id)
    
    def soft_delete(self):
        if self.__is_topic_exists():
            topic_id = self.__repo.get_topic_id(self.topic)
            if isinstance(topic_id , str):
                self.__repo.soft_delete(topic_id)
                logger.info("Soft deleted topic %r (%s)", self.topic, topic_id)
            else:
                logger.error(
                    "Topic %r resolved to a %s, not a topic id",
                    self.topic,
                    type(topic_id).__name__,
                )
                raise ValueError(f"expects str but got  {type(topic_id)}")
        else:
            raise Exception("the topic doesn't exists")

    def close(self) -> None:
        self.__repo.close()

    def __enter__(self) -> "TopicManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
