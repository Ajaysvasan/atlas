"""What the query decomposition layer hands to the layers after it."""

from enum import Enum
from typing import NamedTuple, Tuple


class Mode(Enum):
    """M1 asks for something to be built or planned; M2 asks for an explanation."""

    DEVELOPMENT = "development"
    DOUBT = "doubt"


class TopicPath(NamedTuple):
    topic: str
    sub_topics: Tuple[str, ...]


class Analysis(NamedTuple):
    topics: Tuple[TopicPath, ...]
    sufficient: bool


class Decomposition(NamedTuple):
    """One query, decomposed: `query` is stored, `working_query` is what the model read."""

    exchange_id: str
    query: str
    working_query: str
    mode: Mode
    topics: Tuple[TopicPath, ...]
    sufficient: bool
    sub_queries: Tuple[str, ...]

    @property
    def primary_topic(self) -> str | None:
        """The most relevant topic, which routes the query in the memory layer."""
        return self.topics[0].topic if self.topics else None

    @property
    def expanded(self) -> bool:
        return bool(self.sub_queries)
