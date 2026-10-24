"""The query decomposition layer: what a query asks for, what it is about, and what it hides."""

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from config import Config, get_logger, log_context, log_timing
from memory.identifiers import require_identifier
from memory.memory_database import MemoryDatabase
from memory.topic_pool.topic_pool_repo.topic_pool_meta_handler import (
    TopicPoolMetaHandler,
)
from query_decomposition import conversation_table
from query_decomposition.conversation_table import Exchange
from query_decomposition.draft_model import DraftModel
from query_decomposition.models import Analysis, Decomposition, Mode, TopicPath
from query_decomposition.parsing import (
    ANALYSIS_SCHEMA,
    COMPLEXITY_SCHEMA,
    expansion_schema,
    read_analysis,
    read_mode,
    read_sub_queries,
)
from query_decomposition.preprocessing import clean, require_within
from query_decomposition.prompts import SystemPrompts, load_prompts
from query_decomposition.query_decomposition_exceptions import UnusableReply

logger = get_logger(__name__)

CONDENSE_MAX_TOKENS = 1024
COMPLEXITY_MAX_TOKENS = 16
ANALYSIS_MAX_TOKENS = 512
EXPANSION_MAX_TOKENS = 512

UNANALYSED = Analysis((), True)


class Decompose:
    """Decomposes each query with the draft model and records it in the conversation table."""

    def __init__(
        self,
        *,
        database: MemoryDatabase | str | Path | None = None,
        exchanges=None,
        model: Callable[[], DraftModel] = DraftModel,
        prompts: SystemPrompts | None = None,
        soft_limit_tokens: int = Config.QUERY_SOFT_LIMIT_TOKENS,
        hard_limit_tokens: int = Config.QUERY_HARD_LIMIT_TOKENS,
        max_sub_queries: int = Config.MAX_SUB_QUERIES,
    ) -> None:
        if not 0 < soft_limit_tokens <= hard_limit_tokens:
            raise ValueError(
                f"need 0 < soft_limit_tokens <= hard_limit_tokens; got "
                f"{soft_limit_tokens} and {hard_limit_tokens}"
            )
        if max_sub_queries < 1:
            raise ValueError(f"max_sub_queries must be at least 1, got {max_sub_queries}")
        self.database = MemoryDatabase.of(database)
        self.prompts = load_prompts() if prompts is None else prompts
        self.model = model
        self.soft_limit_tokens = soft_limit_tokens
        self.hard_limit_tokens = hard_limit_tokens
        self.max_sub_queries = max_sub_queries
        self.__owns_exchanges = exchanges is None
        self.exchanges = (
            conversation_table.open_conversation_table() if exchanges is None else exchanges
        )

    def decompose(self, query: str, conversation_id: str, user_id: str) -> Decomposition:
        """Record the query, then classify, analyse and, for an M1 query that says too little, expand it."""
        conversation_id = require_identifier(conversation_id, "conversation_id")
        user_id = require_identifier(user_id, "user_id")
        text = clean(query)
        tokens = require_within(text, self.hard_limit_tokens)
        exchange = self.exchanges.record_query(conversation_id, user_id, text)

        with log_context(exchange_id=exchange.exchange_id):
            existing_topics = self.__existing_topics()
            with log_timing(logger, "decomposing a query", tokens=tokens):
                with self.model() as model:
                    working = (
                        self.__condense(model, text)
                        if tokens > self.soft_limit_tokens
                        else text
                    )
                    mode = self.__mode(model, working)
                    analysis = self.__analysis(model, working, existing_topics)
                    sub_queries = (
                        self.__expand(model, working, analysis.topics)
                        if mode is Mode.DEVELOPMENT and not analysis.sufficient
                        else ()
                    )
            decomposition = Decomposition(
                exchange_id=exchange.exchange_id,
                query=text,
                working_query=working,
                mode=mode,
                topics=analysis.topics,
                sufficient=analysis.sufficient,
                sub_queries=sub_queries,
            )
            logger.info(
                "Decomposed a %s query: primary topic %r, %s, sub-queries: %d",
                mode.value,
                decomposition.primary_topic,
                "sufficient" if analysis.sufficient else "insufficient",
                len(sub_queries),
            )
            return decomposition

    def record_response(self, exchange_id: str, system_response: str) -> Exchange:
        """Attach the system's response to the query decompose() recorded."""
        if not isinstance(system_response, str) or not system_response.strip():
            raise ValueError("system_response must be a non-empty string")
        return self.exchanges.record_response(
            require_identifier(exchange_id, "exchange_id"), system_response
        )

    def __existing_topics(self) -> List[str]:
        topics = TopicPoolMetaHandler(self.database)
        try:
            return [topic.topic_name for topic in topics.get_all_topics()]
        finally:
            topics.close()

    def __condense(self, model, text: str) -> str:
        condensed = model.infer(
            self.prompts.condense, text, max_tokens=CONDENSE_MAX_TOKENS
        ).strip()
        if not condensed or len(condensed) >= len(text):
            logger.warning("Condensing did not shorten the query; analysing it whole")
            return text
        logger.debug(
            "Condensed the query from %d to %d characters", len(text), len(condensed)
        )
        return condensed

    def __mode(self, model, text: str) -> Mode:
        try:
            return read_mode(
                model.infer(
                    self.prompts.complexity,
                    text,
                    schema=COMPLEXITY_SCHEMA,
                    max_tokens=COMPLEXITY_MAX_TOKENS,
                )
            )
        except UnusableReply as error:
            logger.warning("%s; treating the query as %s", error, Mode.DOUBT.value)
            return Mode.DOUBT

    def __analysis(self, model, text: str, existing_topics: Sequence[str]) -> Analysis:
        listed = ", ".join(existing_topics) if existing_topics else "none yet"
        try:
            return read_analysis(
                model.infer(
                    self.prompts.analysis,
                    f"Existing topics: {listed}\n\nRequest:\n{text}",
                    schema=ANALYSIS_SCHEMA,
                    max_tokens=ANALYSIS_MAX_TOKENS,
                ),
                existing_topics,
            )
        except UnusableReply as error:
            logger.warning("%s; going on with no topics, as sufficient", error)
            return UNANALYSED

    def __expand(self, model, text: str, topics: Tuple[TopicPath, ...]) -> Tuple[str, ...]:
        listed = "; ".join(
            f"{path.topic} ({', '.join(path.sub_topics)})" if path.sub_topics else path.topic
            for path in topics
        )
        try:
            return read_sub_queries(
                model.infer(
                    self.prompts.expansion,
                    f"Topics: {listed or 'none'}\n\nRequest:\n{text}",
                    schema=expansion_schema(self.max_sub_queries),
                    max_tokens=EXPANSION_MAX_TOKENS,
                ),
                self.max_sub_queries,
            )
        except UnusableReply as error:
            logger.warning("%s; the query goes on unexpanded", error)
            return ()

    def close(self) -> None:
        if self.__owns_exchanges:
            self.exchanges.close()

    def __enter__(self) -> "Decompose":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
