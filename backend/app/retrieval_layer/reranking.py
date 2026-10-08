"""Rescoring candidates with a model that reads the query and passage together."""

from typing import Callable, List, Sequence

from config import get_logger
from retrieval_layer.models import Passage, QueryPlan
from retrieval_layer.retrieval_exceptions import RerankerUnavailable
from retrieval_layer.settings import RERANK_BATCH, RERANK_MODEL

logger = get_logger(__name__)

RERANKED = "reranked"

Scorer = Callable[[Sequence[tuple]], Sequence[float]]


class Reranker:
    """A cross-encoder over (query, passage) pairs.

    Bi-encoder retrieval embeds the query and the passage apart and compares
    the results, which is what makes an index possible and also what loses the
    interaction between them. A cross-encoder reads both at once and is far
    more accurate — and far too slow to run over a corpus, which is why it only
    ever sees the candidates the first stages have already narrowed to.
    """

    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        batch_size: int = RERANK_BATCH,
        scorer: Scorer | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.__scorer = scorer

    @property
    def scorer(self) -> Scorer:
        """Loaded on first use, so turning reranking off costs nothing."""
        if self.__scorer is None:
            self.__scorer = self.__load()
        return self.__scorer

    def __load(self) -> Scorer:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as error:
            raise RerankerUnavailable(self.model_name, str(error)) from error
        try:
            model = CrossEncoder(self.model_name)
        except Exception as error:
            raise RerankerUnavailable(self.model_name, str(error)) from error
        logger.info("Loaded the reranking model %s", self.model_name)
        return lambda pairs: model.predict(
            list(pairs), batch_size=self.batch_size
        )

    def rerank(
        self, plan: QueryPlan, passages: Sequence[Passage]
    ) -> List[Passage]:
        """The same passages, rescored by the cross-encoder and reordered."""
        passages = list(passages)
        if len(passages) < 2:
            return passages
        pairs = [(plan.text, passage.text) for passage in passages]
        try:
            scores = self.scorer(pairs)
        except RerankerUnavailable:
            raise
        except Exception as error:
            # The earlier stages already produced a usable ranking; losing the
            # improvement is better than losing the results.
            logger.warning("Reranking failed, keeping the fused order: %s", error)
            return passages

        rescored = [
            passage.rescored(float(score), RERANKED)
            for passage, score in zip(passages, scores)
        ]
        rescored.sort(key=lambda passage: passage.score, reverse=True)
        logger.debug("Reranked %d passage(s)", len(rescored))
        return rescored
