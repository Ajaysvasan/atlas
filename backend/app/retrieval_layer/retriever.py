"""The retrieval layer's only public entry point."""

from pathlib import Path

from config import Config, get_logger, log_context, new_correlation_id
from data_layer.vector_db_manager.repository.vectorMetaDataRepository import (
    VectorMetaDataRepository,
)
from retrieval_layer.assembly import assemble, deduplicate
from retrieval_layer.caching import ResultCache, cache_key
from retrieval_layer.diversity import maximal_marginal_relevance
from retrieval_layer.fusion import reciprocal_rank_fusion
from retrieval_layer.hydration import Hydration
from retrieval_layer.keyword_search import FTS_PATH, KeywordSearch
from retrieval_layer.models import RetrievalRequest, RetrievalResult
from retrieval_layer.query_preparation import QueryPreparation
from retrieval_layer.reranking import Reranker
from retrieval_layer.retrieval_exceptions import EmptyQuery, InvalidRetrievalSetting
from retrieval_layer.settings import RetrievalSettings
from retrieval_layer.telemetry import StageRecorder
from retrieval_layer.vector_search import VectorSearch

logger = get_logger(__name__)


class Retriever:
    """Query in; ranked, diverse passages that fit the token budget out.

    Every stage can be handed in, so tests run without a model or an index.
    """

    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        *,
        chunk_store_path: str | Path | None = None,
        keyword_index_path: str | Path = FTS_PATH,
        preparation: QueryPreparation | None = None,
        vector_search: VectorSearch | None = None,
        keyword_search: KeywordSearch | None = None,
        hydration: Hydration | None = None,
        reranker: Reranker | None = None,
        cache: ResultCache | None = None,
    ) -> None:
        self.settings = RetrievalSettings() if settings is None else settings
        if chunk_store_path is None:
            chunk_store_path = Config.DB_PATH
        self.__owned: list = []
        if keyword_search is None:
            mapping = VectorMetaDataRepository(str(chunk_store_path))
            keyword_search = KeywordSearch(
                mapping, chunk_store_path, keyword_index_path
            )
            self.__owned += [keyword_search, mapping]
            keyword_search.sync()
        self.keyword_search = keyword_search
        self.preparation = QueryPreparation() if preparation is None else preparation
        self.vector_search = VectorSearch() if vector_search is None else vector_search
        self.hydration = (
            Hydration(chunk_store_path) if hydration is None else hydration
        )
        self.reranker = (
            Reranker(self.settings.rerank_model, self.settings.rerank_batch)
            if reranker is None else reranker
        )
        self.cache = (
            ResultCache(self.settings.cache_size, self.settings.cache_ttl_seconds)
            if cache is None else cache
        )

    def retrieve(self, request: RetrievalRequest | str) -> RetrievalResult:
        """The passages that answer a query, best first, within the budget."""
        if isinstance(request, str):
            request = RetrievalRequest(query=request)
        request = request.with_defaults(self.settings)
        if request.top_k < 1:
            raise InvalidRetrievalSetting("top_k", request.top_k, "at least 1")
        if not isinstance(request.query, str) or not request.query.strip():
            raise EmptyQuery(request.query)

        recorder = StageRecorder()
        key = cache_key(request)
        with log_context(retrieval_id=new_correlation_id()):
            with recorder.stage("cache") as stage:
                hit = self.cache.get(key)
                stage.produced = 0 if hit is None else len(hit.passages)
            if hit is not None:
                recorder.log_breakdown()
                return hit._replace(
                    passages=list(hit.passages),
                    timings=recorder.timings,
                    cached=True,
                )
            result = self.__retrieve(request, recorder)
            self.cache.put(key, result._replace(passages=list(result.passages)))
            recorder.log_breakdown()
        return result

    def __retrieve(
        self, request: RetrievalRequest, recorder: StageRecorder
    ) -> RetrievalResult:
        settings = self.settings
        pool = request.top_k * settings.candidate_multiplier

        with recorder.stage("prepare") as stage:
            plan = self.preparation.prepare(request.query)
            stage.produced = 1
        # Sequential on purpose. Overlapping the searches can save at most the
        # dense one — about 30 µs at 200k vectors — and handing work to a
        # thread costs more than that.
        with recorder.stage("vector search") as stage:
            dense = self.vector_search.search(plan, pool)
            stage.produced = len(dense)
        with recorder.stage("keyword search") as stage:
            sparse = self.keyword_search.search(plan, pool)
            stage.produced = len(sparse)
        with recorder.stage("fusion") as stage:
            fused = reciprocal_rank_fusion([dense, sparse], settings.rrf_k)[:pool]
            stage.produced = len(fused)
        with recorder.stage("hydration") as stage:
            passages, unresolvable = self.hydration.hydrate(fused)
            passages, _ = deduplicate(passages)
            stage.produced = len(passages)
        if request.rerank:
            with recorder.stage("rerank") as stage:
                passages = self.reranker.rerank(plan, passages)
                stage.produced = len(passages)
        with recorder.stage("diversity") as stage:
            passages = maximal_marginal_relevance(
                plan,
                passages,
                request.top_k,
                settings.mmr_lambda,
                embed=self.preparation.encode,
                relevance=[passage.score for passage in passages],
            )
            stage.produced = len(passages)
        with recorder.stage("assembly") as stage:
            passages, tokens, _ = assemble(
                passages, settings.token_budget, settings.chars_per_token
            )
            stage.produced = len(passages)

        return RetrievalResult(
            passages=passages,
            timings=recorder.timings,
            candidates_considered=len(fused),
            dropped_unresolvable=unresolvable,
            tokens=tokens,
        )

    def refresh(self) -> int:
        """Catch up with an ingest. Returns how many chunks were newly indexed.

        All three, or the corpus is half-visible: new chunks keyword-searchable
        but absent from the vectors, or findable by both and masked by an
        answer cached before they existed. Only what is new is indexed; the
        graph is not rebuilt.
        """
        added = self.keyword_search.sync()
        self.vector_search.catch_up()
        self.cache.clear()
        logger.info("Retrieval refreshed; %d new chunk(s) indexed", added)
        return added

    def close(self) -> None:
        """Release what this retriever opened. What was handed in stays open."""
        for resource in self.__owned:
            resource.close()
        self.__owned = []

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_) -> None:
        self.close()
