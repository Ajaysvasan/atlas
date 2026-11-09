"""Stand-ins for each layer at its public interface, so the pipeline's own decisions can be tested."""

import json
from typing import Dict, List

import numpy as np

from config import Config
from inference.sessions import CacheReport, SessionGeneration
from knowledge_sufficiency.verdict import Sufficiency
from memory.memory_manager import KNOWN, NO_TOPIC, ConversationContext, Route
from query_decomposition.models import Decomposition, Mode, TopicPath
from retrieval_layer.models import Passage, RetrievalResult

DIM = Config.EMBEDDING_DIMENSIONS


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class Encoder:
    """The query and anything containing `relevant` embed together; everything else is unrelated."""

    def __init__(self, relevant: str = "MVCC") -> None:
        self.relevant = relevant
        self.query = unit(1)

    def encode(self, texts):
        out = []
        for i, text in enumerate(texts):
            if i == 0 or self.relevant in text:
                out.append(self.query)
            else:
                v = unit(1000 + len(text))
                v = v - self.query * float(v @ self.query)
                out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


class Models:
    def __init__(self, encoder: Encoder | None = None) -> None:
        self.preparation = encoder or Encoder()


class FakeDecomposer:
    def __init__(self, mode: Mode = Mode.DOUBT, sub_queries=()) -> None:
        self.mode = mode
        self.sub_queries = tuple(sub_queries)
        self.responses: Dict[str, str] = {}
        self.queries: List[tuple] = []
        self.fail_record = False

    def decompose(self, query, conversation_id, user_id) -> Decomposition:
        self.queries.append((query, conversation_id, user_id))
        return Decomposition(f"x{len(self.queries)}", query, query, self.mode,
                             (TopicPath("Databases", ("PostgreSQL",)),), True, self.sub_queries)

    def record_response(self, exchange_id, text):
        if self.fail_record:
            raise ConnectionError("PostgreSQL is down")
        self.responses[exchange_id] = text


class FakeMemory:
    """MemoryManager's contract over a dict: routes, recalls and stores."""

    store: Dict[str, dict] = {}
    fail_update = False
    recall: List[Passage] = []
    sufficiency = None

    def __init__(self, query, conversation_id, user_id, topic, **kwargs) -> None:
        self.query = query
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.topic = topic
        entry = FakeMemory.store.get(conversation_id)
        self.route = (Route(KNOWN, "t1", "p1", entry["project"], None, None) if entry
                      else Route(NO_TOPIC, None, None, None, None, None))
        self.sufficiency = FakeMemory.sufficiency

    def conversation_retrieval(self):
        return list(FakeMemory.recall) or None

    def conversation_context(self, max_turns):
        entry = FakeMemory.store.get(self.conversation_id)
        turns = tuple(entry["turns"][-max_turns:]) if entry else ()
        return ConversationContext(entry and entry.get("summary"), turns, len(entry["turns"]) if entry else 0)

    def __store(self, text, project):
        if FakeMemory.fail_update:
            raise ConnectionError("memory store down")
        entry = FakeMemory.store.setdefault(self.conversation_id, {"project": project, "turns": []})
        entry["turns"] += [("user", self.query), ("assistant", text)]

    def update(self, text):
        self.__store(text, None)

    def store_new_topic(self, text, project_name):
        self.__store(text, project_name)
        FakeMemory.store[self.conversation_id]["created"] = ("topic", self.topic, project_name)

    def store_new_project(self, text, project_name):
        self.__store(text, project_name)
        FakeMemory.store[self.conversation_id]["created"] = ("project", self.topic, project_name)


def passage(vector_id: int, text: str, score: float = 1.0, document_id="doc1") -> Passage:
    return Passage(vector_id, f"chunk{vector_id}", text, score, "fused", document_id)


class FakeRetriever:
    def __init__(self, passages=(), fail: Exception | None = None) -> None:
        self.passages = list(passages)
        self.fail = fail
        self.queries: List[str] = []
        self.viewers: List[str | None] = []
        self.refreshed = 0

    def retrieve(self, request):
        if self.fail:
            raise self.fail
        text = getattr(request, "query", request)
        self.queries.append(text)
        self.viewers.append(getattr(request, "viewer", None))
        return RetrievalResult(list(self.passages), [])

    def refresh(self):
        self.refreshed += 1


class Handle:
    def __init__(self, retriever) -> None:
        self.retriever = retriever

    def get(self):
        return self.retriever

    def refresh(self):
        self.retriever.refresh()

    def close(self):
        pass


class FakeSessions:
    """The KV session manager's contract: answers, and reports a cache outcome."""

    def __init__(self, text="The final answer [G1].", fail: Exception | None = None) -> None:
        self.text = text
        self.fail = fail
        self.calls: List[dict] = []

    def generate(self, conversation_id, user_id, prefix, turn, *, max_tokens, temperature=0.0):
        if self.fail:
            raise self.fail
        self.calls.append({"conversation_id": conversation_id, "prefix": prefix, "turn": turn})
        report = CacheReport(conversation_id, "miss", None, 100, 100, 0, 0, 0.0, 1.0, 50, 1000, None)
        return SessionGeneration(self.text, "stop", 10, report)


class NamingModel:
    def __init__(self, names=("PostgreSQL MVCC",), fail=False) -> None:
        self.names = list(names)
        self.fail = fail
        self.calls = 0

    def infer(self, system, user, **options):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model down")
        name = self.names[min(self.calls - 1, len(self.names) - 1)]
        return json.dumps({"name": name})


class FakeAcquirer:
    def __init__(self, on_acquire=None) -> None:
        self.calls = []
        self.on_acquire = on_acquire

    def acquire(self, topic, query, subtopics=()):
        from pipeline.acquisition import AcquisitionOutcome

        self.calls.append((topic, query))
        if self.on_acquire:
            self.on_acquire()
        return AcquisitionOutcome("full", 1, 1, ("https://www.postgresql.org/docs/x",), 0, None)


__all__ = ["Sufficiency"]
