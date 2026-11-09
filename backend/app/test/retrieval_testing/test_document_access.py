"""Private uploads: who may retrieve which document, and the retriever honouring it."""

import sqlite3

import numpy as np
import pytest

from data_layer.ingestion.document_owners import DocumentOwners
from retrieval_layer.models import KEYWORD, VECTOR, Passage, RetrievalRequest, ScoredId
from retrieval_layer.query_preparation import QueryPreparation
from retrieval_layer.reranking import Reranker
from retrieval_layer.retriever import Retriever
from retrieval_layer.settings import RetrievalSettings

ALICE, BOB = "alice-uuid", "bob-uuid"


@pytest.fixture
def store(tmp_path):
    """A chunk store with one sectioned document (context -> section -> document)."""
    path = tmp_path / "chunks.db"
    connection = sqlite3.connect(path)
    connection.execute("create table Sections(sectionId text primary key, documentId text)")
    connection.execute("create table Contexts(contextId text primary key, sectionId text)")
    connection.execute("insert into Sections values ('s1', 'doc-sectioned')")
    connection.execute("insert into Contexts values ('ctx1', 's1')")
    connection.commit()
    connection.close()
    return path


class TestDocumentOwners:
    def test_without_any_owner_everything_is_public(self, store):
        owners = DocumentOwners(store)
        assert not owners.any_private()
        assert owners.visible(["doc-a", "ctx1"], None) == {"doc-a", "ctx1"}

    def test_a_private_document_is_visible_to_its_owners_only(self, store):
        owners = DocumentOwners(store)
        assert owners.add("doc-private", ALICE) is True
        assert owners.add("doc-private", ALICE) is False
        assert owners.any_private()
        parents = ["doc-private", "doc-public"]
        assert owners.visible(parents, ALICE) == {"doc-private", "doc-public"}
        assert owners.visible(parents, BOB) == {"doc-public"}
        assert owners.visible(parents, None) == {"doc-public"}

    def test_two_owners_of_one_identical_upload_both_see_it(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-shared", ALICE)
        owners.add("doc-shared", BOB)
        assert owners.visible(["doc-shared"], BOB) == {"doc-shared"}
        assert owners.owners("doc-shared") == {ALICE, BOB}

    def test_a_context_id_is_judged_by_its_document(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-sectioned", ALICE)
        assert owners.visible(["ctx1"], ALICE) == {"ctx1"}
        assert owners.visible(["ctx1"], BOB) == set()

    def test_an_unknown_parent_is_dropped_never_guessed(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-private", ALICE)
        assert owners.visible([None, ""], ALICE) == set()

    def test_removing_the_last_owner_makes_it_public_again(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-x", ALICE)
        owners.remove("doc-x", ALICE)
        assert owners.visible(["doc-x"], BOB) == {"doc-x"}

    def test_no_store_yet_means_nothing_is_private(self, tmp_path):
        owners = DocumentOwners(tmp_path / "absent.db")
        assert not owners.any_private()
        assert owners.visible(["doc"], None) == {"doc"}


class Search:
    def __init__(self, ids, source):
        self.ids, self.source = ids, source
        self.asked_for = []

    def search(self, plan, k):
        self.asked_for.append(k)
        return [ScoredId(i, 1.0 / r, self.source, r) for r, i in enumerate(self.ids[:k], start=1)]

    def catch_up(self):
        return 0

    def sync(self):
        return 0

    def close(self):
        pass


class Hydration:
    """Vector id n belongs to document doc-n."""

    def hydrate(self, scored):
        return [Passage(s.vector_id, f"c{s.vector_id}", f"passage {s.vector_id}", s.score, s.source,
                        f"doc-{s.vector_id}") for s in scored], 0


def build(owners, ids=(1, 2, 3, 4)):
    vector = Search(list(ids), VECTOR)
    rng = np.random.default_rng(0)
    return Retriever(
        RetrievalSettings(rerank=False, top_k=4),
        preparation=QueryPreparation(lambda texts: rng.normal(size=(len(texts), 8)).astype(np.float32)),
        vector_search=vector,
        keyword_search=Search(list(ids), KEYWORD),
        hydration=Hydration(),
        reranker=Reranker(scorer=lambda pairs: [0.0] * len(pairs)),
        owners=owners,
    ), vector


class TestRetrieverAccess:
    def test_another_users_upload_never_comes_back(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-2", ALICE)
        owners.add("doc-3", BOB)
        retriever, _ = build(owners)

        def documents(viewer):
            result = retriever.retrieve(RetrievalRequest("query", viewer=viewer))
            return {p.document_id for p in result.passages}

        assert documents(ALICE) == {"doc-1", "doc-2", "doc-4"}
        assert documents(BOB) == {"doc-1", "doc-3", "doc-4"}
        assert documents(None) == {"doc-1", "doc-4"}

    def test_one_users_cached_answer_is_not_served_to_another(self, store):
        owners = DocumentOwners(store)
        owners.add("doc-2", ALICE)
        retriever, _ = build(owners)
        first = retriever.retrieve(RetrievalRequest("same words", viewer=ALICE))
        second = retriever.retrieve(RetrievalRequest("same words", viewer=BOB))
        assert not second.cached
        assert "doc-2" in {p.document_id for p in first.passages}
        assert "doc-2" not in {p.document_id for p in second.passages}

    def test_private_documents_widen_the_candidate_pool(self, store):
        public_only, search = build(DocumentOwners(store))
        public_only.retrieve("q")
        owners = DocumentOwners(store)
        owners.add("doc-9", ALICE)
        with_private, wider = build(owners)
        with_private.retrieve("q")
        assert wider.asked_for[0] == 2 * search.asked_for[0]
