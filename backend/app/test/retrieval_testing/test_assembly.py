"""Deduplicating and budgeting the passages that become the context."""

import pytest

from retrieval_layer.assembly import (
    assemble,
    deduplicate,
    fit_to_budget,
    in_reading_order,
)
from retrieval_layer.models import Passage, VECTOR


def passage(i, text="text", chunk_id=None, document_id="d", offset=0, score=1.0):
    return Passage(i, chunk_id or f"c{i}", text, score, VECTOR,
                   document_id=document_id, start_offset=offset)


class TestDeduplicating:
    def test_the_same_chunk_found_twice_appears_once(self):
        """Both searchers can return the same chunk; fusion keeps one id."""
        kept, dropped = deduplicate([passage(1, chunk_id="c1"),
                                     passage(2, chunk_id="c1")])
        assert len(kept) == 1 and dropped == 1

    def test_identical_text_under_different_ids_appears_once(self):
        """Windowing repeats content at chunk boundaries on purpose."""
        kept, dropped = deduplicate([passage(1, "same words"),
                                     passage(2, "same words")])
        assert len(kept) == 1 and dropped == 1

    def test_whitespace_differences_still_count_as_identical(self):
        kept, _ = deduplicate([passage(1, "same  words"),
                               passage(2, "same\nwords")])
        assert len(kept) == 1

    def test_the_first_copy_is_the_one_kept(self):
        """The list arrives ranked, so the first is the best-scoring."""
        kept, _ = deduplicate([passage(1, "same", score=0.9),
                               passage(2, "same", score=0.1)])
        assert kept[0].vector_id == 1

    def test_different_text_is_left_alone(self):
        kept, dropped = deduplicate([passage(1, "one"), passage(2, "two")])
        assert len(kept) == 2 and dropped == 0

    def test_nothing_in_nothing_out(self):
        assert deduplicate([]) == ([], 0)


class TestTheBudget:
    def test_it_stops_when_the_budget_is_spent(self):
        passages = [passage(i, "a" * 400) for i in range(1, 6)]
        kept, spent = fit_to_budget(passages, token_budget=200)
        assert len(kept) == 2 and spent == 200

    def test_a_passage_is_never_cut_in_half(self):
        """Half a passage reads as though the source said something it did not."""
        kept, spent = fit_to_budget([passage(1, "a" * 400)], token_budget=50)
        assert kept == [] and spent == 0

    def test_a_smaller_later_passage_does_not_jump_the_queue(self):
        """Taking it would reorder the context by size rather than by rank."""
        passages = [passage(1, "a" * 4000), passage(2, "b" * 40)]
        kept, _ = fit_to_budget(passages, token_budget=100)
        assert kept == []

    def test_everything_fits_when_the_budget_is_large(self):
        passages = [passage(i, "a" * 40) for i in range(1, 4)]
        kept, _ = fit_to_budget(passages, token_budget=10_000)
        assert len(kept) == 3

    def test_the_spend_is_reported(self):
        kept, spent = fit_to_budget([passage(1, "a" * 400)], token_budget=1000)
        assert spent == 100


class TestReadingOrder:
    def test_chunks_of_one_document_come_back_in_sequence(self):
        out = in_reading_order([passage(1, offset=900), passage(2, offset=100)])
        assert [p.vector_id for p in out] == [2, 1]

    def test_documents_do_not_interleave(self):
        out = in_reading_order([
            passage(1, document_id="b", offset=0),
            passage(2, document_id="a", offset=900),
            passage(3, document_id="a", offset=100),
        ])
        assert [p.document_id for p in out] == ["a", "a", "b"]

    def test_a_missing_offset_does_not_raise(self):
        out = in_reading_order([Passage(1, "c1", "t", 1.0, VECTOR)])
        assert len(out) == 1


class TestAssembling:
    def test_it_deduplicates_then_budgets(self):
        passages = [passage(1, "a" * 400), passage(2, "a" * 400),
                    passage(3, "b" * 400), passage(4, "c" * 400)]
        kept, spent, duplicates = assemble(passages, token_budget=200)
        assert duplicates == 1
        assert len(kept) == 2 and spent == 200

    def test_the_ranking_is_the_default_order(self):
        passages = [passage(1, "one", offset=900), passage(2, "two", offset=0)]
        kept, _, _ = assemble(passages, token_budget=1000)
        assert [p.vector_id for p in kept] == [1, 2]

    def test_reading_order_is_available_when_asked_for(self):
        passages = [passage(1, "one", offset=900), passage(2, "two", offset=0)]
        kept, _, _ = assemble(passages, token_budget=1000, reading_order=True)
        assert [p.vector_id for p in kept] == [2, 1]

    def test_nothing_in_nothing_out(self):
        assert assemble([]) == ([], 0, 0)
