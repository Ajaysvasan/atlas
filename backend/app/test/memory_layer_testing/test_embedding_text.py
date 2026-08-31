"""
Tests for the raw-text embedding API the memory layer needs.

The ingestion pipeline only ever embeds HChunk/RChunk. Conversation summaries
and turns never pass through the Chunker, so they need embed_text/embed_texts.

The SentenceTransformer itself is stubbed — these cover id generation, argument
validation and batching, not the quality of the vectors.
"""

import hashlib
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from data_layer.datalayer_exceptions.datalayer_exceptions import (
    InvalidEmbeddingArgument,
)

_MODEL = "data_layer.ingestion.embedding.EmbeddingManager.SentenceTransformer"

SIGNED_64_MAX = 2**63 - 1


@pytest.fixture
def manager():
    with patch(_MODEL) as MockST:
        MockST.return_value.encode.side_effect = lambda text, **kw: (
            np.arange(128, dtype=np.float32)
            if isinstance(text, str)
            else np.zeros((len(text), 128), dtype=np.float32)
        )
        from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager

        yield EmbeddingManager()


# ---------------------------------------------------------------------------
# Vector id range
#
# Vector ids land in SQLite INTEGER and Postgres bigint, both signed 64-bit.
# The unsigned MD5-derived id overflowed both for ~49% of inputs.
# ---------------------------------------------------------------------------

class TestVectorIdRange:
    def test_ids_fit_signed_64_bit(self, manager):
        for i in range(2000):
            vid = manager.embed_text(f"chunk number {i}").vector_id
            assert 0 <= vid <= SIGNED_64_MAX, f"{vid} is out of range for bigint"

    def test_ids_are_non_negative(self, manager):
        """Masking, not signing — negative ids would be legal but surprising."""
        for i in range(500):
            assert manager.embed_text(f"text {i}").vector_id >= 0

    def test_id_is_storable_in_sqlite(self, manager):
        """The exact failure that aborted snapshot writes before the fix."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE v (vector_id INTEGER PRIMARY KEY)")
        for i in range(500):
            vid = manager.embed_text(f"payload {i}").vector_id
            conn.execute("INSERT OR IGNORE INTO v VALUES (?)", (vid,))

    def test_id_matches_masked_md5_of_the_chunk_id(self, manager):
        """The id is derived from the chunk id, not the text: identical text
        under two different chunk ids has to reach two different vectors."""
        chunk_id = hashlib.sha256(b"deterministic").hexdigest()
        digest = hashlib.md5(chunk_id.encode("utf-8")).digest()
        expected = int.from_bytes(digest[:8], "little", signed=False) & SIGNED_64_MAX
        assert manager.embed_text("deterministic").vector_id == expected

    def test_same_text_under_different_chunk_ids_gets_different_ids(self, manager):
        assert (
            manager.embed_text("repeated", chunk_id="a").vector_id
            != manager.embed_text("repeated", chunk_id="b").vector_id
        )

    def test_id_is_deterministic(self, manager):
        assert (
            manager.embed_text("same text").vector_id
            == manager.embed_text("same text").vector_id
        )

    def test_different_text_gives_different_id(self, manager):
        assert (
            manager.embed_text("alpha").vector_id
            != manager.embed_text("beta").vector_id
        )


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_float32_vector(self, manager):
        result = manager.embed_text("hello")
        assert result.vector.dtype == np.float32

    def test_chunk_id_defaults_to_content_hash(self, manager):
        result = manager.embed_text("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert result.meta_data.chunk_id == expected

    def test_explicit_chunk_id_is_kept(self, manager):
        result = manager.embed_text("hello", chunk_id="turn_42")
        assert result.meta_data.chunk_id == "turn_42"

    def test_text_is_carried_into_metadata(self, manager):
        assert manager.embed_text("carry me").meta_data.chunk == "carry me"

    @pytest.mark.parametrize("bad", ["", "   ", "\n", None, 42, b"bytes", ["list"]])
    def test_invalid_input_rejected(self, manager, bad):
        with pytest.raises(InvalidEmbeddingArgument):
            manager.embed_text(bad)


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    def test_returns_one_result_per_text(self, manager):
        assert len(manager.embed_texts(["a", "b", "c"])) == 3

    def test_empty_list_returns_empty(self, manager):
        assert manager.embed_texts([]) == []

    def test_chunk_ids_are_applied_in_order(self, manager):
        results = manager.embed_texts(["a", "b"], ["id_a", "id_b"])
        assert [r.meta_data.chunk_id for r in results] == ["id_a", "id_b"]

    def test_mismatched_chunk_ids_rejected(self, manager):
        with pytest.raises(InvalidEmbeddingArgument):
            manager.embed_texts(["a", "b", "c"], ["only_one"])

    def test_omitted_chunk_ids_fall_back_to_hashes(self, manager):
        results = manager.embed_texts(["a", "b"])
        assert results[0].meta_data.chunk_id == hashlib.sha256(b"a").hexdigest()

    def test_invalid_entry_rejected(self, manager):
        with pytest.raises(InvalidEmbeddingArgument):
            manager.embed_texts(["fine", ""])

    def test_repeated_text_yields_repeated_ids(self, manager):
        """
        Ids are content-derived, so overlapping snapshot windows re-emit the same
        id for the same chunk. The metadata inserts are INSERT OR IGNORE for
        exactly this reason.
        """
        results = manager.embed_texts(["dup", "dup"])
        assert results[0].vector_id == results[1].vector_id
