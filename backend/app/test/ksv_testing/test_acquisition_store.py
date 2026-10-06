"""The record of what KSV has acquired and from where.

Owned by this subsystem rather than the memory layer: it answers a question
about acquisition, not about any conversation or project.
"""

import sqlite3

import pytest

from knowledge_sufficiency.acquisition_store import AcquiredRecord, AcquisitionStore


@pytest.fixture
def store(tmp_path):
    s = AcquisitionStore(tmp_path / "acquired.sql")
    yield s
    s.close()


class TestTheSchema:
    def test_the_table_exists_after_construction(self, store):
        with sqlite3.connect(store.db_path) as conn:
            tables = {r[0] for r in conn.execute(
                "select name from sqlite_master where type='table'"
            )}
        assert "acquired_knowledge" in tables

    def test_the_columns_are_as_specified(self, store):
        with sqlite3.connect(store.db_path) as conn:
            columns = [r[1] for r in
                       conn.execute("pragma table_info(acquired_knowledge)")]
        assert columns == ["id", "topic", "query", "url", "created_at"]

    def test_wal_is_in_force(self, store):
        assert store.journal_mode == "wal"

    def test_topic_query_and_url_are_required(self, store):
        with sqlite3.connect(store.db_path) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "insert into acquired_knowledge (topic, query, url) "
                    "values (null, 'q', 'u')"
                )


class TestRecording:
    def test_an_acquisition_gets_an_id(self, store):
        assert store.record("databases", "how does WAL work", "https://a/1") == 1

    def test_ids_increase(self, store):
        first = store.record("t", "q", "https://a/1")
        second = store.record("t", "q", "https://a/2")
        assert second > first

    def test_an_id_is_never_reused_after_a_delete(self, store):
        """AUTOINCREMENT, not a bare INTEGER PRIMARY KEY.

        A reused id would point at two different documents over the table's
        life, which is worse for a provenance record than a gap in the numbers.
        """
        first = store.record("t", "q", "https://a/1")
        with sqlite3.connect(store.db_path) as conn:
            conn.execute("delete from acquired_knowledge where id = ?", (first,))
            conn.commit()
        assert store.record("t", "q", "https://a/2") > first

    def test_what_went_in_reads_back(self, store):
        store.record("databases", "how does WAL work", "https://sqlite.org/wal.html")
        record = store.all_records()[0]
        assert record.topic == "databases"
        assert record.query == "how does WAL work"
        assert record.url == "https://sqlite.org/wal.html"

    def test_the_same_url_can_be_recorded_twice(self, store):
        """It is a log of acquisitions, not an index of documents held.

        The same page fetched for a different query is a different acquisition,
        and collapsing them would lose which query it answered.
        """
        store.record("t", "first query", "https://a/1")
        store.record("t", "second query", "https://a/1")
        assert len(store.all_records()) == 2


class TestWhenItWasAcquired:
    def test_a_record_is_stamped(self, store):
        store.record("t", "q", "https://a/1")
        assert store.all_records()[0].created_at

    def test_the_stamp_is_sortable_iso_8601_utc(self, store):
        """Stored as TEXT, so it has to sort correctly as text."""
        from datetime import datetime

        store.record("t", "q", "https://a/1")
        stamp = store.all_records()[0].created_at
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None
        assert stamp == parsed.isoformat(timespec="microseconds")

    def test_later_records_are_stamped_later(self, store):
        store.record("t", "q", "https://a/1")
        store.record("t", "q", "https://a/2")
        first, second = store.all_records()
        assert first.created_at <= second.created_at

    def test_the_stamp_is_required(self, store):
        with sqlite3.connect(store.db_path) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "insert into acquired_knowledge (topic, query, url, created_at) "
                    "values ('t', 'q', 'u', null)"
                )

    def test_one_batch_shares_one_stamp(self, store):
        """They came from a single acquisition; the id already orders them."""
        store.record_many("t", "q", ["https://a/1", "https://a/2", "https://a/3"])
        stamps = {r.created_at for r in store.all_records()}
        assert len(stamps) == 1

    def test_the_project_timestamp_helper_is_used(self, store):
        """Not a second format. utc_now exists because it was once four.

        bug 4.58 was utc_now defined over and over; a copy here would be the
        fifth, and two formats in one database do not sort against each other.
        """
        from storage.timestamps import utc_now

        store.record("t", "q", "https://a/1")
        stamp = store.all_records()[0].created_at
        assert len(stamp) == len(utc_now())


class TestRecordingSeveral:
    def test_one_call_records_each_url(self, store):
        ids = store.record_many("t", "q", ["https://a/1", "https://a/2"])
        assert len(ids) == 2
        assert len(store.all_records()) == 2

    def test_the_returned_ids_match_the_rows(self, store):
        ids = store.record_many("t", "q", ["https://a/1", "https://a/2"])
        assert [r.id for r in store.all_records()] == ids

    def test_nothing_to_record_is_not_an_error(self, store):
        assert store.record_many("t", "q", []) == []
        assert store.all_records() == []

    def test_it_continues_from_the_existing_ids(self, store):
        store.record("t", "q", "https://a/0")
        ids = store.record_many("t", "q", ["https://a/1"])
        assert ids == [2]


class TestReading:
    def test_a_topic_sees_only_its_own(self, store):
        store.record("databases", "q", "https://a/1")
        store.record("web", "q", "https://b/1")
        assert [r.url for r in store.for_topic("databases")] == ["https://a/1"]

    def test_a_topic_with_nothing_reads_empty(self, store):
        assert store.for_topic("never used") == []

    def test_records_come_back_oldest_first(self, store):
        for i in range(3):
            store.record("t", "q", f"https://a/{i}")
        assert [r.url for r in store.for_topic("t")] == [
            "https://a/0", "https://a/1", "https://a/2",
        ]

    def test_all_records_come_back_oldest_first(self, store):
        """Urls chosen so alphabetical order contradicts insertion order."""
        store.record("t", "q", "https://zebra/1")
        store.record("t", "q", "https://apple/2")

        assert [r.url for r in store.all_records()] == [
            "https://zebra/1", "https://apple/2",
        ]

    def test_seen_reports_a_known_url(self, store):
        store.record("t", "q", "https://a/1")
        assert store.seen("https://a/1")

    def test_seen_reports_an_unknown_url(self, store):
        assert not store.seen("https://never/fetched")

    def test_a_record_is_a_named_tuple(self, store):
        store.record("t", "q", "https://a/1")
        record = store.all_records()[0]
        assert isinstance(record, AcquiredRecord)
        _id, topic, query, url, created_at = record


class TestLifecycle:
    def test_two_stores_share_one_file(self, tmp_path):
        first = AcquisitionStore(tmp_path / "acquired.sql")
        first.record("t", "q", "https://a/1")
        second = AcquisitionStore(tmp_path / "acquired.sql")
        assert len(second.all_records()) == 1
        first.close()
        second.close()

    def test_close_is_safe_to_call_twice(self, tmp_path):
        store = AcquisitionStore(tmp_path / "acquired.sql")
        store.close()
        store.close()

    def test_records_survive_reopening(self, tmp_path):
        store = AcquisitionStore(tmp_path / "acquired.sql")
        store.record("t", "q", "https://a/1")
        store.close()

        reopened = AcquisitionStore(tmp_path / "acquired.sql")
        assert len(reopened.all_records()) == 1
        reopened.close()
