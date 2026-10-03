"""The project snapshot registry: metadata plus the append-only mapping.

`project_snapshot_mapping` is append-only by decision, so it holds the ordered
chain of every snapshot a project has had rather than a pointer at the current
one. The watermark on each row is the conversation-snapshot `seq` it folded in,
which is what makes the next snapshot incremental.
"""

import pytest

from config import Config
from memory.memory_pool_exceptions import InvalidIdentifier
from memory.topic_pool.project_pool.project_data_repo.project_snapshot_repo import (
    ProjectSnapshotRepository,
    project_snapshot_id,
)


@pytest.fixture
def repo(tmp_path):
    r = ProjectSnapshotRepository("proj_1", db_path=tmp_path / "project.sql")
    yield r
    r.close()


class TestTheId:
    def test_it_fits_the_pgvector_column(self):
        """It is also the vector_id, which is a signed 64-bit bigint."""
        value = project_snapshot_id("p", "2026-10-01", "a summary")
        assert 0 <= value <= Config.VECTOR_ID_MASK

    def test_it_is_stable_for_the_same_inputs(self):
        args = ("p", "2026-10-01", "a summary")
        assert project_snapshot_id(*args) == project_snapshot_id(*args)

    def test_the_same_text_at_a_different_time_is_a_different_snapshot(self):
        """A project summarised twice to the same words is still two snapshots."""
        assert project_snapshot_id("p", "2026-10-01", "same") != \
               project_snapshot_id("p", "2026-10-02", "same")

    def test_two_projects_do_not_collide(self):
        assert project_snapshot_id("p1", "2026-10-01", "same") != \
               project_snapshot_id("p2", "2026-10-01", "same")


class TestAnEmptyProject:
    def test_it_has_no_latest(self, repo):
        assert repo.latest() is None

    def test_its_watermark_is_zero(self, repo):
        """So the first snapshot folds in everything written so far."""
        assert repo.last_seq_included() == 0

    def test_its_history_is_empty(self, repo):
        assert repo.history() == []


class TestStoringSnapshots:
    def test_the_watermark_is_what_was_folded_in(self, repo):
        repo.add_snapshot("about X", last_seq_included=3)
        assert repo.last_seq_included() == 3

    def test_the_latest_is_the_one_with_the_highest_watermark(self, repo):
        repo.add_snapshot("about X", last_seq_included=3)
        repo.add_snapshot("X and Y", last_seq_included=7)
        assert repo.latest().summary == "X and Y"

    def test_the_mapping_keeps_the_whole_chain(self, repo):
        """Append-only: an overwritten pointer would lose the earlier ones."""
        repo.add_snapshot("first", last_seq_included=1)
        repo.add_snapshot("second", last_seq_included=2)
        repo.add_snapshot("third", last_seq_included=3)

        assert [row.summary for row in repo.history()] == [
            "first", "second", "third",
        ]

    def test_a_snapshot_reads_back_by_its_id(self, repo):
        snapshot_id = repo.add_snapshot("about X", last_seq_included=3)
        row = repo.get_snapshot(snapshot_id)

        assert row.summary == "about X"
        assert row.last_seq_included == 3
        assert row.project_id == "proj_1"

    def test_the_length_is_stored(self, repo):
        snapshot_id = repo.add_snapshot("abcde", last_seq_included=1)
        assert repo.get_snapshot(snapshot_id).len_of_the_summary == 5

    def test_an_unknown_id_reads_back_as_none(self, repo):
        assert repo.get_snapshot(123456789) is None

    def test_re_adding_the_identical_snapshot_is_a_no_op(self, repo):
        """Same project, same timestamp, same text: one snapshot, not two."""
        first = repo.add_snapshot("about X", last_seq_included=3,
                                  created_at="2026-10-01T00:00:00+00:00")
        again = repo.add_snapshot("about X", last_seq_included=3,
                                  created_at="2026-10-01T00:00:00+00:00")

        assert first == again
        assert len(repo.history()) == 1


class TestProjectsAreIsolated:
    def test_one_project_does_not_see_another(self, tmp_path):
        db = tmp_path / "project.sql"
        one = ProjectSnapshotRepository("proj_1", db_path=db)
        two = ProjectSnapshotRepository("proj_2", db_path=db)
        one.add_snapshot("about one", last_seq_included=5)
        two.add_snapshot("about two", last_seq_included=9)

        assert [r.summary for r in one.history()] == ["about one"]
        assert [r.summary for r in two.history()] == ["about two"]
        assert one.last_seq_included() == 5
        assert two.last_seq_included() == 9
        one.close()
        two.close()

    def test_they_share_the_one_registry_file(self, tmp_path):
        db = tmp_path / "project.sql"
        one = ProjectSnapshotRepository("proj_1", db_path=db)
        two = ProjectSnapshotRepository("proj_2", db_path=db)
        assert one.db_path == two.db_path
        one.close()
        two.close()


class TestItRefusesABadProjectId:
    @pytest.mark.parametrize("value", ["", "   ", None])
    def test_it_raises(self, tmp_path, value):
        with pytest.raises(InvalidIdentifier):
            ProjectSnapshotRepository(value, db_path=tmp_path / "project.sql")
