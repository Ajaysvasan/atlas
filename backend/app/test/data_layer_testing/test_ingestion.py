"""Regression tests for the ingestion pipeline: document coverage and chunking.

The bugs pinned here were all reachable from a plain `python main.py` run over a
folder of ordinary documents, and most of them lost data quietly rather than
raising.
"""

import importlib
import os
import sqlite3
import sys
import unittest.mock as mock

import pytest

sys.modules.setdefault("diskannpy", mock.MagicMock())

from data_layer.datalayer_exceptions.datalayer_exceptions import InvalidFileType
from data_layer.ingestion.Chunker.chunker import Chunker
from data_layer.ingestion.Chunker.HierarchicalChunker import HierarchicalChunker
from data_layer.ingestion.Chunker.RecursiveChunker import RecursiveChunker
from data_layer.ingestion.Chunker.windowing import sliding_windows
from data_layer.ingestion.normalizer.normalizer import (
    NormalizationProfiles,
    TextNormalizer,
    _heading_name,
)
from data_layer.ingestion.TextFileProcessor.file_loader import FileLoader
from data_layer.ingestion.TextFileProcessor.text_extractor import TextExtractor

SECTIONED = (
    "INTRODUCTION\n\n"
    "This is the first paragraph of the intro.\n\n"
    "This is the second paragraph.\n\n"
    "METHODS\n\n"
    "We did things here.\n\n"
    "NOTES\n\n"
    "Some notes.\n"
)


@pytest.fixture
def normalizer():
    return NormalizationProfiles.rag_ingestion()


@pytest.fixture
def sectioned(normalizer):
    return normalizer.normalize_text("report.txt", SECTIONED)


class TestNormalizationPreservesStructure:
    """The whitespace collapse used to flatten every document to a single line,
    after which no heading, paragraph or "\\n\\n" separator could ever match."""

    def test_paragraph_breaks_survive(self, sectioned):
        assert "\n\n" in sectioned.content

    def test_sections_are_found(self, sectioned):
        assert sectioned.has_section
        assert [s.name for s in sectioned.sections] == [
            "INTRODUCTION",
            "METHODS",
            "NOTES",
        ]

    def test_section_spans_locate_their_body(self, sectioned):
        first = sectioned.sections[0]
        body = sectioned.content[first.content_start : first.content_end]
        assert "first paragraph" in body
        assert "second paragraph" in body
        assert "METHODS" not in body

    def test_last_section_runs_to_the_end(self, sectioned):
        assert sectioned.sections[-1].content_end == len(sectioned.content)

    def test_has_section_agrees_with_the_spans(self, normalizer):
        flat = normalizer.normalize_text("flat.txt", "Just a sentence. And more.")
        assert flat.has_section is False
        assert flat.sections == ()

    def test_horizontal_whitespace_still_collapses(self, normalizer):
        result = normalizer.normalize_text("s.txt", "a     b\t\tc")
        assert result.content == "a b c"

    def test_empty_document(self, normalizer):
        result = normalizer.normalize_text("empty.txt", "")
        assert result.content == ""
        assert result.sections == ()

    def test_windows_line_endings(self, normalizer):
        result = normalizer.normalize_text("crlf.txt", "NOTES\r\n\r\nBody here.\r\n")
        assert "\r" not in result.content
        assert [s.name for s in result.sections] == ["NOTES"]


class TestHeadingDetection:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("# Getting Started", "Getting Started"),
            ("### Deep Section", "Deep Section"),
            ("1. Introduction", "1. Introduction"),
            ("2.3 Related Work", "2.3 Related Work"),
            ("BACKGROUND", "BACKGROUND"),
            ("HELLO WORLD", "HELLO WORLD"),
        ],
    )
    def test_recognised(self, line, expected):
        assert _heading_name(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "This is ordinary prose.",
            "1. The cat sat on the mat and went home.",
            "Q1 | 1.2M",
            "1.2M | 3.4M",
            "A",
            "",
            "   ",
            "A" * 100,
        ],
    )
    def test_rejected(self, line):
        assert _heading_name(line) is None

    def test_setext_underline_is_dropped(self, normalizer):
        result = normalizer.normalize_text("s.md", "Overview\n========\n\nBody.\n")
        assert [s.name for s in result.sections] == ["Overview"]
        assert "=====" not in result.content

    def test_code_fence_hides_headings(self, normalizer):
        result = normalizer.normalize_text(
            "s.md", "Intro line.\n\n```\n# not a heading\n```\n"
        )
        assert result.sections == ()

    def test_uppercase_heading_survives_a_lowercasing_profile(self):
        """Detection runs before the lexical transforms, so a profile that
        lowercases cannot destroy the signal it is routed on."""
        result = TextNormalizer(lowercase=True).normalize_text(
            "s.txt", "HELLO SECTION\n\nSome content."
        )
        assert result.has_section
        assert result.content.islower()


class TestSlidingWindows:
    def test_never_exceeds_size(self):
        text = "word " * 200
        assert all(end - start <= 30 for start, end in sliding_windows(text, 30, 5))

    def test_covers_the_whole_text(self):
        text = "alpha beta gamma delta epsilon zeta"
        spans = sliding_windows(text, 12, 0)
        assert "".join(text[s:e] for s, e in spans).replace(" ", "") == text.replace(
            " ", ""
        )

    def test_breaks_on_word_boundaries(self):
        text = "alpha beta gamma delta epsilon"
        spans = sliding_windows(text, 12, 0)
        assert all(text[s:e].strip() == text[s:e] for s, e in spans)
        assert not any(text[s:e].endswith(("alph", "bet", "gamm")) for s, e in spans)

    def test_text_with_no_whitespace_is_split_hard(self):
        spans = sliding_windows("A" * 50, 10, 0)
        assert [e - s for s, e in spans] == [10, 10, 10, 10, 10]

    def test_terminates_when_a_window_is_shorter_than_the_overlap(self):
        """A window pulled back to a word boundary can be shorter than the
        overlap; stepping back by the overlap would then never advance."""
        assert sliding_windows("ab cd ef gh ij kl", 6, 5)

    @pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (10, 10), (10, 11), (10, -1)])
    def test_rejects_impossible_geometry(self, size, overlap):
        with pytest.raises(ValueError):
            sliding_windows("some text", size, overlap)


class TestRecursiveChunker:
    def test_respects_chunk_size(self, normalizer):
        content = normalizer.normalize_text("s.txt", "A" * 100 + "\n\n" + "B" * 100)
        chunks = RecursiveChunker([content], chunk_size=10, overlap=5).recursive_chunker()
        assert chunks
        assert all(len(chunk.chunk) <= 10 for chunk in chunks)

    def test_does_not_invent_a_trailing_separator(self, normalizer):
        text = "alpha beta. gamma delta. epsilon zeta theta iota kappa lambda mu."
        content = normalizer.normalize_text("s.txt", text)
        chunks = RecursiveChunker([content], chunk_size=30, overlap=0).recursive_chunker()
        assert "".join(chunk.chunk for chunk in chunks) == content.content

    def test_offsets_locate_the_chunk_in_the_document(self, normalizer):
        content = normalizer.normalize_text("s.txt", "word " * 200)
        for chunk in RecursiveChunker(
            [content], chunk_size=40, overlap=8
        ).recursive_chunker():
            assert content.content[chunk.start_off_set : chunk.end_off_set] == chunk.chunk

    def test_overlap_is_applied_once(self, normalizer):
        """Overlap used to run at every level of the recursion, so text near a
        boundary was copied into a chunk several times over."""
        content = normalizer.normalize_text("s.txt", "word " * 400)
        chunks = RecursiveChunker(
            [content], chunk_size=50, overlap=10
        ).recursive_chunker()
        assert all(len(chunk.chunk) <= 50 for chunk in chunks)
        total = sum(len(chunk.chunk) for chunk in chunks)
        assert total < len(content.content) * 1.5

    def test_overlap_actually_overlaps(self, normalizer):
        content = normalizer.normalize_text("s.txt", "word " * 100)
        chunks = RecursiveChunker(
            [content], chunk_size=40, overlap=10
        ).recursive_chunker()
        assert any(
            later.start_off_set < earlier.end_off_set
            for earlier, later in zip(chunks, chunks[1:])
        )

    def test_repeated_text_gets_distinct_ids(self, normalizer):
        content = normalizer.normalize_text("s.txt", "same text. same text. same text.")
        chunks = RecursiveChunker([content], chunk_size=12, overlap=0).recursive_chunker()
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    @pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 20)])
    def test_rejects_impossible_geometry(self, size, overlap):
        with pytest.raises(ValueError):
            RecursiveChunker([], size, overlap)

    def test_persists_when_given_a_db(self, normalizer, tmp_path):
        db = str(tmp_path / "chunks.db")
        content = normalizer.normalize_text("s.txt", "word " * 200)
        chunks = RecursiveChunker(
            [content], chunk_size=40, overlap=8, db_path=db
        ).recursive_chunker()
        connection = sqlite3.connect(db)
        stored = connection.execute("select count(*) from RecursiveChunks").fetchone()[0]
        connection.close()
        assert stored == len(chunks)


class TestHierarchicalChunker:
    def test_uses_the_sections_the_normalizer_found(self, sectioned, tmp_path):
        db = str(tmp_path / "chunks.db")
        chunks = HierarchicalChunker(10, 60, db, [sectioned]).process_doc()
        assert {chunk.meta_data.section_name for chunk in chunks} == {
            "INTRODUCTION",
            "METHODS",
            "NOTES",
        }

    def test_preamble_is_kept(self, normalizer, tmp_path):
        content = normalizer.normalize_text(
            "s.txt", "This is a preamble.\n\nMY SECTION\n\nThis is content."
        )
        chunks = HierarchicalChunker(
            10, 50, str(tmp_path / "c.db"), [content]
        ).process_doc()
        text = " ".join(chunk.chunk for chunk in chunks)
        assert "This is a preamble" in text
        assert "This is content" in text

    def test_repeated_section_names_do_not_collide(self, normalizer, tmp_path):
        """sectionId hashed name + documentId alone, so a document with two
        "NOTES" headings aborted on the primary key."""
        db = str(tmp_path / "c.db")
        content = normalizer.normalize_text(
            "s.txt", "NOTES\n\naaa\n\nBODY\n\nbbb\n\nNOTES\n\nccc\n"
        )
        HierarchicalChunker(10, 60, db, [content]).process_doc()
        connection = sqlite3.connect(db)
        names = [
            row[0]
            for row in connection.execute(
                "select sectionName from Sections order by startoffset"
            )
        ]
        connection.close()
        assert names == ["NOTES", "BODY", "NOTES"]

    def test_repeated_paragraphs_do_not_collide(self, normalizer, tmp_path):
        content = normalizer.normalize_text(
            "s.txt", "SECTION A\n\nsame text\n\nsame text\n"
        )
        chunks = HierarchicalChunker(
            10, 200, str(tmp_path / "c.db"), [content]
        ).process_doc()
        assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    def test_offsets_locate_the_chunk_in_the_document(self, sectioned, tmp_path):
        chunks = HierarchicalChunker(
            10, 40, str(tmp_path / "c.db"), [sectioned]
        ).process_doc()
        for chunk in chunks:
            assert (
                sectioned.content[chunk.start_off_set : chunk.end_off_set] == chunk.chunk
            )

    def test_empty_input_touches_no_database(self, tmp_path):
        db = str(tmp_path / "unused.db")
        assert HierarchicalChunker(10, 60, db, []).process_doc() == []
        assert not os.path.exists(db)

    def test_connection_is_closed_when_a_write_fails(self, sectioned, tmp_path):
        """process_doc used to leak the SQLite connection on any error, because
        close() sat after the loop rather than in a finally."""
        opened = []
        real_manager = importlib.import_module(
            "data_layer.ingestion.Chunker.HierarchicalChunker"
        ).Manager

        def tracking_manager(*args, **kwargs):
            manager = real_manager(*args, **kwargs)
            opened.append(manager)
            return manager

        chunker = HierarchicalChunker(10, 60, str(tmp_path / "c.db"), [sectioned])
        with mock.patch.object(
            HierarchicalChunker, "_HierarchicalChunker__find_sections",
            side_effect=RuntimeError,
        ), mock.patch(
            "data_layer.ingestion.Chunker.HierarchicalChunker.Manager", tracking_manager
        ):
            with pytest.raises(RuntimeError):
                chunker.process_doc()

        assert opened
        with pytest.raises(sqlite3.ProgrammingError):
            opened[0].connection.execute("select 1")

    @pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, 20)])
    def test_rejects_impossible_geometry(self, size, overlap):
        with pytest.raises(ValueError):
            HierarchicalChunker(overlap, size, ":memory:", [])


class TestReIngestion:
    def test_the_same_folder_can_be_ingested_twice(self, sectioned, tmp_path):
        """documentId is a primary key, so a second run over unchanged files
        used to abort on a UNIQUE constraint."""
        chunker = Chunker(chunk_size=60, overlap=10, db_path=str(tmp_path / "c.db"))
        first, _ = chunker.chunk_per_document([sectioned])
        second, _ = chunker.chunk_per_document([sectioned])
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]

    def test_no_duplicate_rows_after_a_second_run(self, sectioned, tmp_path):
        db = str(tmp_path / "c.db")
        chunker = Chunker(chunk_size=60, overlap=10, db_path=db)
        chunks, _ = chunker.chunk_per_document([sectioned])
        chunker.chunk_per_document([sectioned])
        connection = sqlite3.connect(db)
        stored = connection.execute("select count(*) from Chunks").fetchone()[0]
        connection.close()
        assert stored == len(chunks)


class TestChunkerRouting:
    def test_sectioned_documents_go_hierarchical(self, sectioned, tmp_path):
        h, r = Chunker(60, 10, str(tmp_path / "c.db")).chunk_per_document([sectioned])
        assert h and not r

    def test_flat_documents_go_recursive(self, normalizer, tmp_path):
        flat = normalizer.normalize_text("s.txt", "Just prose. " * 40)
        h, r = Chunker(60, 10, str(tmp_path / "c.db")).chunk_per_document([flat])
        assert r and not h

    def test_no_database_is_created_for_an_empty_batch(self, tmp_path):
        db = str(tmp_path / "c.db")
        assert Chunker(60, 10, db).chunk_per_document([]) == ([], [])
        assert not os.path.exists(db)


class TestFileLoader:
    def test_accepts_unknown_extensions(self, tmp_path):
        for name in ["a.rs", "b.tex", "c.log", "d.yaml", "Makefile"]:
            (tmp_path / name).write_text("content")
        found = {
            path.name
            for paths in FileLoader().load_files(str(tmp_path)).values()
            for path in paths
        }
        assert found == {"a.rs", "b.tex", "c.log", "d.yaml", "Makefile"}

    def test_excludes_binaries(self, tmp_path):
        (tmp_path / "keep.txt").write_text("text")
        (tmp_path / "drop.png").write_bytes(b"\x89PNG")
        (tmp_path / "drop.pyc").write_bytes(b"\x00\x01")
        found = {
            path.name
            for paths in FileLoader().load_files(str(tmp_path)).values()
            for path in paths
        }
        assert found == {"keep.txt"}

    def test_symlink_loop_terminates(self, tmp_path):
        """A link back to an ancestor recursed until Python's stack limit, and
        the RecursionError was swallowed as if the folder were unreadable."""
        nested = tmp_path / "sub"
        nested.mkdir()
        (nested / "a.txt").write_text("text")
        (nested / "loop").symlink_to(tmp_path)
        found = {
            path.name
            for paths in FileLoader().load_files(str(tmp_path)).values()
            for path in paths
        }
        assert found == {"a.txt"}

    def test_skips_ignored_directories(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "junk.txt").write_text("junk")
        (tmp_path / "keep.txt").write_text("text")
        found = {
            path.name
            for paths in FileLoader().load_files(str(tmp_path)).values()
            for path in paths
        }
        assert found == {"keep.txt"}

    def test_reports_no_empty_categories(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        assert all(paths for paths in FileLoader().load_files(str(tmp_path)).values())

    def test_skips_oversized_files(self, tmp_path):
        (tmp_path / "big.txt").write_text("x" * 100)
        loader = FileLoader(max_file_size=10)
        assert loader.load_files(str(tmp_path)) == {}

    def test_allowlist_still_available(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.rs").write_text("code")
        loaded = FileLoader(allowed_extensions={".txt"}).load_files(str(tmp_path))
        assert list(loaded) == ["txt"]

    def test_rejects_a_non_directory(self, tmp_path):
        target = tmp_path / "a.txt"
        target.write_text("text")
        with pytest.raises(ValueError):
            FileLoader().load_files(str(target))


class TestTextExtractor:
    @pytest.fixture
    def extractor(self):
        return TextExtractor()

    @pytest.mark.parametrize(
        "name,body,needle",
        [
            ("s.rs", 'fn main() { println!("hi"); }', "println"),
            ("s.tex", "\\section{Intro}\nText.", "Intro"),
            ("s.log", "2026-01-01 ERROR boom", "ERROR"),
            ("s.yaml", "key: value\n", "key: value"),
            ("noext", "plain content", "plain content"),
        ],
    )
    def test_unknown_extensions_fall_back_to_text(
        self, extractor, tmp_path, name, body, needle
    ):
        target = tmp_path / name
        target.write_text(body)
        assert needle in extractor.extract_text_from_file(str(target))[1]

    def test_markup_is_stripped(self, extractor, tmp_path):
        target = tmp_path / "s.html"
        target.write_text(
            "<html><head><style>p{color:red}</style></head>"
            "<body><h1>Report</h1><p>Findings.</p></body></html>"
        )
        text = extractor.extract_text_from_file(str(target))[1]
        assert "Findings." in text
        assert "color:red" not in text
        assert "<p>" not in text

    def test_html_headings_become_markdown(self, extractor, tmp_path):
        """Otherwise an HTML document has no heading syntax the normalizer can
        see, and every page routes to the flat chunker."""
        target = tmp_path / "s.html"
        target.write_text("<h1>Report</h1><p>Findings.</p>")
        assert "# Report" in extractor.extract_text_from_file(str(target))[1]

    def test_json_is_flattened_with_its_keys(self, extractor, tmp_path):
        target = tmp_path / "s.json"
        target.write_text('{"title": "Report", "tags": ["a", "b"]}')
        text = extractor.extract_text_from_file(str(target))[1]
        assert "title: Report" in text
        assert "tags[0]: a" in text

    def test_invalid_json_falls_back_to_text(self, extractor, tmp_path):
        target = tmp_path / "s.json"
        target.write_text("{not json")
        assert extractor.extract_text_from_file(str(target))[1] == "{not json"

    def test_notebook_cells_are_extracted(self, extractor, tmp_path):
        target = tmp_path / "s.ipynb"
        target.write_text(
            '{"cells":[{"cell_type":"markdown","source":["# Title\\n"]},'
            '{"cell_type":"code","source":["print(1)"]}]}'
        )
        text = extractor.extract_text_from_file(str(target))[1]
        assert "# Title" in text
        assert "print(1)" in text

    def test_latin_1_is_decoded_as_latin_1(self, extractor, tmp_path):
        """utf-16 accepts almost any even-length input, so trying it without a
        byte order mark turned accented text into CJK."""
        target = tmp_path / "s.txt"
        target.write_bytes("café naïve".encode("latin-1"))
        assert extractor.extract_text_from_file(str(target))[1] == "café naïve"

    def test_utf_16_with_a_bom_is_decoded(self, extractor, tmp_path):
        target = tmp_path / "s.txt"
        target.write_bytes("héllo wörld".encode("utf-16"))
        assert extractor.extract_text_from_file(str(target))[1] == "héllo wörld"

    def test_utf_8_bom_is_stripped(self, extractor, tmp_path):
        target = tmp_path / "s.txt"
        target.write_bytes("plain".encode("utf-8-sig"))
        assert extractor.extract_text_from_file(str(target))[1] == "plain"

    def test_binary_content_is_rejected(self, extractor, tmp_path):
        target = tmp_path / "s.unknown"
        target.write_bytes(b"\x00\x01\x02binary\x00")
        with pytest.raises(InvalidFileType):
            extractor.extract_text_from_file(str(target))

    def test_paths_are_returned_as_strings(self, extractor, tmp_path):
        """The key was a PosixPath, which the normalizer then had to guess at."""
        target = tmp_path / "s.txt"
        target.write_text("text")
        assert isinstance(extractor.extract_text_from_file(target)[0], str)
        assert all(isinstance(key, str) for key in extractor.extract_all({"txt": [target]}))

    def test_a_bad_file_does_not_abandon_the_batch(self, extractor, tmp_path):
        good = tmp_path / "good.txt"
        good.write_text("readable")
        bad = tmp_path / "bad.unknown"
        bad.write_bytes(b"\x00\x00\x00")
        extracted = extractor.extract_all({"x": [good, bad]})
        assert list(extracted.values()) == ["readable"]

    def test_pdf_pages_without_a_text_layer(self, extractor, tmp_path):
        """extract_text() returns None for image-only pages, which the join
        over page text used to raise a TypeError on."""
        target = tmp_path / "s.pdf"
        target.write_bytes(b"%PDF-1.4")
        page = mock.MagicMock()
        page.extract_text.return_value = None
        reader = mock.MagicMock()
        reader.pages = [page]
        with mock.patch.dict(
            sys.modules, {"PyPDF2": mock.MagicMock(PdfReader=lambda _: reader)}
        ):
            assert extractor.extract_text_from_file(str(target))[1] == ""


class TestVectorIdsSurviveRepeatedText:
    """A chunk id that is unique buys nothing if the vector id derived from it
    is not: colliding ids leave all but one of the chunks unreachable."""

    @pytest.fixture
    def embedder(self):
        with mock.patch(
            "data_layer.ingestion.embedding.EmbeddingManager.SentenceTransformer"
        ) as model:
            import numpy as np

            model.return_value.encode.side_effect = lambda text, **kw: (
                np.zeros(128, dtype=np.float32)
                if isinstance(text, str)
                else np.zeros((len(text), 128), dtype=np.float32)
            )
            from data_layer.ingestion.embedding.EmbeddingManager import EmbeddingManager

            yield EmbeddingManager()

    def test_repeated_paragraphs_get_distinct_vector_ids(
        self, normalizer, embedder, tmp_path
    ):
        content = normalizer.normalize_text(
            "s.txt", "SECTION A\n\nboilerplate\n\nSECTION B\n\nboilerplate\n"
        )
        h, r = Chunker(200, 20, str(tmp_path / "c.db")).chunk_per_document([content])
        embedded = embedder.embed(h + r)
        assert len({e.vector_id for e in embedded}) == len(embedded)
