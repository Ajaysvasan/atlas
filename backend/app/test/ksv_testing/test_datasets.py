"""Datasets as a source, where the prose caps do not apply.

A documentation page is bounded by MAX_BYTES. A dataset is not: it can be
hundreds of gigabytes, and it carries a licence that may forbid the use being
made of it. Those two questions are what this file is about.
"""

import json

import numpy as np
import pytest

from knowledge_sufficiency.datasets import (
    MAX_DATASET_BYTES,
    PERMISSIVE_LICENCES,
    DatasetCandidate,
    find_datasets,
    parse_candidates,
    screen,
)
from knowledge_sufficiency.fetching import Document
from knowledge_sufficiency.ksv_exceptions import FetchFailed
from knowledge_sufficiency.sources import DATASET, DOCS, TrustedSource
from knowledge_sufficiency.target import AcquisitionTarget

HF = TrustedSource(
    "hf", "huggingface.co", "datasets",
    "https://huggingface.co/api/datasets?search={terms}", DATASET,
)
DOCS_SOURCE = TrustedSource("sqlite", "sqlite.org", "docs", "https://sqlite.org/s?q={terms}")


def target():
    v = np.ones(128, dtype=np.float32)
    return AcquisitionTarget(topic="sentiment", query="labelled reviews",
                             topic_vector=v, query_vector=v)


def candidate(name="a/b", licence="mit", size=1000):
    return DatasetCandidate(name, f"https://huggingface.co/datasets/{name}",
                            licence, size, "a description")


def serving(payload):
    return lambda url: Document(url=url, text=payload, content_type="application/json")


class TestScreeningOnLicence:
    @pytest.mark.parametrize("licence", sorted(PERMISSIVE_LICENCES)[:5])
    def test_a_permissive_licence_passes(self, licence):
        assert screen([candidate(licence=licence)]).accepted

    def test_a_restrictive_licence_is_refused(self):
        result = screen([candidate(licence="cc-by-nc-4.0")])
        assert result.accepted == []
        assert "cc-by-nc" in result.rejected[0][1]

    def test_an_absent_licence_is_refused_not_assumed_open(self):
        """The cost of being wrong is redistributing work without the right to."""
        result = screen([candidate(licence="")])
        assert result.accepted == []
        assert "no licence" in result.rejected[0][1]

    def test_case_and_spacing_do_not_matter(self):
        assert screen([candidate(licence="  MIT  ")]).accepted

    def test_the_licence_check_can_be_waived_deliberately(self):
        assert screen([candidate(licence="")], require_licence=False).accepted


class TestScreeningOnSize:
    def test_a_dataset_over_the_cap_is_refused(self):
        result = screen([candidate(size=MAX_DATASET_BYTES + 1)])
        assert result.accepted == []
        assert "over the cap" in result.rejected[0][1]

    def test_a_dataset_under_the_cap_passes(self):
        assert screen([candidate(size=MAX_DATASET_BYTES - 1)]).accepted

    def test_an_unknown_size_is_refused(self):
        """It cannot be bounded in advance, and no prose cap applies to it."""
        result = screen([candidate(size=None)])
        assert result.accepted == []
        assert "size unknown" in result.rejected[0][1]

    def test_the_cap_can_be_lowered(self):
        assert screen([candidate(size=5000)], max_bytes=1000).accepted == []

    def test_the_default_cap_is_finite_and_large(self):
        assert 0 < MAX_DATASET_BYTES <= 10 * 1024**3


class TestReadingAListing:
    def test_entries_become_candidates(self):
        payload = json.dumps([
            {"id": "org/reviews", "cardData": {"license": "mit"},
             "size_bytes": 1000},
        ])
        found = parse_candidates(payload, HF)
        assert found[0].name == "org/reviews"
        assert found[0].licence == "mit"

    def test_the_url_is_built_on_the_trusted_host(self):
        payload = json.dumps([{"id": "org/x", "size_bytes": 1}])
        assert parse_candidates(payload, HF)[0].url.startswith("https://huggingface.co/")

    def test_malformed_json_yields_nothing_rather_than_raising(self):
        assert parse_candidates("not json at all", HF) == []

    def test_entries_without_a_name_are_skipped(self):
        payload = json.dumps([{"cardData": {"license": "mit"}}, {"id": "ok"}])
        assert [c.name for c in parse_candidates(payload, HF)] == ["ok"]

    def test_a_wrapped_listing_is_read(self):
        payload = json.dumps({"datasets": [{"id": "org/x"}]})
        assert len(parse_candidates(payload, HF)) == 1

    def test_a_missing_licence_reads_as_empty_not_as_an_error(self):
        payload = json.dumps([{"id": "org/x"}])
        assert parse_candidates(payload, HF)[0].licence == ""


class TestFinding:
    def test_only_dataset_sources_are_searched(self):
        asked = []

        def fetch(url):
            asked.append(url)
            return Document(url=url, text="[]", content_type="application/json")

        find_datasets(target(), [HF, DOCS_SOURCE], fetch)
        assert len(asked) == 1
        assert "huggingface" in asked[0]

    def test_screening_is_applied_to_what_is_found(self):
        payload = json.dumps([
            {"id": "good", "cardData": {"license": "mit"}, "size_bytes": 10},
            {"id": "unlicensed", "size_bytes": 10},
            {"id": "huge", "cardData": {"license": "mit"},
             "size_bytes": MAX_DATASET_BYTES * 2},
        ])
        result = find_datasets(target(), [HF], serving(payload))
        assert [c.name for c in result.accepted] == ["good"]
        assert len(result.rejected) == 2

    def test_a_failing_search_is_survived(self):
        def fetch(url):
            raise FetchFailed(url, "HTTP 503")

        assert find_datasets(target(), [HF], fetch).accepted == []

    def test_it_returns_references_not_payloads(self):
        """Downloading and ingesting is the data layer's pipeline, not this one."""
        payload = json.dumps([{"id": "org/x", "cardData": {"license": "mit"},
                               "size_bytes": 10}])
        accepted = find_datasets(target(), [HF], serving(payload)).accepted
        assert accepted[0].url
        assert not hasattr(accepted[0], "rows")
