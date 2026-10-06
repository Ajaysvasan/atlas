"""Which sources knowledge may be taken from.

An allowlist is only worth what its matching rule is worth, so most of this file
is the ways a URL can look trusted without being so: a lookalike domain, a
redirect off the list, plain http, a name that resolves back into this network.
"""

import pytest

from knowledge_sufficiency.ksv_exceptions import UntrustedSource
from knowledge_sufficiency.sources import (
    DEFAULT_SOURCES,
    TrustedSource,
    check_trusted,
    host_is_listed,
    is_trusted,
)

LISTED = (TrustedSource("python", "docs.python.org", "docs"),)


class TestWhatIsAccepted:
    def test_a_listed_host(self):
        assert is_trusted("https://docs.python.org/3/library/sqlite3.html", LISTED)

    def test_a_subdomain_of_a_listed_host(self):
        assert is_trusted("https://cdn.docs.python.org/x", LISTED)

    def test_the_case_of_the_host_does_not_matter(self):
        assert is_trusted("https://DOCS.PYTHON.ORG/x", LISTED)

    def test_a_trailing_dot_does_not_matter(self):
        """`docs.python.org.` is the same host written absolutely."""
        assert host_is_listed("docs.python.org.", LISTED)


class TestWhatIsRefused:
    def test_a_lookalike_suffix(self):
        """The reason matching is label-by-label and not `endswith`."""
        assert not is_trusted("https://docs.python.org.attacker.example/x", LISTED)

    def test_a_host_that_merely_contains_a_listed_one(self):
        assert not is_trusted("https://notdocs.python.org.evil.example/x", LISTED)

    def test_an_unlisted_host(self):
        assert not is_trusted("https://some-blog.example/post", LISTED)

    def test_plain_http(self):
        """Even a listed host: the content is rewritable in transit."""
        assert not is_trusted("http://docs.python.org/x", LISTED)

    @pytest.mark.parametrize("scheme", ["ftp", "file", "data", "gopher"])
    def test_other_schemes(self, scheme):
        assert not is_trusted(f"{scheme}://docs.python.org/x", LISTED)

    def test_a_url_with_no_host(self):
        assert not is_trusted("https:///just-a-path", LISTED)

    def test_an_empty_allowlist_accepts_nothing(self):
        assert not is_trusted("https://docs.python.org/x", ())


class TestItSaysWhy:
    def test_the_reason_names_the_scheme(self):
        with pytest.raises(UntrustedSource, match="https"):
            check_trusted("http://docs.python.org/x", LISTED)

    def test_the_reason_names_the_host(self):
        with pytest.raises(UntrustedSource, match="some-blog.example"):
            check_trusted("https://some-blog.example/x", LISTED)

    def test_the_exception_carries_the_url(self):
        with pytest.raises(UntrustedSource) as caught:
            check_trusted("https://some-blog.example/x", LISTED)
        assert caught.value.url == "https://some-blog.example/x"


class TestAddressesInsideThisNetwork:
    """A listed name whose DNS answer points inward would turn a fetch into a
    request against this machine or a cloud metadata service.
    """

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
    def test_loopback_is_refused_even_when_listed(self, host):
        listed = (TrustedSource("local", host, "deliberately listed"),)
        with pytest.raises(UntrustedSource, match="inside this network"):
            check_trusted(f"https://{host}/secret", listed)

    def test_a_private_address_is_refused_even_when_listed(self):
        listed = (TrustedSource("lan", "10.0.0.1", "deliberately listed"),)
        with pytest.raises(UntrustedSource, match="inside this network"):
            check_trusted("https://10.0.0.1/secret", listed)

    def test_the_cloud_metadata_address_is_refused(self):
        listed = (TrustedSource("meta", "169.254.169.254", "deliberately listed"),)
        with pytest.raises(UntrustedSource, match="inside this network"):
            check_trusted("https://169.254.169.254/latest/meta-data/", listed)

    def test_the_check_can_be_waived_deliberately(self):
        """For a local mirror. Off by default, and the caller has to say so."""
        listed = (TrustedSource("local", "127.0.0.1", "a mirror"),)
        assert check_trusted("https://127.0.0.1/x", listed, allow_private=True)


class TestTheShippedAllowlist:
    def test_every_entry_is_a_bare_host(self):
        """A scheme or path in an entry would make the label match meaningless."""
        for source in DEFAULT_SOURCES:
            assert "/" not in source.host
            assert ":" not in source.host

    def test_every_entry_is_described(self):
        """Adding one is deliberate, so each says what it is."""
        for source in DEFAULT_SOURCES:
            assert source.description.strip()

    def test_the_names_are_unique(self):
        names = [source.name for source in DEFAULT_SOURCES]
        assert len(names) == len(set(names))

    def test_nothing_general_purpose_is_listed(self):
        """Primary and standards sources only — not the open internet."""
        hosts = {source.host for source in DEFAULT_SOURCES}
        for host in ("google.com", "reddit.com", "medium.com", "stackoverflow.com",
                     "twitter.com", "x.com", "facebook.com", "quora.com"):
            assert host not in hosts
