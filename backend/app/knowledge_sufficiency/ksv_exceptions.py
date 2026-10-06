"""Exceptions raised while acquiring knowledge."""


class UntrustedSource(Exception):
    def __init__(self, url, reason) -> None:
        self.url = url
        self.reason = reason
        super().__init__(self.url, self.reason)

    def __str__(self) -> str:
        return f"Refused {self.url!r}: {self.reason}"


class FetchFailed(Exception):
    def __init__(self, url, reason) -> None:
        self.url = url
        self.reason = reason
        super().__init__(self.url, self.reason)

    def __str__(self) -> str:
        return f"Could not fetch {self.url!r}: {self.reason}"
