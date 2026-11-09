"""Exceptions raised while acquiring knowledge."""

from data_layer.datalayer_exceptions.datalayer_exceptions import IngestionQuotaExceeded


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


class AcquisitionLimitReached(IngestionQuotaExceeded):
    """Storing more acquired material would pass MAX_ACQUISITION_STORAGE_BYTES."""

    def __init__(self, used: int, requested: int, limit: int) -> None:
        super().__init__(used, requested, limit)
        self.args = (f"acquired material needs ~{requested} more bytes; "
                     f"{used} of the {limit}-byte acquisition limit are already used",)

    def __str__(self) -> str:
        return self.args[0]
