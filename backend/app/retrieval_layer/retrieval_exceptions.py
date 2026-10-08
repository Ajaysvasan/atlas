"""Exceptions raised while retrieving."""


class EmptyQuery(Exception):
    def __init__(self, value) -> None:
        self.value = value
        super().__init__(self.value)

    def __str__(self) -> str:
        return (
            f"A retrieval needs a query, got {self.value!r}. Nothing can be "
            f"ranked against an empty one."
        )


class IndexUnavailable(Exception):
    def __init__(self, path, reason) -> None:
        self.path = path
        self.reason = reason
        super().__init__(self.path, self.reason)

    def __str__(self) -> str:
        return (
            f"The vector index could not be built from {self.path!r}: "
            f"{self.reason}. Ingest a corpus first."
        )


class RerankerUnavailable(Exception):
    def __init__(self, model, reason) -> None:
        self.model = model
        self.reason = reason
        super().__init__(self.model, self.reason)

    def __str__(self) -> str:
        return (
            f"The reranking model {self.model!r} could not be loaded: "
            f"{self.reason}. Retrieval still works without it; set "
            f"rerank=False to say so deliberately."
        )


class InvalidRetrievalSetting(Exception):
    def __init__(self, name, value, expected) -> None:
        self.name = name
        self.value = value
        self.expected = expected
        super().__init__(self.name, self.value, self.expected)

    def __str__(self) -> str:
        return f"{self.name} must be {self.expected}, got {self.value!r}"
