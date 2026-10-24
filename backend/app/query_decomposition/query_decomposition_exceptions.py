"""Exceptions raised while decomposing a query."""


class EmptyQuery(Exception):
    def __init__(self, value) -> None:
        self.value = value
        super().__init__(self.value)

    def __str__(self) -> str:
        return f"A query is needed to decompose, got {self.value!r}."


class QueryTooLarge(Exception):
    def __init__(self, estimated_tokens: int, limit: int) -> None:
        self.estimated_tokens = estimated_tokens
        self.limit = limit
        super().__init__(self.estimated_tokens, self.limit)

    def __str__(self) -> str:
        return (
            f"The query is about {self.estimated_tokens} tokens; at most "
            f"{self.limit} can be decomposed. Shorten it or split it into "
            f"several queries."
        )


class SystemPromptMissing(Exception):
    def __init__(self, path, key: str) -> None:
        self.path = path
        self.key = key
        super().__init__(self.path, self.key)

    def __str__(self) -> str:
        return (
            f"No system prompt {self.key!r} in {self.path}. Every prompt the "
            f"query decomposition layer uses must be a non-empty string there."
        )


class UnusableReply(ValueError):
    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(self.stage, self.reason)

    def __str__(self) -> str:
        return f"The draft model's {self.stage} reply could not be used: {self.reason}"


class ExchangeNotFound(Exception):
    def __init__(self, exchange_id: str) -> None:
        self.exchange_id = exchange_id
        super().__init__(self.exchange_id)

    def __str__(self) -> str:
        return f"No query was recorded as exchange {self.exchange_id!r}."


class ResponseAlreadyRecorded(Exception):
    def __init__(self, exchange_id: str) -> None:
        self.exchange_id = exchange_id
        super().__init__(self.exchange_id)

    def __str__(self) -> str:
        return (
            f"Exchange {self.exchange_id!r} already has a system response; "
            f"a response is recorded once."
        )
