class InvalidCursorException(Exception):
    def __init__(self, pointer_name: str, index_accessed: int) -> None:
        self.pointer_name = pointer_name
        self.index_accessed = index_accessed

        super().__init__(self.pointer_name, self.index_accessed)

    def __str__(self) -> str:
        return f"The {self.pointer_name} went out of bound. Tried to access index {self.index_accessed}"


class NullPointerException(Exception):
    def __init__(self, error_message):
        self.error_message = error_message
        super().__init__(error_message)

    def __str__(self) -> str:
        return self.error_message


class InvalidVectorDimension(Exception):
    def __init__(self, invalid_dimension, expected_dimension):
        self.expected_dimension = expected_dimension
        self.dimension = invalid_dimension
        super().__init__(self.dimension, self.expected_dimension)

    def __str__(self) -> str:
        return f"Got the dimension {self.dimension}. Expected the dimension {self.expected_dimension}"


class MisMatchCount(Exception):
    pass


class InvalidRole(Exception):
    def __init__(self, role, allowed_roles) -> None:
        self.role = role
        self.allowed_roles = allowed_roles
        super().__init__(self.role, self.allowed_roles)

    def __str__(self) -> str:
        allowed = ", ".join(sorted(self.allowed_roles))
        return f"Got the role {self.role!r}. Expected one of: {allowed}"


class EmptyTurnContent(Exception):
    def __init__(self, role) -> None:
        self.role = role
        super().__init__(self.role)

    def __str__(self) -> str:
        return f"The {self.role!r} turn carries no text. A turn must have content."


class InvalidVectorId(Exception):
    def __init__(self, vector_id, max_vector_id) -> None:
        self.vector_id = vector_id
        self.max_vector_id = max_vector_id
        super().__init__(self.vector_id, self.max_vector_id)

    def __str__(self) -> str:
        return (
            f"Got the vector id {self.vector_id!r}. Expected a whole number in "
            f"0..{self.max_vector_id} — the range a signed 64-bit column holds. "
            "Ids outside it wrap or overflow; mask them with Config.VECTOR_ID_MASK."
        )
