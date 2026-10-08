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

class EmptyQueryException(Exception):
    def __init__(self) -> None:
        pass
    def __str__(self) -> str:
        return( "The entered query is empty. Enter a valid query")



class TopicNotFound(Exception):
    def __init__(self, topic) -> None:
        self.topic = topic
        super().__init__(self.topic)

    def __str__(self) -> str:
        return f"No active topic {self.topic!r}. It was never created, or it was soft deleted."


class TopicAlreadyExists(Exception):
    def __init__(self, topic) -> None:
        self.topic = topic
        super().__init__(self.topic)

    def __str__(self) -> str:
        return f"An active topic named {self.topic!r} already exists."


class InvalidIdentifier(Exception):
    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value
        super().__init__(self.name, self.value)

    def __str__(self) -> str:
        return (
            f"{self.name} must be a non-empty string, got {self.value!r}. "
            f"It scopes rows in the database, so an empty or missing value "
            f"would silently partition nothing."
        )


class InvalidSnapshotScope(Exception):
    def __init__(self, scope, allowed) -> None:
        self.scope = scope
        self.allowed = allowed
        super().__init__(self.scope, self.allowed)

    def __str__(self) -> str:
        allowed = ", ".join(sorted(self.allowed))
        return f"Got the scope {self.scope!r}. Expected one of: {allowed}"


class WrongSnapshotScope(Exception):
    def __init__(self, operation, required, actual) -> None:
        self.operation = operation
        self.required = required
        self.actual = actual
        super().__init__(self.operation, self.required, self.actual)

    def __str__(self) -> str:
        return (
            f"{self.operation} belongs to a {self.required!r} snapshot, but this "
            f"one is scoped to {self.actual!r}. The two keep their history in "
            f"different stores, so the call cannot be served."
        )


class NewerMemorySchema(Exception):
    def __init__(self, path, found, supported) -> None:
        self.path = path
        self.found = found
        self.supported = supported
        super().__init__(self.path, self.found, self.supported)

    def __str__(self) -> str:
        return (
            f"{self.path} is at memory schema {self.found}, but this code only "
            f"knows up to {self.supported}. It was written by a newer version; "
            f"opening it here could corrupt it."
        )


class ProjectNotFound(Exception):
    def __init__(self, project_id) -> None:
        self.project_id = project_id
        super().__init__(self.project_id)

    def __str__(self) -> str:
        return (
            f"No project {self.project_id!r} is registered. Register it through "
            f"ProjectMetaData.add_project_vector before writing anything that "
            f"belongs to it."
        )


class ProjectInAnotherTopic(Exception):
    def __init__(self, project_id, topic_id, actual_topic_id) -> None:
        self.project_id = project_id
        self.topic_id = topic_id
        self.actual_topic_id = actual_topic_id
        super().__init__(self.project_id, self.topic_id, self.actual_topic_id)

    def __str__(self) -> str:
        return (
            f"Project {self.project_id!r} is under topic {self.actual_topic_id!r}, "
            f"not {self.topic_id!r}. Re-register it under {self.topic_id!r} to "
            f"move it; its rows follow."
        )
