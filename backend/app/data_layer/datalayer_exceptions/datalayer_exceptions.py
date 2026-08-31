from numpy import uint32


class InvalidFileType(Exception):
    def __init__(self, file_extention) -> None:
        self.file_extention = file_extention
        super().__init__(file_extention)

    def __str__(self):
        return f"Unsupported file type : {self.file_extention}"


class VectorInsertionError(Exception):
    """A vector write failed, in either vector store.

    `cause` carries the driver exception separately from `vector_id`, which
    used to hold whichever of the two the raising site happened to have: the
    DiskANN driver passed an id, the pgvector repository passed the psycopg
    error, so anything reading the attribute got one or the other.
    """

    MAX_IDS_SHOWN = 5

    def __init__(self, vector_id, cause: BaseException | None = None) -> None:
        self.vector_id = vector_id
        self.cause = cause
        super().__init__(vector_id, cause)

    def __describe_ids(self) -> str:
        if isinstance(self.vector_id, (str, bytes)) or not hasattr(
            self.vector_id, "__len__"
        ):
            return str(self.vector_id)
        ids = list(self.vector_id)
        shown = ", ".join(str(vector_id) for vector_id in ids[: self.MAX_IDS_SHOWN])
        if len(ids) > self.MAX_IDS_SHOWN:
            return f"{shown}, ... ({len(ids)} ids)"
        return shown

    def __str__(self):
        message = f"An Error occured while inserting the vector : {self.__describe_ids()}"
        if self.cause is None:
            return message
        return f"{message}. Caused by {type(self.cause).__name__}: {self.cause}"


class IndexDirectoryDoesNotExists(Exception):
    def __init__(self, directory_name) -> None:
        self.directory_name = directory_name
        super().__init__(self.directory_name)

    def __str__(self):
        return f"The directory with the following name doesn't exist: {self.directory_name}"


class InsertionError(Exception):
    def __init__(self, error, tableName, id) -> None:
        self.error = error
        self.message = tableName
        self.id = id
        super().__init__(self.error)

    def __str__(self):
        return f"{self.error}:Error occured while inserting values in the following table {self.message} , for the id : {self.id}"


class InvalidEmbeddingArgument(Exception):
    def __init__(self, error_message) -> None:
        self.error_message = error_message
        super().__init__(error_message)

    def __str__(self):
        return self.error_message


class VectorNotFoundEror(Exception):
    def __init__(self, vector_id: uint32) -> None:
        self.vector_id = vector_id
        super().__init__(vector_id)

    def __str__(self):
        return f"No vectors found for the vector Id : {self.vector_id}"


class DuplicateVectorException(Exception):
    """`VectorRepository.insert` was given an id the project already stores.

    Distinct from VectorInsertionError so a caller can tell "already written"
    from "the write failed", which matters to the vectors-first snapshot path:
    the first needs no compensating delete, the second does. batch_insert does
    not raise it — that path is `on conflict do nothing` by design.
    """

    def __init__(self, vector_id: uint32) -> None:
        self.vector_id = vector_id
        super().__init__(vector_id)

    def __str__(self):
        return f"The vector with vector id {self.vector_id} , already exists"


class InvalidBatchSize(Exception):

    def __init__(self, error_message) -> None:
        self.error_message = error_message
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message


class InvalidVectorDimension(Exception):
    def __init__(self, passed_dimension: int, expected_dimension: int) -> None:
        self.passed_dimension = passed_dimension
        self.expected_dimension = expected_dimension
        super().__init__(self.passed_dimension, self.expected_dimension)

    def __str__(self) -> str:
        return f"Expected dimension {self.expected_dimension} , got {self.passed_dimension}"


class InvalidVectorID(Exception):
    """No `vector_meta_data` row exists for the requested id.

    The name says invalid, the condition is missing; it is kept because every
    `except InvalidVectorID` in the tree would break with it renamed.
    """

    def __init__(self, vectorID) -> None:
        self.vectorId = vectorID
        super().__init__(self.vectorId)

    def __str__(self) -> str:
        return f"No vector meta data found for the vector id : {self.vectorId}"


class InvalidColumnNameException(Exception):
    def __init__(self, columnName: str):
        self.columnName = columnName
        super().__init__(self.columnName)

    def __str__(self):
        return f"Got invalid column name : {self.columnName}"


class MissingDatabaseConfiguration(Exception):
    def __init__(self, missing_keys) -> None:
        self.missing_keys = list(missing_keys)
        super().__init__(self.missing_keys)

    def __str__(self) -> str:
        keys = ", ".join(self.missing_keys)
        return (
            f"Missing PostgreSQL settings: {keys}. Copy .env.example to .env and "
            "fill them in. They are not optional: psycopg substitutes libpq's "
            "defaults for anything left unset — including the OS username — and "
            "the connection then silently goes somewhere unintended."
        )
